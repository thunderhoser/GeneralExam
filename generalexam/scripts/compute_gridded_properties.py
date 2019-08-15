"""Computes properties from gridded front labels."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.machine_learning import cnn

TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME
)[0]

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file` with '
    '`read_deterministic == True`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will compute WF and CF properties'
    ' at each grid cell for the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  File will be written by '
    '`climatology_utils.write_gridded_properties`, to a location therein '
    'determined by `climatology_utils.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _compute_properties_one_time(label_matrix, num_half_rows_for_cnn,
                                 num_half_columns_for_cnn):
    """Computes gridded properties for one time.

    M = number of rows in grid
    N = number of columns in grid

    :param label_matrix: M-by-N numpy array of integer front labels.
    :param num_half_rows_for_cnn: Number of half-rows in grids fed to CNN.
    :param num_half_columns_for_cnn: Number of half-columns in grids fed to CNN.
    :return: front_property_dict: Dictionary with the following keys.
    front_property_dict["wf_length_matrix_metres"]: M-by-N numpy array with
        warm-front length at each grid cell (NaN if there is no warm front).
    front_property_dict["wf_area_matrix_m2"]: Same but for warm-front area.
    front_property_dict["cf_length_matrix_metres"]: Same but for cold-front
        length.
    front_property_dict["cf_area_matrix_m2"]: Same but for cold-front area.
    """

    first_relevant_row = num_half_rows_for_cnn + 1
    last_relevant_row = label_matrix.shape[0] - num_half_rows_for_cnn - 2
    first_relevant_column = num_half_columns_for_cnn + 1
    last_relevant_column = label_matrix.shape[1] - num_half_columns_for_cnn - 2

    region_dict = front_utils.gridded_labels_to_regions(
        ternary_label_matrix=label_matrix, compute_lengths=True)

    this_matrix = numpy.full(label_matrix.shape, numpy.nan)

    front_property_dict = {
        climo_utils.WARM_FRONT_LENGTHS_KEY: this_matrix + 0.,
        climo_utils.WARM_FRONT_AREAS_KEY: this_matrix + 0.,
        climo_utils.COLD_FRONT_LENGTHS_KEY: this_matrix + 0.,
        climo_utils.COLD_FRONT_AREAS_KEY: this_matrix + 0.,
    }

    num_fronts = len(region_dict[front_utils.FRONT_TYPES_KEY])

    for k in range(num_fronts):
        these_rows = region_dict[front_utils.ROWS_BY_REGION_KEY][k]
        these_columns = region_dict[front_utils.COLUMNS_BY_REGION_KEY][k]

        if (
                numpy.any(these_rows < first_relevant_row) or
                numpy.any(these_rows > last_relevant_row) or
                numpy.any(these_columns < first_relevant_column) or
                numpy.any(these_columns > last_relevant_column)
        ):
            continue

        this_area_metres2 = len(these_rows) * GRID_SPACING_METRES ** 2
        this_length_metres = (
            GRID_SPACING_METRES *
            region_dict[front_utils.MAJOR_AXIS_LENGTHS_KEY][k]
        )

        this_front_type_string = region_dict[front_utils.FRONT_TYPES_KEY][k]

        if this_front_type_string == front_utils.WARM_FRONT_STRING:
            front_property_dict[climo_utils.WARM_FRONT_LENGTHS_KEY][
                these_rows, these_columns
            ] = this_length_metres

            front_property_dict[climo_utils.WARM_FRONT_AREAS_KEY][
                these_rows, these_columns
            ] = this_area_metres2
        else:
            front_property_dict[climo_utils.COLD_FRONT_LENGTHS_KEY][
                these_rows, these_columns
            ] = this_length_metres

            front_property_dict[climo_utils.COLD_FRONT_AREAS_KEY][
                these_rows, these_columns
            ] = this_area_metres2

    return front_property_dict


def _run(prediction_dir_name, first_time_string, last_time_string,
         output_dir_name):
    """Computes properties from gridded front labels.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT)

    prediction_file_names, valid_times_unix_sec = (
        prediction_io.find_files_for_climo(
            directory_name=prediction_dir_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec)
    )

    num_half_rows_for_cnn = None
    num_half_columns_for_cnn = None

    for i in range(len(prediction_file_names)):
        print('Reading deterministic labels from: "{0:s}"...'.format(
            prediction_file_names[i]
        ))

        this_prediction_dict = prediction_io.read_file(
            netcdf_file_name=prediction_file_names[i], read_deterministic=True
        )

        if num_half_rows_for_cnn is None:
            cnn_metafile_name = cnn.find_metafile(
                this_prediction_dict[prediction_io.MODEL_FILE_KEY]
            )

            print('Reading CNN metadata from: "{0:s}"...'.format(
                cnn_metafile_name))

            cnn_metadata_dict = cnn.read_metadata(cnn_metafile_name)
            num_half_rows_for_cnn = cnn_metadata_dict[cnn.NUM_HALF_ROWS_KEY]
            num_half_columns_for_cnn = cnn_metadata_dict[
                cnn.NUM_HALF_COLUMNS_KEY]

        this_property_dict = _compute_properties_one_time(
            label_matrix=this_prediction_dict[
                prediction_io.PREDICTED_LABELS_KEY][0, ...],
            num_half_rows_for_cnn=num_half_rows_for_cnn,
            num_half_columns_for_cnn=num_half_columns_for_cnn
        )

        this_output_file_name = climo_utils.find_basic_file(
            directory_name=output_dir_name,
            file_type_string=climo_utils.FRONT_PROPERTIES_STRING,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print('Writing front properties to: "{0:s}"...'.format(
            this_output_file_name))

        climo_utils.write_gridded_properties(
            netcdf_file_name=this_output_file_name,
            wf_length_matrix_metres=this_property_dict[
                climo_utils.WARM_FRONT_LENGTHS_KEY],
            wf_area_matrix_m2=this_property_dict[
                climo_utils.WARM_FRONT_AREAS_KEY],
            cf_length_matrix_metres=this_property_dict[
                climo_utils.COLD_FRONT_LENGTHS_KEY],
            cf_area_matrix_m2=this_property_dict[
                climo_utils.COLD_FRONT_AREAS_KEY],
            prediction_file_name=prediction_file_names[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
