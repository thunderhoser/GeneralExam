"""Converts gridded NFA (numerical frontal analysis) predictions to objects."""

import os.path
import argparse
import numpy
import pandas
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import nfa
from generalexam.ge_utils import front_utils
from generalexam.evaluation import object_based_evaluation as object_eval

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES2_TO_KM2 = 1e-6
INPUT_TIME_FORMAT = '%Y%m%d%H'
NARR_TIME_INTERVAL_SECONDS = 10800

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RANDOMIZE_TIMES_ARG_NAME = 'randomize_times'
NUM_TIMES_ARG_NAME = 'num_times'
MIN_AREA_ARG_NAME = 'min_object_area_metres2'
MIN_LENGTH_ARG_NAME = 'min_endpoint_length_metres'
FRONT_LINE_DIR_ARG_NAME = 'input_front_line_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of input directory, containing gridded predictions.  Files therein '
    'will be found by `nfa.find_gridded_prediction_file` and read by '
    '`nfa.read_gridded_predictions`.')

TIME_HELP_STRING = (
    'Input time (format "yyyymmddHH").  Times will be randomly drawn from the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of input times (to be drawn randomly from `{0:s}`...`{1:s}`).'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

MIN_AREA_HELP_STRING = (
    'Minimum area for predicted frontal region (before skeletonization).  '
    'Smaller regions will be thrown out.')

MIN_LENGTH_HELP_STRING = (
    'Minimum end-to-end length for skeleton line (predicted frontal polyline).'
    '  Shorter lines will be thrown out.')

FRONT_LINE_DIR_HELP_STRING = (
    'Name of top-level directory with actual fronts (polylines).  Files therein'
    ' will be found by `fronts_io.find_file_for_one_time` and read by '
    '`fronts_io.read_polylines_from_file`.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Actual and predicted fronts (polylines) will be '
    'written here by `object_based_evaluation.write_predictions_and_obs`.')

DEFAULT_MIN_AREA_METRES2 = 1e10  # 10 000 km^2
DEFAULT_MIN_LENGTH_METRES = 5e5  # 500 km
TOP_FRONT_LINE_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/fronts/polylines'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=True,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_AREA_ARG_NAME, type=float, required=False,
    default=DEFAULT_MIN_AREA_METRES2, help=MIN_AREA_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LENGTH_ARG_NAME, type=float, required=False,
    default=DEFAULT_MIN_LENGTH_METRES, help=MIN_LENGTH_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_LINE_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_LINE_DIR_NAME_DEFAULT, help=FRONT_LINE_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _read_actual_polylines(
        top_input_dir_name, unix_times_sec, narr_mask_matrix):
    """Reads actual fronts (polylines) for each time step.

    :param top_input_dir_name: See documentation at top of file.
    :param unix_times_sec: 1-D numpy array of valid times.
    :param narr_mask_matrix: See doc for
        `front_utils.remove_polylines_in_masked_area`.
    :return: polyline_table: See doc for `fronts_io.write_polylines_to_file`.
    """

    list_of_polyline_tables = []
    for this_time_unix_sec in unix_times_sec:
        this_file_name = fronts_io.find_file_for_one_time(
            top_directory_name=top_input_dir_name,
            file_type=fronts_io.POLYLINE_FILE_TYPE,
            valid_time_unix_sec=this_time_unix_sec)

        print 'Reading data from: "{0:s}"...'.format(this_file_name)
        list_of_polyline_tables.append(
            fronts_io.read_polylines_from_file(this_file_name)
        )
        if len(list_of_polyline_tables) == 1:
            continue

        list_of_polyline_tables[-1] = list_of_polyline_tables[-1].align(
            list_of_polyline_tables[0], axis=1)[0]

    polyline_table = pandas.concat(
        list_of_polyline_tables, axis=0, ignore_index=True)

    print 'Removing fronts in masked area...'
    return front_utils.remove_polylines_in_masked_area(
        polyline_table=polyline_table, narr_mask_matrix=narr_mask_matrix)


def _run(input_prediction_dir_name, first_time_string, last_time_string,
         num_times, min_object_area_metres2, min_endpoint_length_metres,
         top_front_line_dir_name, output_file_name):
    """Converts gridded NFA (numerical frontal analysis) predictions to objects.

    This is effectively the main method.

    :param input_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param min_object_area_metres2: Same.
    :param min_endpoint_length_metres: Same.
    :param top_front_line_dir_name: Same.
    :param output_file_name: Same.
    """

    grid_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME)[0]
    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    possible_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(possible_times_unix_sec)

    unix_times_sec = []
    list_of_predicted_region_tables = []
    num_times_done = 0
    narr_mask_matrix = None

    for i in range(len(possible_times_unix_sec)):
        if num_times_done == num_times:
            break

        this_prediction_file_name = nfa.find_gridded_prediction_file(
            directory_name=input_prediction_dir_name,
            first_valid_time_unix_sec=possible_times_unix_sec[i],
            last_valid_time_unix_sec=possible_times_unix_sec[i],
            raise_error_if_missing=False)
        if not os.path.isfile(this_prediction_file_name):
            continue

        num_times_done += 1
        unix_times_sec.append(possible_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_prediction_file_name)
        this_predicted_label_matrix = nfa.read_gridded_predictions(
            this_prediction_file_name)[0]

        print 'Converting image to frontal regions...'
        list_of_predicted_region_tables.append(
            object_eval.images_to_regions(
                predicted_label_matrix=this_predicted_label_matrix,
                image_times_unix_sec=possible_times_unix_sec[[i]])
        )

        print 'Throwing out frontal regions with area < {0:f} km^2...'.format(
            METRES2_TO_KM2 * min_object_area_metres2)
        list_of_predicted_region_tables[
            -1
        ] = object_eval.discard_regions_with_small_area(
            predicted_region_table=list_of_predicted_region_tables[-1],
            x_grid_spacing_metres=grid_spacing_metres,
            y_grid_spacing_metres=grid_spacing_metres,
            min_area_metres2=min_object_area_metres2)

        print 'Skeletonizing frontal regions...'
        list_of_predicted_region_tables[
            -1
        ] = object_eval.skeletonize_frontal_regions(
            predicted_region_table=list_of_predicted_region_tables[-1],
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

        list_of_predicted_region_tables[-1] = object_eval.find_main_skeletons(
            predicted_region_table=list_of_predicted_region_tables[-1],
            image_times_unix_sec=possible_times_unix_sec[[i]],
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns,
            x_grid_spacing_metres=grid_spacing_metres,
            y_grid_spacing_metres=grid_spacing_metres,
            min_endpoint_length_metres=min_endpoint_length_metres)

        if num_times_done != num_times:
            print '\n'

        if len(list_of_predicted_region_tables) == 1:
            continue

        list_of_predicted_region_tables[-1] = (
            list_of_predicted_region_tables[-1].align(
                list_of_predicted_region_tables[0], axis=1)[0]
        )

    print SEPARATOR_STRING

    unix_times_sec = numpy.array(unix_times_sec, dtype=int)
    predicted_region_table = pandas.concat(
        list_of_predicted_region_tables, axis=0, ignore_index=True)
    predicted_region_table = object_eval.convert_regions_rowcol_to_narr_xy(
        predicted_region_table=predicted_region_table,
        are_predictions_from_fcn=False)

    actual_polyline_table = _read_actual_polylines(
        top_input_dir_name=top_front_line_dir_name,
        unix_times_sec=unix_times_sec, narr_mask_matrix=narr_mask_matrix)
    print SEPARATOR_STRING

    actual_polyline_table = object_eval.project_polylines_latlng_to_narr(
        actual_polyline_table)

    print 'Writing predicted and observed objects to: "{0:s}"...'.format(
        output_file_name)
    object_eval.write_predictions_and_obs(
        predicted_region_table=predicted_region_table,
        actual_polyline_table=actual_polyline_table,
        pickle_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        min_object_area_metres2=getattr(INPUT_ARG_OBJECT, MIN_AREA_ARG_NAME),
        min_endpoint_length_metres=getattr(INPUT_ARG_OBJECT, MIN_LENGTH_ARG_NAME),
        top_front_line_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_LINE_DIR_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME))
