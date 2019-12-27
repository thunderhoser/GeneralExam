"""Runs neighbourhood evaluation and saves results for each grid point.

Keep in mind that neighbourhood evaluation requires deterministic labels, not
probabilities.
"""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io
from generalexam.machine_learning import cnn
from generalexam.ge_utils import neigh_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

TIME_INTERVAL_SECONDS = 10800
INPUT_TIME_FORMAT = '%Y%m%d%H'

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NEIGH_DISTANCES_ARG_NAME = 'neigh_distances_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of input directory.  Files with gridded deterministic labels will be '
    'found therein by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Neighbourhood evaluation will be done for all'
    ' times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NEIGH_DISTANCES_HELP_STRING = (
    'List of neighbourhood distances.  Evaluation will be done for each.'
)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files will be written here by '
    '`neigh_evaluation.write_spatial_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEIGH_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=NEIGH_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _handle_one_prediction_file(
        prediction_file_name, neigh_distances_metres, binary_ct_by_neigh,
        prediction_oriented_ct_by_neigh, actual_oriented_ct_by_neigh,
        training_mask_matrix=None):
    """Handles one prediction file.

    D = number of neighbourhood distances

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param neigh_distances_metres: length-D numpy array of distances for
        neighbourhood evaluation.
    :param binary_ct_by_neigh: length-D list of binary contingency tables in
        format produced by `neigh_evaluation.make_spatial_contingency_tables`.
    :param prediction_oriented_ct_by_neigh: length-D list of prediction-oriented
        contingency tables in format produced by
        `neigh_evaluation.make_spatial_contingency_tables`.
    :param actual_oriented_ct_by_neigh: length-D list of actual-oriented
        contingency tables in format produced by
        `neigh_evaluation.make_spatial_contingency_tables`.
    :param training_mask_matrix: See doc for
        `neigh_evaluation.make_spatial_contingency_tables`.  If this is None,
        will be read from CNN metadata on the fly.
    :return: binary_ct_by_neigh: Same as input but with different values.
    :return: prediction_oriented_ct_by_neigh: Same as input but with different
        values.
    :return: actual_oriented_ct_by_neigh: Same as input but with different
        values.
    :return: training_mask_matrix: Same as input but cannot be None.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(
        netcdf_file_name=prediction_file_name, read_deterministic=True
    )

    predicted_label_matrix = prediction_dict[prediction_io.PREDICTED_LABELS_KEY]
    actual_label_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]

    if training_mask_matrix is None:
        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = cnn.find_metafile(model_file_name)

        print('Reading training mask from: "{0:s}"...'.format(
            model_metafile_name
        ))
        model_metadata_dict = cnn.read_metadata(model_metafile_name)
        training_mask_matrix = model_metadata_dict[cnn.MASK_MATRIX_KEY]

    num_neigh_distances = len(neigh_distances_metres)

    for k in range(num_neigh_distances):
        this_binary_ct, this_prediction_oriented_ct, this_actual_oriented_ct = (
            neigh_evaluation.make_spatial_contingency_tables(
                predicted_label_matrix=predicted_label_matrix,
                actual_label_matrix=actual_label_matrix,
                neigh_distance_metres=neigh_distances_metres[k],
                training_mask_matrix=training_mask_matrix
            )
        )

        num_grid_rows = this_binary_ct.shape[0]
        num_grid_columns = this_binary_ct.shape[1]

        if binary_ct_by_neigh[k] is None:
            binary_ct_by_neigh[k] = copy.deepcopy(this_binary_ct)
            prediction_oriented_ct_by_neigh[k] = (
                this_prediction_oriented_ct + 0.
            )
            actual_oriented_ct_by_neigh[k] = this_actual_oriented_ct + 0.
        else:
            for i in range(num_grid_rows):
                for j in range(num_grid_columns):
                    for this_key in binary_ct_by_neigh[k][i, j]:
                        binary_ct_by_neigh[k][i, j][this_key] += (
                            this_binary_ct[i, j][this_key]
                        )

            prediction_oriented_ct_by_neigh[k] += this_prediction_oriented_ct
            actual_oriented_ct_by_neigh[k] += this_actual_oriented_ct

    return (
        binary_ct_by_neigh, prediction_oriented_ct_by_neigh,
        actual_oriented_ct_by_neigh, training_mask_matrix
    )


def _run(prediction_dir_name, first_time_string, last_time_string,
         neigh_distances_metres, output_dir_name):
    """Runs neighbourhood evaluation and saves results for each grid point.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param neigh_distances_metres: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT
    )
    all_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

    # Read predictions and create contingency tables.
    num_neigh_distances = len(neigh_distances_metres)
    binary_ct_by_neigh = [None] * num_neigh_distances
    prediction_oriented_ct_by_neigh = [None] * num_neigh_distances
    actual_oriented_ct_by_neigh = [None] * num_neigh_distances

    training_mask_matrix = None
    prediction_file_names = []

    for this_time_unix_sec in all_times_unix_sec:
        this_prediction_file_name = prediction_io.find_file(
            directory_name=prediction_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False)

        if not os.path.isfile(this_prediction_file_name):
            continue

        prediction_file_names.append(this_prediction_file_name)
        print(MINOR_SEPARATOR_STRING)

        (
            binary_ct_by_neigh, prediction_oriented_ct_by_neigh,
            actual_oriented_ct_by_neigh, training_mask_matrix
        ) = _handle_one_prediction_file(
            prediction_file_name=this_prediction_file_name,
            neigh_distances_metres=neigh_distances_metres,
            binary_ct_by_neigh=binary_ct_by_neigh,
            prediction_oriented_ct_by_neigh=prediction_oriented_ct_by_neigh,
            actual_oriented_ct_by_neigh=actual_oriented_ct_by_neigh,
            training_mask_matrix=training_mask_matrix
        )

    print(SEPARATOR_STRING)

    # good_rows, good_cols = numpy.where(
    #     prediction_oriented_ct_by_neigh[0][..., 1, 1] > 0
    # )
    # good_row = good_rows[0]
    # good_col = good_cols[0]
    #
    # print(binary_ct_by_neigh[0][good_row, good_col])
    # print(prediction_oriented_ct_by_neigh[0][good_row, good_col])
    # print(actual_oriented_ct_by_neigh[0][good_row, good_col])

    # Write results.
    for k in range(num_neigh_distances):
        this_output_file_name = (
            '{0:s}/spatial_evaluation_neigh-distance-metres={1:06d}.p'
        ).format(
            output_dir_name, int(numpy.round(neigh_distances_metres[k]))
        )

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))

        neigh_evaluation.write_spatial_results(
            pickle_file_name=this_output_file_name,
            prediction_file_names=prediction_file_names,
            neigh_distance_metres=neigh_distances_metres[k],
            binary_ct_dict_matrix=binary_ct_by_neigh[k],
            prediction_oriented_ct_matrix=prediction_oriented_ct_by_neigh[k],
            actual_oriented_ct_matrix=actual_oriented_ct_by_neigh[k]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        neigh_distances_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEIGH_DISTANCES_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
