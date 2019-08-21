"""Stitches together inner and outer grids of determinized CNN predictions."""

import argparse
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import prediction_io

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

INNER_GRID_DIR_ARG_NAME = 'inner_grid_dir_name'
OUTER_GRID_DIR_ARG_NAME = 'outer_grid_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INNER_GRID_DIR_HELP_STRING = (
    'Name of top-level directory with CNN predictions for inner grid.  Files '
    'therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.')

OUTER_GRID_DIR_HELP_STRING = 'Same as `{0:s}` but for outer grid.'.format(
    INNER_GRID_DIR_ARG_NAME)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will stitch grids for all times '
    'in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory (for stitched grid).  Results will be '
    'written here by `prediction_io.write_probabilities` and '
    '`prediction_io.append_deterministic_labels`, to exact locations determined'
    ' by `prediction_io.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INNER_GRID_DIR_ARG_NAME, type=str, required=True,
    help=INNER_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTER_GRID_DIR_ARG_NAME, type=str, required=True,
    help=OUTER_GRID_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(top_inner_grid_dir_name, top_outer_grid_dir_name, first_time_string,
         last_time_string, top_output_dir_name):
    """Stitches together inner and outer grids of determinized CNN predictions.

    This is effectively the main method.

    :param top_inner_grid_dir_name: See documentation at top of file.
    :param top_outer_grid_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param top_output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    for this_time_unix_sec in valid_times_unix_sec:
        this_inner_grid_file_name = prediction_io.find_file(
            directory_name=top_inner_grid_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec, raise_error_if_missing=True)

        this_outer_grid_file_name = prediction_io.find_file(
            directory_name=top_outer_grid_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec, raise_error_if_missing=True)

        print('Reading data from: "{0:s}"...'.format(this_inner_grid_file_name))
        this_inner_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_inner_grid_file_name, read_deterministic=True)

        print('Reading data from: "{0:s}"...'.format(
            this_outer_grid_file_name))
        this_outer_prediction_dict = prediction_io.read_file(
            netcdf_file_name=this_outer_grid_file_name,
            read_deterministic=True)

        this_predicted_label_matrix = this_outer_prediction_dict[
            prediction_io.PREDICTED_LABELS_KEY]
        this_predicted_label_matrix[:, 120:-120, 120:-120] = (
            this_inner_prediction_dict[prediction_io.PREDICTED_LABELS_KEY][
                :, 20:-20, 20:-20]
        )

        this_class_prob_matrix = this_outer_prediction_dict[
            prediction_io.CLASS_PROBABILITIES_KEY]
        this_class_prob_matrix[:, 120:-120, 120:-120, :] = (
            this_inner_prediction_dict[prediction_io.CLASS_PROBABILITIES_KEY][
                :, 20:-20, 20:-20, :]
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=top_output_dir_name,
            first_time_unix_sec=this_time_unix_sec,
            last_time_unix_sec=this_time_unix_sec, raise_error_if_missing=False)

        print('Writing stitched data to: "{0:s}"...\n'.format(
            this_output_file_name))

        prediction_io.write_probabilities(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_class_prob_matrix,
            valid_times_unix_sec=this_inner_prediction_dict[
                prediction_io.VALID_TIMES_KEY],
            model_file_name=this_inner_prediction_dict[
                prediction_io.MODEL_FILE_KEY],
            used_isotonic=this_inner_prediction_dict[
                prediction_io.USED_ISOTONIC_KEY]
        )

        prediction_io.append_deterministic_labels(
            probability_file_name=this_output_file_name,
            predicted_label_matrix=this_predicted_label_matrix,
            prob_threshold_by_class=this_inner_prediction_dict[
                prediction_io.THRESHOLDS_KEY],
            min_region_length_metres=this_inner_prediction_dict[
                prediction_io.MIN_REGION_LENGTH_KEY],
            region_buffer_distance_metres=this_inner_prediction_dict[
                prediction_io.REGION_BUFFER_DISTANCE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_inner_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, INNER_GRID_DIR_ARG_NAME),
        top_outer_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTER_GRID_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
