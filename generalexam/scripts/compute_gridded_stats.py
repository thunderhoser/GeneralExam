"""Computes statistics for gridded front properties."""

import argparse
import numpy
from generalexam.ge_utils import climatology_utils as climo_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_property_dir_name'
HOURS_ARG_NAME = 'hours_to_keep'
MONTHS_ARG_NAME = 'months_to_keep'
OUTPUT_DIR_ARG_NAME = 'output_statistic_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`climatology_utils.find_many_property_files` and read by '
    '`climatology_utils.average_many_property_files`.')

HOURS_HELP_STRING = (
    'List of UTC hours (integers in 0...23).  This script will compute stats '
    'only for the given hours.  If you want do not want to filter by hour, '
    'leave this argument alone.')

MONTHS_HELP_STRING = (
    'List of months (integers in 1...12).  This script will compute stats only '
    'for the given months.  If you want do not want to filter by month, leave '
    'this argument alone.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  File will be written by '
    '`climatology_utils.write_gridded_stats`, to a location therein determined '
    'by `climatology_utils.find_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + HOURS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=HOURS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MONTHS_ARG_NAME, type=int, nargs='+', required=False, default=[-1],
    help=MONTHS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(input_dir_name, hours_to_keep, months_to_keep, output_dir_name):
    """Computes statistics for gridded front properties.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param hours_to_keep: Same.
    :param months_to_keep: Same.
    :param output_dir_name: Same.
    """

    if len(hours_to_keep) == 1 and hours_to_keep[0] == -1:
        hours_to_keep = None

    if len(months_to_keep) == 1 and months_to_keep[0] == -1:
        months_to_keep = None

    property_file_names = climo_utils.find_many_property_files(
        directory_name=input_dir_name, hours=hours_to_keep,
        months=months_to_keep, raise_error_if_none_found=True)

    front_statistic_dict = climo_utils.average_many_property_files(
        property_file_names)
    print(SEPARATOR_STRING)

    output_file_name = climo_utils.find_file(
        directory_name=output_dir_name,
        file_type_string=climo_utils.FRONT_STATS_STRING,
        first_time_unix_sec=front_statistic_dict[climo_utils.FIRST_TIME_KEY],
        last_time_unix_sec=front_statistic_dict[climo_utils.LAST_TIME_KEY],
        hours=hours_to_keep, months=months_to_keep,
        raise_error_if_missing=False)

    print(front_statistic_dict[climo_utils.MEAN_WF_LENGTHS_KEY].shape)

    print('Writing results to: "{0:s}"...'.format(output_file_name))

    climo_utils.write_gridded_stats(
        netcdf_file_name=output_file_name,
        mean_wf_length_matrix_metres=front_statistic_dict[
            climo_utils.MEAN_WF_LENGTHS_KEY],
        mean_wf_area_matrix_m2=front_statistic_dict[
            climo_utils.MEAN_WF_AREAS_KEY],
        mean_cf_length_matrix_metres=front_statistic_dict[
            climo_utils.MEAN_CF_LENGTHS_KEY],
        mean_cf_area_matrix_m2=front_statistic_dict[
            climo_utils.MEAN_CF_AREAS_KEY],
        first_time_unix_sec=front_statistic_dict[climo_utils.FIRST_TIME_KEY],
        last_time_unix_sec=front_statistic_dict[climo_utils.LAST_TIME_KEY],
        prediction_file_names=front_statistic_dict[
            climo_utils.PREDICTION_FILES_KEY],
        hours=hours_to_keep, months=months_to_keep
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        hours_to_keep=numpy.array(
            getattr(INPUT_ARG_OBJECT, HOURS_ARG_NAME), dtype=int
        ),
        months_to_keep=numpy.array(
            getattr(INPUT_ARG_OBJECT, MONTHS_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
