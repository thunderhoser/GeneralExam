"""IO methods for warm and cold fronts."""

import os.path
import pickle
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

TIME_FORMAT_MONTH = '%Y%m'
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

POLYLINE_FILE_TYPE = 'polylines'
GRIDDED_FILE_TYPE = 'narr_grids'
VALID_FILE_TYPES = [POLYLINE_FILE_TYPE, GRIDDED_FILE_TYPE]

PATHLESS_PREFIX_FOR_POLYLINE_FILES = 'front_locations'
PATHLESS_PREFIX_FOR_GRIDDED_FILES = 'narr_frontal_grids'

REQUIRED_POLYLINE_COLUMNS = [
    front_utils.FRONT_TYPE_COLUMN, front_utils.TIME_COLUMN,
    front_utils.LATITUDES_COLUMN, front_utils.LONGITUDES_COLUMN]

REQUIRED_GRID_COLUMNS = [
    front_utils.TIME_COLUMN, front_utils.WARM_FRONT_ROW_INDICES_COLUMN,
    front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN,
    front_utils.COLD_FRONT_ROW_INDICES_COLUMN,
    front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN]


def _check_file_type(file_type):
    """Ensures that proposed file type is valid.

    :param file_type: File type (must be in list `VALID_FILE_TYPES`).
    :raises: ValueError: if `file_type not in VALID_FILE_TYPES`.
    """

    error_checking.assert_is_string(file_type)
    if file_type not in VALID_FILE_TYPES:
        error_string = ('\n\n{0:s}\nValid file types (listed above) do not '
                        'include "{1:s}".').format(VALID_FILE_TYPES, file_type)
        raise ValueError(error_string)


def write_polylines_to_file(front_table, pickle_file_name):
    """Writes one or more frontal polylines to Pickle file.

    :param front_table: pandas DataFrame with the following columns.  Each row
        is one front.
    front_table.front_type: Type of front (examples: "warm", "cold").
    front_table.unix_time_sec: Valid time.
    front_table.latitudes_deg: 1-D numpy array of latitudes (deg N) along front.
    front_table.longitudes_deg: 1-D numpy array of longitudes (deg E) along
        front.
    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        front_table, REQUIRED_POLYLINE_COLUMNS)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(front_table[REQUIRED_POLYLINE_COLUMNS], pickle_file_handle)
    pickle_file_handle.close()


def read_polylines_from_file(pickle_file_name):
    """Reads one or more frontal polylines from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: front_table: See documentation for `write_polylines_to_file`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    front_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        front_table, REQUIRED_POLYLINE_COLUMNS)
    return front_table


def write_narr_grids_to_file(frontal_grid_table, pickle_file_name):
    """Writes one or more NARR* grids to file.

    * NARR = North American Regional Reanalysis

    :param frontal_grid_table: pandas DataFrame with the following columns.
        Each row is one valid time.
    frontal_grid_table.unix_time_sec: Valid time.
    frontal_grid_table.warm_front_row_indices: length-W numpy array with row
        indices (integers) of grid cells intersected by a warm front.
    frontal_grid_table.warm_front_column_indices: Same as above, except for
        columns.
    frontal_grid_table.cold_front_row_indices: length-C numpy array with row
        indices (integers) of grid cells intersected by a cold front.
    frontal_grid_table.cold_front_column_indices: Same as above, except for
        columns.

    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(
        frontal_grid_table, REQUIRED_GRID_COLUMNS)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(frontal_grid_table[REQUIRED_GRID_COLUMNS], pickle_file_handle)
    pickle_file_handle.close()


def read_narr_grids_from_file(pickle_file_name):
    """Reads one or more NARR* grids from file.

    * NARR = North American Regional Reanalysis

    :param pickle_file_name: Path to input file.
    :return: frontal_grid_table: See documentation for
        `write_narr_grids_to_file`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    frontal_grid_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        frontal_grid_table, REQUIRED_GRID_COLUMNS)
    return frontal_grid_table


def find_file_for_time_period(
        directory_name, file_type, start_time_unix_sec, end_time_unix_sec,
        raise_error_if_missing=True):
    """Finds file with fronts for a contiguous time period.

    Specifically, this file should contain EITHER polylines or NARR grids,
    defining warm and cold fronts, at all 3-hour time steps in the given period.

    :param directory_name: Name of directory.
    :param file_type: Type of file (either "polylines" or "narr_grids").
    :param start_time_unix_sec: Start of contiguous time period.
    :param end_time_unix_sec: End of contiguous time period.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: front_file_name: Path to file.
    """

    error_checking.assert_is_string(directory_name)
    _check_file_type(file_type)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if file_type == POLYLINE_FILE_TYPE:
        this_pathless_file_prefix = PATHLESS_PREFIX_FOR_POLYLINE_FILES
    else:
        this_pathless_file_prefix = PATHLESS_PREFIX_FOR_GRIDDED_FILES

    front_file_name = '{0:s}/{1:s}_{2:s}-{3:s}.p'.format(
        directory_name, this_pathless_file_prefix,
        time_conversion.unix_sec_to_string(
            start_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            end_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES))

    if raise_error_if_missing and not os.path.isfile(front_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                front_file_name))
        raise ValueError(error_string)

    return front_file_name


def find_file_for_one_time(
        top_directory_name, file_type, valid_time_unix_sec,
        raise_error_if_missing=True):
    """Finds file with fronts for a single time step.

    Specifically, this file should contain EITHER polylines or NARR grids,
    defining warm and cold fronts, at one time step.

    :param top_directory_name: Name of top-level directory with polyline files.
    :param file_type: Type of file (either "polylines" or "narr_grids").
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: See documentation for
        `find_file_for_time_period`.
    :return: front_file_name: Path to file.
    """

    error_checking.assert_is_string(top_directory_name)
    _check_file_type(file_type)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if file_type == POLYLINE_FILE_TYPE:
        this_pathless_file_prefix = PATHLESS_PREFIX_FOR_POLYLINE_FILES
    else:
        this_pathless_file_prefix = PATHLESS_PREFIX_FOR_GRIDDED_FILES

    front_file_name = '{0:s}/{1:s}/{2:s}_{3:s}.p'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_MONTH), this_pathless_file_prefix,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES))

    if raise_error_if_missing and not os.path.isfile(front_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                front_file_name))
        raise ValueError(error_string)

    return front_file_name
