"""IO methods for warm and cold fronts."""

import pickle
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

REQUIRED_POLYLINE_COLUMNS = [
    front_utils.FRONT_TYPE_COLUMN, front_utils.TIME_COLUMN,
    front_utils.LATITUDES_COLUMN, front_utils.LONGITUDES_COLUMN]

REQUIRED_GRID_COLUMNS = [
    front_utils.TIME_COLUMN, front_utils.WARM_FRONT_ROW_INDICES_COLUMN,
    front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN,
    front_utils.COLD_FRONT_ROW_INDICES_COLUMN,
    front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN]


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
