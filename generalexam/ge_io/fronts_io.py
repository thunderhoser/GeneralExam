"""IO methods for warm and cold fronts."""

import pickle
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

FRONT_TYPE_COLUMN = 'front_type'
TIME_COLUMN = 'unix_time_sec'
LATITUDES_COLUMN = 'latitudes_deg'
LONGITUDES_COLUMN = 'longitudes_deg'
REQUIRED_COLUMNS = [
    FRONT_TYPE_COLUMN, TIME_COLUMN, LATITUDES_COLUMN, LONGITUDES_COLUMN]

WARM_FRONT_TYPE = 'warm'
COLD_FRONT_TYPE = 'cold'
VALID_FRONT_TYPES = [WARM_FRONT_TYPE, COLD_FRONT_TYPE]


def check_front_type(front_type):
    """Ensures that front type is valid.

    :param front_type: Front type (string).
    :raises: ValueError: if front type is unrecognized.
    """

    error_checking.assert_is_string(front_type)
    if front_type not in VALID_FRONT_TYPES:
        error_string = (
            '\n\n' + str(VALID_FRONT_TYPES) +
            '\n\nValid front types (listed above) do not include "' +
            front_type + '".')
        raise ValueError(error_string)


def write_file(front_table, pickle_file_name):
    """Writes locations of one or more fronts to Pickle file.

    :param front_table: pandas DataFrame with the following columns.  Each row
        is one front.
    front_table.front_type: Type of front (examples: "warm", "cold").
    front_table.unix_time_sec: Valid time.
    front_table.latitudes_deg: 1-D numpy array of latitudes (deg N) along front.
    front_table.longitudes_deg: 1-D numpy array of longitudes (deg E) along
        front.
    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_columns_in_dataframe(front_table, REQUIRED_COLUMNS)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(front_table[REQUIRED_COLUMNS], pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads locations of one or more fronts from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: front_table: See documentation for `write_file`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    front_table = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(front_table, REQUIRED_COLUMNS)
    return front_table
