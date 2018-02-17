"""IO methods for Weather Prediction Center (WPC) bulletins."""

import re
import os.path
import warnings
import numpy
import pandas
from generalexam.ge_utils import front_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

PATHLESS_FILE_NAME_PREFIX = 'KWBCCODSUS_HIRES'
TIME_FORMAT_IN_FILE_NAME = '%Y%m%d_%H00'
TIME_FORMAT_IN_FILE_ITSELF = '%H%'
VALID_FRONT_TYPES = [
    front_utils.WARM_FRONT_STRING_ID, front_utils.COLD_FRONT_STRING_ID]

LATLNG_STRING_PATTERN_5CHARS = '[0-9][0-9][0-1][0-9][0-9]'
LATLNG_STRING_PATTERN_7CHARS = '[0-9][0-9][0-9][0-1][0-9][0-9][0-9]'


def _file_name_to_valid_time(bulletin_file_name):
    """Parses valid time from file name.

    :param bulletin_file_name: Path to input file (text file in WPC format).
    :return: valid_time_unix_sec: Valid time.
    """

    _, pathless_file_name = os.path.split(bulletin_file_name)
    valid_time_string = pathless_file_name.replace(
        PATHLESS_FILE_NAME_PREFIX + '_', '')

    return time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT_IN_FILE_NAME)


def _string_to_latlng(latlng_string, raise_error_if_fails=True):
    """Converts string to latitude and longitude.

    The string must be formatted in one of two ways:

    - NNWWW, where the first 2 characters are latitude (deg N) and last 3
      characters are longitude (deg W).
    - NNNWWWW, where the first 3 characters are latitude (deg N) and last 4
      characters are longitude (deg W).  In this case, there is an implicit
      decimal before the last latitude character and before the last longitude
      character.

    :param latlng_string: Input string.
    :param raise_error_if_fails: Boolean flag.  If lat/long cannot be parsed
        from string and raise_error_if_fails = True, this method will error out.
        If lat/long cannot be parsed and raise_error_if_fails = False, this
        method will return NaN for all output variables.
    :return: latitude_deg: Latitude (deg N).
    :return: longitude_deg: Longitude (deg E).
    :raises: ValueError: if string does not match expected format.
    """

    if (re.match(LATLNG_STRING_PATTERN_7CHARS, latlng_string) is not None
            and len(latlng_string) == 7):
        latitude_deg = float(latlng_string[:3]) / 10
        longitude_deg = -float(latlng_string[3:]) / 10
        return latitude_deg, longitude_deg

    if (re.match(LATLNG_STRING_PATTERN_5CHARS, latlng_string) is not None
            and len(latlng_string) == 5):
        latitude_deg = float(latlng_string[:2])
        longitude_deg = -float(latlng_string[2:])
        return latitude_deg, longitude_deg

    error_string = (
        'Input string ("{0:s}") does not match expected format ("{1:s}" or '
        '"{2:s}").').format(latlng_string, LATLNG_STRING_PATTERN_7CHARS,
                            LATLNG_STRING_PATTERN_5CHARS)
    if raise_error_if_fails:
        raise ValueError(error_string)

    warnings.warn(error_string)
    return numpy.nan, numpy.nan


def find_file(
        valid_time_unix_sec, top_directory_name, raise_error_if_missing=True):
    """Finds file (text file in WPC format) on local machine.

    This file should contain positions of cyclones, anticyclones, fronts, etc.
    for a single valid time.

    :param valid_time_unix_sec: Valid time.
    :param top_directory_name: Name of top-level directory with WPC bulletins.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        raise_error_if_missing = True, this method will error out.  If file is
        missing and raise_error_if_missing = False, this method will return the
        *expected* path to the file.
    :return: bulletin_file_name: Path to file.  If file is missing and
        raise_error_if_missing = False, this is the *expected* path.
    :raises: ValueError: if file is missing and raise_error_if_missing = True.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAME)

    bulletin_file_name = '{0:s}/{1:s}/{2:s}_{3:s}'.format(
        top_directory_name, valid_time_string[:4], PATHLESS_FILE_NAME_PREFIX,
        valid_time_string)

    if raise_error_if_missing and not os.path.isfile(bulletin_file_name):
        error_string = (
            'Cannot find file.  Expected at location: "{0:s}"'.format(
                bulletin_file_name))
        raise ValueError(error_string)

    return bulletin_file_name


def read_fronts_from_file(text_file_name):
    """Reads locations of warm and cold fronts from WPC bulletin.

    Input file should contain positions of cyclones, anticyclones, fronts, etc.
    for a single valid time.

    :param text_file_name: Path to input file (text file in WPC format).
    :return: front_table: pandas DataFrame with the following columns.  Each row
        is one front.
    front_table.front_type: Type of front (examples: "warm", "cold").
    front_table.unix_time_sec: Valid time.
    front_table.latitudes_deg: 1-D numpy array of latitudes (deg N) along front.
    front_table.longitudes_deg: 1-D numpy array of longitudes (deg E) along
        front.
    """

    error_checking.assert_file_exists(text_file_name)
    valid_time_unix_sec = _file_name_to_valid_time(text_file_name)

    front_types = []
    latitudes_2d_list_deg = []
    longitudes_2d_list_deg = []

    for this_line in open(text_file_name, 'r').readlines():
        these_words = this_line.split()  # Need to skip empty lines.
        if not these_words:
            continue

        this_front_type = these_words[0].lower()
        if this_front_type not in VALID_FRONT_TYPES:
            continue

        these_words = these_words[1:]
        this_num_points = len(these_words)
        these_latitudes_deg = numpy.full(this_num_points, numpy.nan)
        these_longitudes_deg = numpy.full(this_num_points, numpy.nan)

        for i in range(this_num_points):
            these_latitudes_deg[i], these_longitudes_deg[i] = _string_to_latlng(
                these_words[i], False)

        if numpy.any(numpy.isnan(these_latitudes_deg)):
            continue

        error_checking.assert_is_valid_lat_numpy_array(these_latitudes_deg)
        these_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
            these_longitudes_deg, allow_nan=False)

        front_types.append(this_front_type)
        latitudes_2d_list_deg.append(these_latitudes_deg)
        longitudes_2d_list_deg.append(these_longitudes_deg)

    num_fronts = len(front_types)
    valid_times_unix_sec = numpy.full(
        num_fronts, valid_time_unix_sec, dtype=int)

    front_dict = {
        front_utils.FRONT_TYPE_COLUMN: front_types,
        front_utils.TIME_COLUMN: valid_times_unix_sec,
        front_utils.LATITUDES_COLUMN: latitudes_2d_list_deg,
        front_utils.LONGITUDES_COLUMN: longitudes_2d_list_deg
    }
    return pandas.DataFrame.from_dict(front_dict)
