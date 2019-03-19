"""Processes ERA5 data (this includes interpolating to NARR grid)."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import moisture_conversions
from generalexam.ge_io import era5_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

YEAR_FORMAT = '%Y'
INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

MB_TO_PASCALS = 100.
TIME_INTERVAL_SECONDS = 10800

INPUT_DIR_ARG_NAME = 'input_dir_name'
RAW_FIELDS_ARG_NAME = 'raw_field_names'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory, containing raw files from John Allen.  '
    'Files therein will be found by `era5_io.find_raw_file` and read by '
    '`era5_io.read_raw_file`.')

RAW_FIELDS_HELP_STRING = (
    'List of fields to process.  Each field name must be accepted by '
    '`era5_io.check_raw_field_name`.')

PRESSURE_LEVEL_HELP_STRING = (
    'Will process data only at this pressure level (millibars).  Surface is '
    'denoted by {0:d}.'
).format(era5_io.DUMMY_SURFACE_PRESSURE_MB)

TIME_HELP_STRING = (
    'Valid time (format "yyyymmddHH").  This script will process data for all '
    'valid times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written by '
    '`era5_io.find_processed_file` to locations therein determined by '
    '`era5_io.write_processed_file`.')

TOP_INPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/raw'
TOP_OUTPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'

DEFAULT_RAW_FIELD_NAMES = [
    era5_io.TEMPERATURE_NAME_RAW, era5_io.HEIGHT_NAME_RAW,
    era5_io.DEWPOINT_NAME_RAW, era5_io.U_WIND_NAME_RAW, era5_io.V_WIND_NAME_RAW
]
DEFAULT_PRESSURE_LEVEL_MB = 1000

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_INPUT_DIR_NAME_DEFAULT, help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_RAW_FIELD_NAMES, help=RAW_FIELDS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_OUTPUT_DIR_NAME_DEFAULT, help=OUTPUT_DIR_HELP_STRING)


def _add_required_fields(raw_field_names, pressure_level_mb):
    """Adds required fields to list.

    Specifically, if the list contains a field x that cannot be processed
    without field y, this method adds y to the list.

    :param raw_field_names: See documentation at top of file.
    :param pressure_level_mb: Same.
    :return: raw_field_names: Same as input, except maybe longer.
    """

    if (era5_io.U_WIND_NAME_RAW in raw_field_names and
            era5_io.V_WIND_NAME_RAW not in raw_field_names):
        raw_field_names.append(era5_io.V_WIND_NAME_RAW)

    if (era5_io.V_WIND_NAME_RAW in raw_field_names and
            era5_io.U_WIND_NAME_RAW not in raw_field_names):
        raw_field_names.append(era5_io.U_WIND_NAME_RAW)

    if (era5_io.DEWPOINT_NAME_RAW in raw_field_names and
            era5_io.PRESSURE_NAME_RAW not in raw_field_names and
            pressure_level_mb == era5_io.DUMMY_SURFACE_PRESSURE_MB):
        raw_field_names.append(era5_io.PRESSURE_NAME_RAW)

    return raw_field_names


def _run(top_input_dir_name, raw_field_names, pressure_level_mb,
         first_time_string, last_time_string, top_output_dir_name):
    """Processes ERA5 data (this includes interpolating to NARR grid).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param raw_field_names: Same.
    :param pressure_level_mb: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param top_output_dir_name: Same.
    """

    raw_field_names = _add_required_fields(
        raw_field_names=raw_field_names, pressure_level_mb=pressure_level_mb)

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    if (era5_io.U_WIND_NAME_RAW in raw_field_names or
            era5_io.V_WIND_NAME_RAW in raw_field_names):
        narr_latitude_matrix_deg, narr_longitude_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME)
        )

        narr_rotation_cos_matrix, narr_rotation_sin_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=narr_latitude_matrix_deg,
                longitudes_deg=narr_longitude_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME)
        )
    else:
        narr_rotation_cos_matrix = None
        narr_rotation_sin_matrix = None

    field_names = [
        era5_io.field_name_raw_to_processed(
            raw_field_name=f, earth_relative=False)
        for f in raw_field_names
    ]

    if era5_io.DEWPOINT_NAME in field_names:
        dewpoint_index = field_names.index(era5_io.DEWPOINT_NAME)
        field_names[dewpoint_index] = era5_io.SPECIFIC_HUMIDITY_NAME

    num_times = len(valid_times_unix_sec)
    num_fields = len(raw_field_names)

    this_era5_dict = None
    era5_x_matrix_metres = None
    era5_y_matrix_metres = None

    for i in range(num_times):
        this_time_string = time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], LOG_MESSAGE_TIME_FORMAT)
        this_year = int(time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], YEAR_FORMAT
        ))

        one_time_data_matrix = None

        for j in range(num_fields):
            this_raw_file_name = era5_io.find_raw_file(
                top_directory_name=top_input_dir_name, year=this_year,
                raw_field_name=raw_field_names[j],
                has_surface_data=(
                    pressure_level_mb == era5_io.DUMMY_SURFACE_PRESSURE_MB)
            )

            print 'Reading data at {0:s} from file: "{1:s}"...'.format(
                this_time_string, this_raw_file_name)

            this_era5_dict = era5_io.read_raw_file(
                netcdf_file_name=this_raw_file_name,
                first_time_unix_sec=valid_times_unix_sec[i],
                last_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_level_mb)

            if one_time_data_matrix is None:
                num_grid_rows = this_era5_dict[
                    era5_io.DATA_MATRIX_KEY].shape[1]
                num_grid_columns = this_era5_dict[
                    era5_io.DATA_MATRIX_KEY].shape[2]

                one_time_data_matrix = numpy.full(
                    (1, num_grid_rows, num_grid_columns, 1, num_fields),
                    numpy.nan
                )

            one_time_data_matrix[0, ..., 0, j] = this_era5_dict[
                era5_io.DATA_MATRIX_KEY][0, ..., 0, 0]

        if era5_io.DEWPOINT_NAME_RAW in raw_field_names:
            print 'Converting dewpoint to specific humidity...'
            dewpoint_index = raw_field_names.index(era5_io.DEWPOINT_NAME_RAW)

            if pressure_level_mb == era5_io.DUMMY_SURFACE_PRESSURE_MB:
                pressure_index = raw_field_names.index(
                    era5_io.PRESSURE_NAME_RAW)
                this_pressure_matrix_pascals = one_time_data_matrix[
                    ..., pressure_index]
            else:
                this_pressure_matrix_pascals = numpy.full(
                    one_time_data_matrix.shape[:-1],
                    MB_TO_PASCALS * pressure_level_mb
                )

            one_time_data_matrix[..., dewpoint_index] = (
                moisture_conversions.dewpoint_to_specific_humidity(
                    dewpoints_kelvins=one_time_data_matrix[
                        ..., dewpoint_index],
                    total_pressures_pascals=this_pressure_matrix_pascals
                )
            )

        one_time_era5_dict = {
            era5_io.DATA_MATRIX_KEY: one_time_data_matrix,
            era5_io.VALID_TIMES_KEY: valid_times_unix_sec[[i]],
            era5_io.LATITUDES_KEY: this_era5_dict[era5_io.LATITUDES_KEY],
            era5_io.LONGITUDES_KEY: this_era5_dict[era5_io.LONGITUDES_KEY],
            era5_io.PRESSURE_LEVELS_KEY:
                numpy.array([pressure_level_mb], dtype=int),
            era5_io.FIELD_NAMES_KEY: field_names
        }

        print '\n'
        one_time_era5_dict = era5_io.interp_to_narr_grid(
            era5_dict=one_time_era5_dict,
            era5_x_matrix_metres=era5_x_matrix_metres,
            era5_y_matrix_metres=era5_y_matrix_metres)
        print '\n'

        if era5_io.U_WIND_NAME_RAW in raw_field_names:
            u_wind_index = raw_field_names.index(era5_io.U_WIND_NAME_RAW)
            v_wind_index = raw_field_names.index(era5_io.V_WIND_NAME_RAW)

            print 'Rotating winds from Earth-relative to grid-relative...'

            (one_time_era5_dict[era5_io.DATA_MATRIX_KEY][
                0, ..., 0, u_wind_index],
             one_time_era5_dict[era5_io.DATA_MATRIX_KEY][
                 0, ..., 0, v_wind_index]
            ) = nwp_model_utils.rotate_winds_to_grid_relative(
                u_winds_earth_relative_m_s01=one_time_era5_dict[
                    era5_io.DATA_MATRIX_KEY][0, ..., 0, u_wind_index],
                v_winds_earth_relative_m_s01=one_time_era5_dict[
                    era5_io.DATA_MATRIX_KEY][0, ..., 0, v_wind_index],
                rotation_angle_cosines=narr_rotation_cos_matrix,
                rotation_angle_sines=narr_rotation_sin_matrix)

        this_processed_file_name = era5_io.find_processed_file(
            top_directory_name=top_output_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing processed ERA5 data on NARR grid to: "{0:s}"...'.format(
            this_processed_file_name)

        era5_io.write_processed_file(netcdf_file_name=this_processed_file_name,
                                     era5_dict=one_time_era5_dict)

        print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        raw_field_names=getattr(INPUT_ARG_OBJECT, RAW_FIELDS_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
