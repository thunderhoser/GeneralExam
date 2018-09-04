"""For each time step, processes a single NARR field and writes to file."""

import argparse
import numpy
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import narr_netcdf_io
from generalexam.ge_io import processed_narr_io

LAST_GRIB_TIME_UNIX_SEC = 1412283600
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

INPUT_TIME_FORMAT = '%Y%m%d%H'
DEFAULT_TIME_FORMAT = '%Y-%m-%d-%H'
MONTH_TIME_FORMAT = '%Y%m'
TIME_INTERVAL_SECONDS = 10800

MB_TO_PASCALS = 100
WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
FIELD_NAME_ARG_NAME = 'field_name'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
NARR_DIR_ARG_NAME = 'input_narr_dir_name'
PROCESSED_DIR_ARG_NAME = 'output_processed_dir_name'

TIME_HELP_STRING = (
    'Time in format "yyyymmddHH".  This script will process the given field, at'
    ' the given pressure level, for all 3-hour time steps from '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

FIELD_NAME_HELP_STRING = (
    'Name of field (must be in list `processed_narr_io.STANDARD_FIELD_NAMES`).'
    '  For each time step, this field (at the given pressure level) will be '
    'written to the output file.  Winds will be rotated from Earth-relative to '
    'grid-relative.')

PRESSURE_LEVEL_HELP_STRING = (
    'Pressure level (millibars).  See description for `{0:s}`'
).format(FIELD_NAME_ARG_NAME)

NARR_DIR_HELP_STRING = (
    'Name of top-level input directory, containing NARR data in both grib and '
    'NetCDF formats.')

PROCESSED_DIR_HELP_STRING = (
    'Name of top-level output directory.  One Pickle file per time step will be'
    ' saved here.')

DEFAULT_PRESSURE_LEVEL_MB = 1000
DEFAULT_NARR_DIR_NAME = '/condo/swatwork/ralager/narr_data'
DEFAULT_PROCESSED_DIR_NAME = '/condo/swatwork/ralager/narr_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_NAME_ARG_NAME, type=str, required=True,
    help=FIELD_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_DIR_NAME, help=NARR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PROCESSED_DIR_ARG_NAME, type=str, required=False,
    default=DEFAULT_PROCESSED_DIR_NAME, help=PROCESSED_DIR_HELP_STRING)


def _field_name_gg_to_grib1(field_name, pressure_level_mb):
    """Converts field name from GewitterGefahr format to grib1 format.

    :param field_name: Field name in GG format.
    :param pressure_level_mb: Pressure level (millibars).
    :return: field_name_grib1: Field name in grib1 format.
    """

    if field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
        field_name_grib1 = 'UGRD'
    elif field_name == processed_narr_io.V_WIND_EARTH_RELATIVE_NAME:
        field_name_grib1 = 'VGRD'
    elif field_name == processed_narr_io.SPECIFIC_HUMIDITY_NAME:
        field_name_grib1 = 'SPFH'
    elif field_name == processed_narr_io.TEMPERATURE_NAME:
        field_name_grib1 = 'TMP'
    elif field_name == processed_narr_io.HEIGHT_NAME:
        field_name_grib1 = 'HGT'
    elif field_name == processed_narr_io.VERTICAL_VELOCITY_NAME:
        field_name_grib1 = 'VVEL'

    return '{0:s}:{1:d} mb'.format(field_name_grib1, pressure_level_mb)


def _run(
        first_time_string, last_time_string, field_name, pressure_level_mb,
        top_input_narr_dir_name, top_processed_dir_name):
    """For each time step, processes a single NARR field and writes to file.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param field_name: Same.
    :param pressure_level_mb: Same.
    :param top_input_narr_dir_name: Same.
    :param top_processed_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, DEFAULT_TIME_FORMAT) for
        t in valid_times_unix_sec
    ]

    if field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
        field_name_other_component = (
            processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
        )
    elif field_name == processed_narr_io.V_WIND_EARTH_RELATIVE_NAME:
        field_name_other_component = (
            processed_narr_io.U_WIND_EARTH_RELATIVE_NAME
        )
    else:
        field_name_other_component = None

    field_name_grib1 = _field_name_gg_to_grib1(
        field_name=field_name, pressure_level_mb=pressure_level_mb)

    rotate_winds = field_name in WIND_FIELD_NAMES
    if rotate_winds:
        field_name_grib1_other_component = _field_name_gg_to_grib1(
            field_name=field_name_other_component,
            pressure_level_mb=pressure_level_mb)
    else:
        field_name_grib1_other_component = None

    num_times = len(valid_times_unix_sec)
    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    if rotate_winds:
        (grid_point_lat_matrix_deg, grid_point_lng_matrix_deg
        ) = nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME)

        (rotation_angle_cos_matrix, rotation_angle_sin_matrix
        ) = nwp_model_utils.get_wind_rotation_angles(
            latitudes_deg=grid_point_lat_matrix_deg,
            longitudes_deg=grid_point_lng_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)

    for i in range(num_times):
        field_matrix = numpy.full(
            (1, num_grid_rows, num_grid_columns), numpy.nan)

        if rotate_winds:
            field_matrix_other_component = numpy.full(
                (1, num_grid_rows, num_grid_columns), numpy.nan)

        if valid_times_unix_sec[i] > LAST_GRIB_TIME_UNIX_SEC:
            this_month_string = time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], MONTH_TIME_FORMAT)

            this_netcdf_file_name = narr_netcdf_io.find_file(
                month_string=this_month_string, field_name=field_name,
                top_directory_name=top_input_narr_dir_name)
            if rotate_winds:
                this_netcdf_file_name_other_component = (
                    narr_netcdf_io.find_file(
                        month_string=this_month_string,
                        field_name=field_name_other_component,
                        top_directory_name=top_input_narr_dir_name)
                )

            print 'Reading "{0:s}" at {1:d} mb from: "{2:s}"...'.format(
                field_name, pressure_level_mb, this_netcdf_file_name)
            field_matrix[0, ...] = narr_netcdf_io.read_data_from_file(
                this_netcdf_file_name, field_name=field_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_level_mb)[0]

            if rotate_winds:
                print 'Reading "{0:s}" at {1:d} mb from: "{2:s}"...'.format(
                    field_name_other_component, pressure_level_mb,
                    this_netcdf_file_name_other_component)

                field_matrix_other_component[0, ...] = (
                    narr_netcdf_io.read_data_from_file(
                        this_netcdf_file_name_other_component,
                        field_name=field_name_other_component,
                        valid_time_unix_sec=valid_times_unix_sec[i],
                        pressure_level_mb=pressure_level_mb)
                )[0]

        else:
            this_grib_file_name = nwp_model_io.find_grib_file(
                init_time_unix_sec=valid_times_unix_sec[i], lead_time_hours=0,
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                top_directory_name=top_input_narr_dir_name)

            print 'Reading "{0:s}" at {1:d} mb from: "{2:s}"...'.format(
                field_name, pressure_level_mb, this_grib_file_name)

            field_matrix[0, ...] = nwp_model_io.read_field_from_grib_file(
                grib_file_name=this_grib_file_name,
                field_name_grib1=field_name_grib1,
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                temporary_dir_name=None, wgrib_exe_name=WGRIB_EXE_NAME,
                wgrib2_exe_name=WGRIB2_EXE_NAME, raise_error_if_fails=True)

            if rotate_winds:
                print 'Reading "{0:s}" at {1:d} mb from: "{2:s}"...'.format(
                    field_name_other_component, pressure_level_mb,
                    this_grib_file_name)

                field_matrix_other_component[
                    0, ...
                ] = nwp_model_io.read_field_from_grib_file(
                    grib_file_name=this_grib_file_name,
                    field_name_grib1=field_name_grib1_other_component,
                    model_name=nwp_model_utils.NARR_MODEL_NAME,
                    temporary_dir_name=None, wgrib_exe_name=WGRIB_EXE_NAME,
                    wgrib2_exe_name=WGRIB2_EXE_NAME, raise_error_if_fails=True)

            print '\n'

        if rotate_winds:
            print ('Rotating winds at {0:s} from Earth-relative to grid-'
                   'relative...').format(valid_time_strings[i])

            if field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
                (field_matrix[0, ...], field_matrix_other_component[0, ...]
                ) = nwp_model_utils.rotate_winds_to_grid_relative(
                    u_winds_earth_relative_m_s01=field_matrix[0, ...],
                    v_winds_earth_relative_m_s01=
                    field_matrix_other_component[0, ...],
                    rotation_angle_cosines=rotation_angle_cos_matrix,
                    rotation_angle_sines=rotation_angle_sin_matrix)
            else:
                (field_matrix_other_component[0, ...], field_matrix[0, ...]
                ) = nwp_model_utils.rotate_winds_to_grid_relative(
                    u_winds_earth_relative_m_s01=
                    field_matrix_other_component[0, ...],
                    v_winds_earth_relative_m_s01=field_matrix[0, ...],
                    rotation_angle_cosines=rotation_angle_cos_matrix,
                    rotation_angle_sines=rotation_angle_sin_matrix)

            field_name = processed_narr_io.field_name_to_grid_relative(
                field_name)
            field_name_other_component = (
                processed_narr_io.field_name_to_grid_relative(
                    field_name_other_component)
            )

        this_processed_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_processed_dir_name, field_name=field_name,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing "{0:s}" at {1:d} mb and {2:s} to: "{3:s}"...'.format(
            field_name, pressure_level_mb, valid_time_strings[i],
            this_processed_file_name)
        processed_narr_io.write_fields_to_file(
            pickle_file_name=this_processed_file_name,
            field_matrix=field_matrix, field_name=field_name,
            pressure_level_pascals=pressure_level_mb * MB_TO_PASCALS,
            valid_times_unix_sec=numpy.array([valid_times_unix_sec[0]]))

        if rotate_winds:
            this_processed_file_name = processed_narr_io.find_file_for_one_time(
                top_directory_name=top_processed_dir_name,
                field_name=field_name_other_component,
                pressure_level_mb=pressure_level_mb,
                valid_time_unix_sec=valid_times_unix_sec[i],
                raise_error_if_missing=False)

            print 'Writing "{0:s}" at {1:d} mb and {2:s} to: "{3:s}"...'.format(
                field_name_other_component, pressure_level_mb,
                valid_time_strings[i], this_processed_file_name)

            processed_narr_io.write_fields_to_file(
                pickle_file_name=this_processed_file_name,
                field_matrix=field_matrix_other_component,
                field_name=field_name_other_component,
                pressure_level_pascals=pressure_level_mb * MB_TO_PASCALS,
                valid_times_unix_sec=numpy.array([valid_times_unix_sec[0]]))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        field_name=getattr(INPUT_ARG_OBJECT, FIELD_NAME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_input_narr_dir_name=getattr(INPUT_ARG_OBJECT, NARR_DIR_ARG_NAME),
        top_processed_dir_name=getattr(INPUT_ARG_OBJECT, PROCESSED_DIR_ARG_NAME)
    )
