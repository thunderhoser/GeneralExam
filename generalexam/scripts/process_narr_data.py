"""Converts NARR data to a more convenient file format."""

import argparse
import numpy
from gewittergefahr.gg_io import nwp_model_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import narr_netcdf_io
from generalexam.ge_io import processed_narr_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

WGRIB_EXE_NAME = '/condo/swatwork/ralager/wgrib/wgrib'
WGRIB2_EXE_NAME = '/condo/swatwork/ralager/grib2/wgrib2/wgrib2'

INPUT_TIME_FORMAT = '%Y%m%d%H'
MONTH_TIME_FORMAT = '%Y%m'
TIME_INTERVAL_SECONDS = 10800
LAST_GRIB_TIME_UNIX_SEC = 1412283600

MB_TO_PASCALS = 100
DUMMY_PRESSURE_LEVEL_MB = 1013

WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]

THERMAL_FIELD_NAMES = [
    processed_narr_io.TEMPERATURE_NAME,
    processed_narr_io.SPECIFIC_HUMIDITY_NAME
]

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
FIELD_NAME_ARG_NAME = 'field_name'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with unprocessed NARR data (in grib and/or '
    'NetCDF files).  Files therein will be found by '
    '`nwp_model_io.find_grib_file` or `narr_netcdf_io.find_file`, and read by '
    '`nwp_model_io.read_field_from_grib_file` or `narr_netcdf_io.read_file`.')

TIME_HELP_STRING = (
    'Valid time (format "yyyymmddHH").  Will convert NARR data for all valid '
    'times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

FIELD_NAME_HELP_STRING = (
    'Field name (must be accepted by `processed_narr_io.check_field_name`).  '
    'Only this field will be extracted and converted.')

PRESSURE_LEVEL_HELP_STRING = (
    'Pressure level (millibars).  The field (`{0:s}`) will be extracted only at'
    ' this pressure level.  If you want the surface field, leave this argument '
    'alone.'
).format(FIELD_NAME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory for processed NARR data.  Output will be '
    'written here by `processed_narr_io.write_fields_to_file`, to file '
    'locations determined by `processed_narr_io.find_file_for_one_time`.')

TOP_INPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data'
TOP_OUTPUT_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_INPUT_DIR_NAME_DEFAULT, help=INPUT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_NAME_ARG_NAME, type=str, required=True,
    help=FIELD_NAME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False, default=-1,
    help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_OUTPUT_DIR_NAME_DEFAULT, help=OUTPUT_DIR_HELP_STRING)


def _std_to_grib1_field_name(field_name, pressure_level_mb=None):
    """Converts field name from standard to grib1 format.

    :param field_name: Field name in standard format (must be accepted by
        `processed_narr_io.check_field_name`).
    :param pressure_level_mb: Pressure level (millibars).  For surface field,
        leave this alone.
    :return: field_name_grib1: Field name in grib1 format.
    """

    processed_narr_io.check_field_name(
        field_name=field_name, require_standard=True)

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

    if pressure_level_mb is not None:
        return '{0:s}:{1:d} mb'.format(field_name_grib1, pressure_level_mb)

    if field_name in WIND_FIELD_NAMES:
        return '{0:s}:10 m above gnd'.format(field_name_grib1)

    if field_name in THERMAL_FIELD_NAMES:
        return '{0:s}:2 m above gnd'.format(field_name_grib1)

    if field_name == processed_narr_io.VERTICAL_VELOCITY_NAME:
        return '{0:s}:30-0 mb above gnd'.format(field_name_grib1)

    # TODO(thunderhoser): This is a HACK.
    if field_name == processed_narr_io.HEIGHT_NAME:
        return 'PRES:2 m above gnd'


def _run(top_input_dir_name, first_time_string, last_time_string,
         input_field_name, pressure_level_mb, top_output_dir_name):
    """Converts NARR data to a more convenient file format.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param input_field_name: Same.
    :param pressure_level_mb: Same.
    :param top_output_dir_name: Same.
    """

    if pressure_level_mb <= 0:
        pressure_level_mb = None

    if pressure_level_mb is None:
        output_pressure_level_mb = DUMMY_PRESSURE_LEVEL_MB + 0
    else:
        output_pressure_level_mb = pressure_level_mb + 0

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS)

    if input_field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
        input_field_name_other = (
            processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
        )
    elif input_field_name == processed_narr_io.V_WIND_EARTH_RELATIVE_NAME:
        input_field_name_other = (
            processed_narr_io.U_WIND_EARTH_RELATIVE_NAME
        )
    else:
        input_field_name_other = None

    input_field_name_grib1 = _std_to_grib1_field_name(
        field_name=input_field_name, pressure_level_mb=pressure_level_mb)

    if input_field_name in WIND_FIELD_NAMES:
        input_field_name_other_grib1 = _std_to_grib1_field_name(
            field_name=input_field_name_other,
            pressure_level_mb=pressure_level_mb)

        output_field_name = processed_narr_io.field_name_to_grid_relative(
            input_field_name)

        output_field_name_other = (
            processed_narr_io.field_name_to_grid_relative(
                input_field_name_other)
        )

        (narr_latitude_matrix_deg, narr_longitude_matrix_deg
        ) = nwp_model_utils.get_latlng_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME)

        (narr_rotation_cosine_matrix, narr_rotation_sine_matrix
        ) = nwp_model_utils.get_wind_rotation_angles(
            latitudes_deg=narr_latitude_matrix_deg,
            longitudes_deg=narr_longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME)
    else:
        input_field_name_other_grib1 = None
        output_field_name = input_field_name + ''
        output_field_name_other = None

    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        if input_field_name in WIND_FIELD_NAMES:
            this_field_matrix_other = None

        if valid_times_unix_sec[i] > LAST_GRIB_TIME_UNIX_SEC:
            this_month_string = time_conversion.unix_sec_to_string(
                valid_times_unix_sec[i], MONTH_TIME_FORMAT)

            this_netcdf_file_name = narr_netcdf_io.find_file(
                top_directory_name=top_input_dir_name,
                field_name=input_field_name, month_string=this_month_string,
                is_surface=pressure_level_mb is None)

            print 'Reading data from: "{0:s}"...'.format(this_netcdf_file_name)
            this_field_matrix = narr_netcdf_io.read_file(
                netcdf_file_name=this_netcdf_file_name,
                field_name=input_field_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_level_mb)

            if input_field_name in WIND_FIELD_NAMES:
                this_netcdf_file_name_other = narr_netcdf_io.find_file(
                    top_directory_name=top_input_dir_name,
                    field_name=input_field_name_other,
                    month_string=this_month_string,
                    is_surface=pressure_level_mb is None)

                print 'Reading data from: "{0:s}"...'.format(
                    this_netcdf_file_name_other)

                this_field_matrix_other = narr_netcdf_io.read_file(
                    netcdf_file_name=this_netcdf_file_name_other,
                    field_name=input_field_name_other,
                    valid_time_unix_sec=valid_times_unix_sec[i],
                    pressure_level_mb=pressure_level_mb)
        else:
            this_grib_file_name = nwp_model_io.find_grib_file(
                top_directory_name=top_input_dir_name,
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                init_time_unix_sec=valid_times_unix_sec[i], lead_time_hours=0)

            print 'Reading data from: "{0:s}"...'.format(this_grib_file_name)
            this_field_matrix = nwp_model_io.read_field_from_grib_file(
                grib_file_name=this_grib_file_name,
                field_name_grib1=input_field_name_grib1,
                model_name=nwp_model_utils.NARR_MODEL_NAME,
                wgrib_exe_name=WGRIB_EXE_NAME, wgrib2_exe_name=WGRIB2_EXE_NAME)

            if input_field_name in WIND_FIELD_NAMES:
                this_field_matrix_other = (
                    nwp_model_io.read_field_from_grib_file(
                        grib_file_name=this_grib_file_name,
                        field_name_grib1=input_field_name_other_grib1,
                        model_name=nwp_model_utils.NARR_MODEL_NAME,
                        wgrib_exe_name=WGRIB_EXE_NAME,
                        wgrib2_exe_name=WGRIB2_EXE_NAME)
                )

        if input_field_name in WIND_FIELD_NAMES:
            print 'Rotating Earth-relative winds to grid-relative...'

            if input_field_name == processed_narr_io.U_WIND_EARTH_RELATIVE_NAME:
                this_field_matrix, this_field_matrix_other = (
                    nwp_model_utils.rotate_winds_to_grid_relative(
                        u_winds_earth_relative_m_s01=this_field_matrix,
                        v_winds_earth_relative_m_s01=this_field_matrix_other,
                        rotation_angle_cosines=narr_rotation_cosine_matrix,
                        rotation_angle_sines=narr_rotation_sine_matrix)
                )
            else:
                this_field_matrix_other, this_field_matrix = (
                    nwp_model_utils.rotate_winds_to_grid_relative(
                        u_winds_earth_relative_m_s01=this_field_matrix_other,
                        v_winds_earth_relative_m_s01=this_field_matrix,
                        rotation_angle_cosines=narr_rotation_cosine_matrix,
                        rotation_angle_sines=narr_rotation_sine_matrix)
                )

        this_output_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_output_dir_name,
            field_name=output_field_name,
            pressure_level_mb=output_pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing processed data to: "{0:s}"...'.format(
            this_output_file_name)

        processed_narr_io.write_fields_to_file(
            pickle_file_name=this_output_file_name,
            field_matrix=numpy.expand_dims(this_field_matrix, axis=0),
            field_name=output_field_name,
            pressure_level_pascals=output_pressure_level_mb * MB_TO_PASCALS,
            valid_times_unix_sec=valid_times_unix_sec[[i]]
        )

        if input_field_name not in WIND_FIELD_NAMES:
            print '\n'
            continue

        this_output_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_output_dir_name,
            field_name=output_field_name_other,
            pressure_level_mb=output_pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing processed data to: "{0:s}"...\n'.format(
            this_output_file_name)

        processed_narr_io.write_fields_to_file(
            pickle_file_name=this_output_file_name,
            field_matrix=numpy.expand_dims(this_field_matrix_other, axis=0),
            field_name=output_field_name_other,
            pressure_level_pascals=output_pressure_level_mb * MB_TO_PASCALS,
            valid_times_unix_sec=valid_times_unix_sec[[i]]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        input_field_name=getattr(INPUT_ARG_OBJECT, FIELD_NAME_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
