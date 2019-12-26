"""Processes ERA5 data (this includes interpolating to NARR grid)."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import era5_input
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

YEAR_FORMAT = '%Y'
INPUT_TIME_FORMAT = '%Y%m%d%H'
LOG_MESSAGE_TIME_FORMAT = '%Y-%m-%d-%H'

MB_TO_PASCALS = 100.
TIME_INTERVAL_SECONDS = 10800
DUMMY_SURFACE_PRESSURE_MB = predictor_utils.DUMMY_SURFACE_PRESSURE_MB

INPUT_DIR_ARG_NAME = 'input_dir_name'
EXTENDED_GRID_ARG_NAME = 'extended_grid'
RAW_FIELDS_ARG_NAME = 'raw_field_names'
PRESSURE_LEVELS_ARG_NAME = 'pressure_levels_mb'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory, containing raw files from John Allen.  '
    'Files therein will be found by `era5_input.find_file` and read by '
    '`era5_input.read_file`.'
)
EXTENDED_GRID_HELP_STRING = (
    'Boolean flag.  If 1, will process data on extended NARR grid.  If 0, only '
    'on main NARR grid.'
)

RAW_FIELDS_HELP_STRING = (
    'List of fields to process.  Each field name must be accepted by '
    '`era5_input._check_raw_field_name`.'
)
PRESSURE_LEVELS_HELP_STRING = (
    'List of pressure levels (one for each field in `{0:s}`).'
).format(RAW_FIELDS_ARG_NAME)

TIME_HELP_STRING = (
    'Valid time (format "yyyymmddHH").  This script will process data for all '
    'valid times in the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Files will be written by '
    '`predictor_io.find_file` to locations therein determined by '
    '`predictor_io.write_file`.'
)

DEFAULT_RAW_FIELD_NAMES = [
    era5_input.TEMPERATURE_NAME_RAW, era5_input.HEIGHT_NAME_RAW,
    era5_input.SPECIFIC_HUMIDITY_NAME_RAW, era5_input.U_WIND_NAME_RAW,
    era5_input.V_WIND_NAME_RAW
]

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXTENDED_GRID_ARG_NAME, type=int, required=True,
    help=EXTENDED_GRID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FIELDS_ARG_NAME, type=str, nargs='+', required=False,
    default=DEFAULT_RAW_FIELD_NAMES, help=RAW_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVELS_ARG_NAME, type=int, nargs='+', required=False,
    default=[DUMMY_SURFACE_PRESSURE_MB], help=PRESSURE_LEVELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _fix_wrong_level_fields(raw_field_names, pressure_levels_mb):
    """Fixes fields at the wrong level.

    In ERA5, the humidity variable is dewpoint at the surface and specific
    humidity aloft, while the pressure variable is pressure at the surface and
    geopotential height aloft.  This method ensures that each variable is at a
    vertical level where it exists.

    :param raw_field_names: See documentation at top of file.
    :param pressure_levels_mb: Same.
    :return: raw_field_names: Same as input but maybe with different elements.
    """

    raw_field_names = numpy.array(raw_field_names)

    these_indices = numpy.where(numpy.logical_and(
        raw_field_names == era5_input.SPECIFIC_HUMIDITY_NAME_RAW,
        pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
    ))
    raw_field_names[these_indices] = era5_input.DEWPOINT_NAME_RAW

    these_indices = numpy.where(numpy.logical_and(
        raw_field_names == era5_input.DEWPOINT_NAME_RAW,
        pressure_levels_mb != DUMMY_SURFACE_PRESSURE_MB
    ))
    raw_field_names[these_indices] = era5_input.SPECIFIC_HUMIDITY_NAME_RAW

    these_indices = numpy.where(numpy.logical_and(
        raw_field_names == era5_input.HEIGHT_NAME_RAW,
        pressure_levels_mb == DUMMY_SURFACE_PRESSURE_MB
    ))
    raw_field_names[these_indices] = era5_input.PRESSURE_NAME_RAW

    these_indices = numpy.where(numpy.logical_and(
        raw_field_names == era5_input.PRESSURE_NAME_RAW,
        pressure_levels_mb != DUMMY_SURFACE_PRESSURE_MB
    ))
    raw_field_names[these_indices] = era5_input.HEIGHT_NAME_RAW

    return raw_field_names.tolist()


def _add_required_fields(raw_field_names, pressure_levels_mb):
    """Adds required fields to list.

    Specifically, if the list contains a field x that cannot be processed
    without field y, this method adds y to the list.

    :param raw_field_names: See documentation at top of file.
    :param pressure_levels_mb: Same.
    :return: raw_field_names: Same as input but maybe with new elements at the
        end.
    :return: pressure_levels_mb: Same as input but maybe with new elements at
        the end.
    """

    raw_field_names = numpy.array(raw_field_names)
    unique_pressures_mb = numpy.unique(pressure_levels_mb)

    new_field_names = []
    new_pressures_mb = []

    for this_pressure_mb in unique_pressures_mb:
        these_indices = numpy.where(pressure_levels_mb == this_pressure_mb)[0]

        add_v_wind = (
            era5_input.U_WIND_NAME_RAW in raw_field_names[these_indices] and
            era5_input.V_WIND_NAME_RAW not in raw_field_names[these_indices]
        )
        if add_v_wind:
            new_field_names.append(era5_input.V_WIND_NAME_RAW)
            new_pressures_mb.append(this_pressure_mb)

        add_u_wind = (
            era5_input.V_WIND_NAME_RAW in raw_field_names[these_indices] and
            era5_input.U_WIND_NAME_RAW not in raw_field_names[these_indices]
        )
        if add_u_wind:
            new_field_names.append(era5_input.U_WIND_NAME_RAW)
            new_pressures_mb.append(this_pressure_mb)

        add_surface_pressure = (
            era5_input.DEWPOINT_NAME_RAW in raw_field_names[these_indices] and
            era5_input.PRESSURE_NAME_RAW not in raw_field_names[these_indices]
        )
        if add_surface_pressure:
            new_field_names.append(era5_input.PRESSURE_NAME_RAW)
            new_pressures_mb.append(this_pressure_mb)

    raw_field_names = raw_field_names.tolist()
    raw_field_names += new_field_names

    new_pressures_mb = numpy.array(new_pressures_mb, dtype=int)
    pressure_levels_mb = numpy.concatenate((
        pressure_levels_mb, new_pressures_mb
    ))

    return raw_field_names, pressure_levels_mb


def _run(top_input_dir_name, extended_grid, raw_field_names, pressure_levels_mb,
         first_time_string, last_time_string, top_output_dir_name):
    """Processes ERA5 data (this includes interpolating to NARR grid).

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param extended_grid: Same.
    :param raw_field_names: Same.
    :param pressure_levels_mb: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param top_output_dir_name: Same.
    """

    grid_name = (
        nwp_model_utils.NAME_OF_EXTENDED_221GRID if extended_grid
        else nwp_model_utils.NAME_OF_221GRID
    )

    num_fields = len(raw_field_names)
    expected_dim = numpy.array([num_fields], dtype=int)
    error_checking.assert_is_numpy_array(
        pressure_levels_mb, exact_dimensions=expected_dim
    )

    raw_field_names = _fix_wrong_level_fields(
        raw_field_names=raw_field_names, pressure_levels_mb=pressure_levels_mb
    )
    raw_field_names, pressure_levels_mb = _add_required_fields(
        raw_field_names=raw_field_names, pressure_levels_mb=pressure_levels_mb
    )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

    if era5_input.U_WIND_NAME_RAW in raw_field_names:
        narr_latitude_matrix_deg, narr_longitude_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name)
        )

        narr_cosine_matrix, narr_sine_matrix = (
            nwp_model_utils.get_wind_rotation_angles(
                latitudes_deg=narr_latitude_matrix_deg,
                longitudes_deg=narr_longitude_matrix_deg,
                model_name=nwp_model_utils.NARR_MODEL_NAME)
        )
    else:
        narr_cosine_matrix = None
        narr_sine_matrix = None

    field_names = [
        era5_input.field_name_raw_to_processed(
            raw_field_name=f, earth_relative=False
        )
        for f in raw_field_names
    ]

    field_names = numpy.array(field_names)
    dewpoint_indices = numpy.where(
        field_names == predictor_utils.DEWPOINT_NAME
    )[0]
    field_names[dewpoint_indices] = predictor_utils.SPECIFIC_HUMIDITY_NAME
    field_names = field_names.tolist()

    num_times = len(valid_times_unix_sec)
    num_fields = len(raw_field_names)

    this_predictor_dict = None
    era5_x_matrix_metres = None
    era5_y_matrix_metres = None

    for i in range(num_times):
        this_time_string = time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], LOG_MESSAGE_TIME_FORMAT
        )
        this_year = int(time_conversion.unix_sec_to_string(
            valid_times_unix_sec[i], YEAR_FORMAT
        ))

        one_time_data_matrix = None

        for j in range(num_fields):
            this_raw_file_name = era5_input.find_file(
                top_directory_name=top_input_dir_name, year=this_year,
                raw_field_name=raw_field_names[j],
                has_surface_data=(
                    pressure_levels_mb[j] == DUMMY_SURFACE_PRESSURE_MB
                )
            )

            print('Reading {0:s} at {1:d} mb and {2:s} from: "{3:s}"...'.format(
                field_names[j], pressure_levels_mb[j], this_time_string,
                this_raw_file_name
            ))

            this_predictor_dict = era5_input.read_file(
                netcdf_file_name=this_raw_file_name,
                first_time_unix_sec=valid_times_unix_sec[i],
                last_time_unix_sec=valid_times_unix_sec[i],
                pressure_level_mb=pressure_levels_mb[j]
            )

            if one_time_data_matrix is None:
                num_grid_rows = this_predictor_dict[
                    predictor_utils.DATA_MATRIX_KEY
                ].shape[1]

                num_grid_columns = this_predictor_dict[
                    predictor_utils.DATA_MATRIX_KEY
                ].shape[2]

                one_time_data_matrix = numpy.full(
                    (1, num_grid_rows, num_grid_columns, num_fields),
                    numpy.nan
                )

            print(this_predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape)

            one_time_data_matrix[0, ..., j] = this_predictor_dict[
                predictor_utils.DATA_MATRIX_KEY
            ][0, ..., 0]

        for j in range(num_fields):
            if raw_field_names[j] != era5_input.DEWPOINT_NAME_RAW:
                continue

            print((
                'Converting {0:d}-mb dewpoints to specific humidities...'
            ).format(
                pressure_levels_mb[j]
            ))

            if pressure_levels_mb[j] == DUMMY_SURFACE_PRESSURE_MB:
                this_pressure_index = numpy.where(numpy.logical_and(
                    numpy.array(raw_field_names) ==
                    era5_input.PRESSURE_NAME_RAW,
                    pressure_levels_mb == pressure_levels_mb[j]
                ))[0][0]

                this_pressure_matrix_pa = one_time_data_matrix[
                    ..., this_pressure_index
                ]
            else:
                this_pressure_matrix_pa = numpy.full(
                    one_time_data_matrix.shape[:-1],
                    MB_TO_PASCALS * pressure_levels_mb[j]
                )

            one_time_data_matrix[..., j] = (
                moisture_conversions.dewpoint_to_specific_humidity(
                    dewpoints_kelvins=one_time_data_matrix[..., j],
                    total_pressures_pascals=this_pressure_matrix_pa
                )
            )

        one_time_predictor_dict = {
            predictor_utils.DATA_MATRIX_KEY: one_time_data_matrix,
            predictor_utils.VALID_TIMES_KEY: valid_times_unix_sec[[i]],
            predictor_utils.LATITUDES_KEY:
                this_predictor_dict[predictor_utils.LATITUDES_KEY],
            predictor_utils.LONGITUDES_KEY:
                this_predictor_dict[predictor_utils.LONGITUDES_KEY],
            predictor_utils.FIELD_NAMES_KEY: field_names,
            predictor_utils.PRESSURE_LEVELS_KEY: pressure_levels_mb
        }

        print('\n')
        one_time_predictor_dict = era5_input.interp_to_narr_grid(
            predictor_dict=one_time_predictor_dict, grid_name=grid_name,
            era5_x_matrix_metres=era5_x_matrix_metres,
            era5_y_matrix_metres=era5_y_matrix_metres
        )
        print('\n')

        for j in range(num_fields):
            if raw_field_names[j] != era5_input.U_WIND_NAME_RAW:
                continue

            print('Rotating {0:d}-mb winds to grid-relative...'.format(
                pressure_levels_mb[j]
            ))

            this_v_wind_index = numpy.where(numpy.logical_and(
                numpy.array(raw_field_names) == era5_input.V_WIND_NAME_RAW,
                pressure_levels_mb == pressure_levels_mb[j]
            ))[0][0]

            this_u_wind_matrix = one_time_predictor_dict[
                predictor_utils.DATA_MATRIX_KEY
            ][0, ..., j]

            this_v_wind_matrix = one_time_predictor_dict[
                predictor_utils.DATA_MATRIX_KEY
            ][0, ..., this_v_wind_index]

            this_u_wind_matrix, this_v_wind_matrix = (
                nwp_model_utils.rotate_winds_to_grid_relative(
                    u_winds_earth_relative_m_s01=this_u_wind_matrix,
                    v_winds_earth_relative_m_s01=this_v_wind_matrix,
                    rotation_angle_cosines=narr_cosine_matrix,
                    rotation_angle_sines=narr_sine_matrix)
            )

            one_time_predictor_dict[predictor_utils.DATA_MATRIX_KEY][
                0, ..., j
            ] = this_u_wind_matrix

            one_time_predictor_dict[predictor_utils.DATA_MATRIX_KEY][
                0, ..., this_v_wind_index
            ] = this_v_wind_matrix

        this_processed_file_name = predictor_io.find_file(
            top_directory_name=top_output_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False
        )

        print('Writing processed ERA5 data on NARR grid to: "{0:s}"...'.format(
            this_processed_file_name
        ))

        predictor_io.write_file(
            netcdf_file_name=this_processed_file_name,
            predictor_dict=one_time_predictor_dict
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        extended_grid=bool(getattr(INPUT_ARG_OBJECT, EXTENDED_GRID_ARG_NAME)),
        raw_field_names=getattr(INPUT_ARG_OBJECT, RAW_FIELDS_ARG_NAME),
        pressure_levels_mb=numpy.array(
            getattr(INPUT_ARG_OBJECT, PRESSURE_LEVELS_ARG_NAME), dtype=int
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
