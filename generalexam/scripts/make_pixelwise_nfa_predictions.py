"""Uses NFA (numerical frontal analysis) to predict front type at each pixel."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import nfa
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import predictor_utils
from generalexam.ge_utils import utils as general_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

RANDOM_SEED = 6695

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

VALID_THERMAL_FIELD_NAMES = [
    predictor_utils.TEMPERATURE_NAME,
    predictor_utils.WET_BULB_THETA_NAME,
    predictor_utils.SPECIFIC_HUMIDITY_NAME
]

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RANDOMIZE_TIMES_ARG_NAME = 'randomize_times'
NUM_TIMES_ARG_NAME = 'num_times'
THERMAL_FIELD_ARG_NAME = 'thermal_field_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_pixels'
WF_PERCENTILE_ARG_NAME = 'warm_front_percentile'
CF_PERCENTILE_ARG_NAME = 'cold_front_percentile'
NUM_CLOSING_ITERS_ARG_NAME = 'num_closing_iters'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Target times will be randomly drawn from the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

RANDOMIZE_TIMES_HELP_STRING = (
    'Boolean flag.  If 1, target times will be sampled randomly from '
    '`{0:s}`...`{1:s}`.  If 0, all times from `{0:s}`...`{1:s}` will be used.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME, NUM_TIMES_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    '[used iff {0:s} = 1] Number of target times (to be sampled from '
    '`{1:s}`...`{2:s}`).'
).format(RANDOMIZE_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

THERMAL_FIELD_HELP_STRING = (
    'Thermal field.  Both this field and the wind field will be used as '
    'predictors.  Valid thermal fields are listed below.\n{0:s}'
).format(str(VALID_THERMAL_FIELD_NAMES))

SMOOTHING_RADIUS_HELP_STRING = (
    'Smoothing radius (standard deviation of Gaussian kernel).  Will be applied'
    ' to both thermal and wind fields.'
)

WF_PERCENTILE_HELP_STRING = (
    'Used to locate warm fronts.  For grid cell [i, j] to be considered part of'
    ' a warm front, its locating variable must be <= the [q]th percentile of '
    'all non-positive values in the grid, where q = `100 - {0:s}`.  For more '
    'details, see `nfa.get_front_types`.'
).format(WF_PERCENTILE_ARG_NAME)

CF_PERCENTILE_HELP_STRING = (
    'Used to locate cold fronts.  For grid cell [i, j] to be considered part of'
    ' a cold front, its locating variable must be >= the [q]th percentile of '
    'all non-negative values in the grid, where q = `{0:s}`.  For more details,'
    ' see `nfa.get_front_types`.'
).format(CF_PERCENTILE_ARG_NAME)

NUM_CLOSING_ITERS_HELP_STRING = (
    'Number of binary-closing iterations.  Will be applied to both warm-front '
    'and cold-front labels independently.  More iterations lead to larger '
    'frontal regions.')

PRESSURE_LEVEL_HELP_STRING = (
    'Pressure level (millibars).  Both thermal and wind fields will be taken '
    'from this level only.')

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.')

MASK_FILE_HELP_STRING = (
    'Path to mask file (predictions will not be made for masked grid cells).  '
    'Will be read by `machine_learning_utils.read_narr_mask`.  If you do not '
    'want a mask, leave this empty.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each time step, gridded predictions will be'
    ' written here by `nfa.write_gridded_predictions`, to a location determined'
    ' by `nfa.find_prediction_file`.')

DEFAULT_THERMAL_FIELD_NAME = predictor_utils.WET_BULB_THETA_NAME
DEFAULT_SMOOTHING_RADIUS_PIXELS = 1.
DEFAULT_WARM_FRONT_PERCENTILE = 96.
DEFAULT_COLD_FRONT_PERCENTILE = 96.
DEFAULT_NUM_CLOSING_ITERS = 2
DEFAULT_PRESSURE_LEVEL_MB = 900

TOP_PREDICTOR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/era5_data/processed'
DEFAULT_MASK_FILE_NAME = '/condo/swatwork/ralager/fronts_netcdf/era5_mask.p'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + RANDOMIZE_TIMES_ARG_NAME, type=int, required=False, default=1,
    help=RANDOMIZE_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + THERMAL_FIELD_ARG_NAME, type=str, required=False,
    default=DEFAULT_THERMAL_FIELD_NAME, help=THERMAL_FIELD_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=DEFAULT_SMOOTHING_RADIUS_PIXELS, help=SMOOTHING_RADIUS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + WF_PERCENTILE_ARG_NAME, type=float, required=False,
    default=DEFAULT_WARM_FRONT_PERCENTILE, help=WF_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + CF_PERCENTILE_ARG_NAME, type=float, required=False,
    default=DEFAULT_COLD_FRONT_PERCENTILE, help=CF_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CLOSING_ITERS_ARG_NAME, type=int, required=False,
    default=DEFAULT_NUM_CLOSING_ITERS, help=NUM_CLOSING_ITERS_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_LEVEL_ARG_NAME, type=int, required=False,
    default=DEFAULT_PRESSURE_LEVEL_MB, help=PRESSURE_LEVEL_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_PREDICTOR_DIR_NAME_DEFAULT, help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_MASK_FILE_NAME, help=MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(first_time_string, last_time_string, randomize_times, num_times,
         thermal_field_name, smoothing_radius_pixels, warm_front_percentile,
         cold_front_percentile, num_closing_iters, pressure_level_mb,
         top_predictor_dir_name, mask_file_name, output_dir_name):
    """Uses NFA (numerical frontal analysis) to predict front type at each px.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param randomize_times: Same.
    :param num_times: Same.
    :param thermal_field_name: Same.
    :param smoothing_radius_pixels: Same.
    :param warm_front_percentile: Same.
    :param cold_front_percentile: Same.
    :param num_closing_iters: Same.
    :param pressure_level_mb: Same.
    :param top_predictor_dir_name: Same.
    :param mask_file_name: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if
        `thermal_field_name not in VALID_THERMAL_FIELD_NAMES`.
    """

    cutoff_radius_pixels = 4 * smoothing_radius_pixels
    if mask_file_name in ['', 'None']:
        mask_file_name = None

    if thermal_field_name not in VALID_THERMAL_FIELD_NAMES:
        error_string = (
            '\n{0:s}\nValid thermal fields (listed above) do not include '
            '"{1:s}".'
        ).format(str(VALID_THERMAL_FIELD_NAMES), thermal_field_name)

        raise ValueError(error_string)

    field_names_to_read = [
        thermal_field_name, predictor_utils.U_WIND_GRID_RELATIVE_NAME,
        predictor_utils.V_WIND_GRID_RELATIVE_NAME
    ]

    pressure_levels_to_read_mb = numpy.full(
        len(field_names_to_read), pressure_level_mb, dtype=int
    )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    if randomize_times:
        error_checking.assert_is_leq(num_times, len(valid_times_unix_sec))

        numpy.random.seed(RANDOM_SEED)
        numpy.random.shuffle(valid_times_unix_sec)
        valid_times_unix_sec = valid_times_unix_sec[:num_times]

    if mask_file_name is None:
        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME,
            grid_name=nwp_model_utils.NAME_OF_221GRID)

        mask_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), 1, dtype=int)
    else:
        print 'Reading mask from: "{0:s}"...\n'.format(mask_file_name)
        mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]

    x_spacing_metres, y_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_221GRID)

    num_times = len(valid_times_unix_sec)

    for i in range(num_times):
        this_file_name = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading predictors from: "{0:s}"...'.format(this_file_name)
        this_predictor_dict = predictor_io.read_file(
            netcdf_file_name=this_file_name,
            pressure_levels_to_keep_mb=pressure_levels_to_read_mb,
            field_names_to_keep=field_names_to_read)

        this_thermal_matrix_kelvins = this_predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][0, ..., 0]

        this_thermal_matrix_kelvins = general_utils.fill_nans(
            this_thermal_matrix_kelvins)
        this_thermal_matrix_kelvins = nfa.gaussian_smooth_2d_field(
            field_matrix=this_thermal_matrix_kelvins,
            standard_deviation_pixels=smoothing_radius_pixels,
            cutoff_radius_pixels=cutoff_radius_pixels)

        this_u_wind_matrix_m_s01 = this_predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][0, ..., 1]

        this_u_wind_matrix_m_s01 = general_utils.fill_nans(
            this_u_wind_matrix_m_s01)
        this_u_wind_matrix_m_s01 = nfa.gaussian_smooth_2d_field(
            field_matrix=this_u_wind_matrix_m_s01,
            standard_deviation_pixels=smoothing_radius_pixels,
            cutoff_radius_pixels=cutoff_radius_pixels)

        this_v_wind_matrix_m_s01 = this_predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][0, ..., 2]

        this_v_wind_matrix_m_s01 = general_utils.fill_nans(
            this_v_wind_matrix_m_s01)
        this_v_wind_matrix_m_s01 = nfa.gaussian_smooth_2d_field(
            field_matrix=this_v_wind_matrix_m_s01,
            standard_deviation_pixels=smoothing_radius_pixels,
            cutoff_radius_pixels=cutoff_radius_pixels)

        this_tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
            thermal_field_matrix_kelvins=this_thermal_matrix_kelvins,
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)
        this_tfp_matrix_kelvins_m02[mask_matrix == 0] = 0.

        this_proj_velocity_matrix_m_s01 = nfa.project_wind_to_thermal_gradient(
            u_matrix_grid_relative_m_s01=this_u_wind_matrix_m_s01,
            v_matrix_grid_relative_m_s01=this_v_wind_matrix_m_s01,
            thermal_field_matrix_kelvins=this_thermal_matrix_kelvins,
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)

        this_locating_var_matrix_m01_s01 = nfa.get_locating_variable(
            tfp_matrix_kelvins_m02=this_tfp_matrix_kelvins_m02,
            projected_velocity_matrix_m_s01=this_proj_velocity_matrix_m_s01)

        this_predicted_label_matrix = nfa.get_front_types(
            locating_var_matrix_m01_s01=this_locating_var_matrix_m01_s01,
            warm_front_percentile=warm_front_percentile,
            cold_front_percentile=cold_front_percentile)

        this_predicted_label_matrix = front_utils.close_gridded_labels(
            ternary_label_matrix=this_predicted_label_matrix,
            num_iterations=num_closing_iters)

        this_prediction_file_name = nfa.find_prediction_file(
            directory_name=output_dir_name,
            first_valid_time_unix_sec=valid_times_unix_sec[i],
            last_valid_time_unix_sec=valid_times_unix_sec[i],
            ensembled=False, raise_error_if_missing=False)

        print 'Writing gridded predictions to file: "{0:s}"...\n'.format(
            this_prediction_file_name)

        nfa.write_gridded_predictions(
            pickle_file_name=this_prediction_file_name,
            predicted_label_matrix=numpy.expand_dims(
                this_predicted_label_matrix, axis=0),
            valid_times_unix_sec=valid_times_unix_sec[[i]],
            narr_mask_matrix=mask_matrix,
            pressure_level_mb=pressure_level_mb,
            smoothing_radius_pixels=smoothing_radius_pixels,
            cutoff_radius_pixels=cutoff_radius_pixels,
            warm_front_percentile=warm_front_percentile,
            cold_front_percentile=cold_front_percentile,
            num_closing_iters=num_closing_iters)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        randomize_times=bool(getattr(
            INPUT_ARG_OBJECT, RANDOMIZE_TIMES_ARG_NAME)),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        thermal_field_name=getattr(INPUT_ARG_OBJECT, THERMAL_FIELD_ARG_NAME),
        smoothing_radius_pixels=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME),
        warm_front_percentile=getattr(INPUT_ARG_OBJECT, WF_PERCENTILE_ARG_NAME),
        cold_front_percentile=getattr(INPUT_ARG_OBJECT, CF_PERCENTILE_ARG_NAME),
        num_closing_iters=getattr(INPUT_ARG_OBJECT, NUM_CLOSING_ITERS_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
