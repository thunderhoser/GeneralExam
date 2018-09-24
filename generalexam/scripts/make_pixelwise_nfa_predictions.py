"""Uses NFA (numerical frontal analysis) to predict front type at each pixel."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_utils import nfa
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import utils as general_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

# TODO(thunderhoser): Deal with smoothing.

INPUT_TIME_FORMAT = '%Y%m%d%H'
NARR_TIME_INTERVAL_SEC = 10800
NARR_PREDICTOR_NAMES = [
    processed_narr_io.WET_BULB_THETA_NAME,
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME
]

FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
RANDOMIZE_TIMES_ARG_NAME = 'randomize_times'
NUM_TIMES_ARG_NAME = 'num_times'
WF_PERCENTILE_ARG_NAME = 'warm_front_percentile'
CF_PERCENTILE_ARG_NAME = 'cold_front_percentile'
NUM_CLOSING_ITERS_ARG_NAME = 'num_closing_iters'
PRESSURE_LEVEL_ARG_NAME = 'pressure_level_mb'
NARR_DIRECTORY_ARG_NAME = 'input_narr_dir_name'
NARR_MASK_FILE_ARG_NAME = 'input_narr_mask_file_name'
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
    'Predictors (listed below) will be taken from this pressure level '
    '(millibars).\n{0:s}'
).format(str(NARR_PREDICTOR_NAMES))

NARR_DIRECTORY_HELP_STRING = (
    'Name of top-level NARR directory (predictors will be read from here).  '
    'Files therein will be found by `processed_narr_io.find_file_for_one_time` '
    'and read by `processed_narr_io.read_fields_from_file`.')

NARR_MASK_FILE_HELP_STRING = (
    'Pickle file with NARR mask (will be read by `machine_learning_utils.'
    'read_narr_mask`).  Predictions will not be made for masked grid cells.  If'
    ' you do not want a mask, make this the empty string ("").')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each time step, gridded predictions will be'
    ' written here by `nfa.write_gridded_predictions`, to a location determined'
    ' by `nfa.find_gridded_prediction_file`.')

DEFAULT_WARM_FRONT_PERCENTILE = 97.
DEFAULT_COLD_FRONT_PERCENTILE = 97.
DEFAULT_NUM_CLOSING_ITERS = 3
DEFAULT_PRESSURE_LEVEL_MB = 850
TOP_NARR_DIR_NAME_DEFAULT = '/condo/swatwork/ralager/narr_data/processed'
DEFAULT_NARR_MASK_FILE_NAME = (
    '/condo/swatwork/ralager/fronts/narr_grids/narr_mask.p')

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
    '--' + NARR_DIRECTORY_ARG_NAME, type=str, required=False,
    default=TOP_NARR_DIR_NAME_DEFAULT, help=NARR_DIRECTORY_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NARR_MASK_FILE_ARG_NAME, type=str, required=False,
    default=DEFAULT_NARR_MASK_FILE_NAME, help=NARR_MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(first_time_string, last_time_string, randomize_times, num_times,
         warm_front_percentile, cold_front_percentile, num_closing_iters,
         pressure_level_mb, top_narr_directory_name, narr_mask_file_name,
         output_dir_name):
    """Uses NFA (numerical frontal analysis) to predict front type at each px.

    This is effectively the main method.

    :param first_time_string: See documentation at top of file.
    :param last_time_string: Same.
    :param randomize_times: Same.
    :param num_times: Same.
    :param warm_front_percentile: Same.
    :param cold_front_percentile: Same.
    :param num_closing_iters: Same.
    :param pressure_level_mb: Same.
    :param top_narr_directory_name: Same.
    :param narr_mask_file_name: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SEC, include_endpoint=True)

    if randomize_times:
        error_checking.assert_is_leq(
            num_times, len(valid_times_unix_sec))
        numpy.random.shuffle(valid_times_unix_sec)
        valid_times_unix_sec = valid_times_unix_sec[:num_times]

    if narr_mask_file_name == '':
        num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
            model_name=nwp_model_utils.NARR_MODEL_NAME)
        narr_mask_matrix = numpy.full(
            (num_grid_rows, num_grid_columns), 1, dtype=int)
    else:
        print 'Reading mask from: "{0:s}"...\n'.format(narr_mask_file_name)
        narr_mask_matrix = ml_utils.read_narr_mask(narr_mask_file_name)

    x_spacing_metres, y_spacing_metres = nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME)

    num_times = len(valid_times_unix_sec)
    for i in range(num_times):
        this_wet_bulb_theta_file_name = (
            processed_narr_io.find_file_for_one_time(
                top_directory_name=top_narr_directory_name,
                field_name=processed_narr_io.WET_BULB_THETA_NAME,
                pressure_level_mb=pressure_level_mb,
                valid_time_unix_sec=valid_times_unix_sec[i])
        )

        print 'Reading data from: "{0:s}"...'.format(
            this_wet_bulb_theta_file_name)
        this_wet_bulb_theta_matrix_kelvins = (
            processed_narr_io.read_fields_from_file(
                this_wet_bulb_theta_file_name)[0][0, ...]
        )
        this_wet_bulb_theta_matrix_kelvins = general_utils.fill_nans(
            this_wet_bulb_theta_matrix_kelvins)

        this_u_wind_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_narr_directory_name,
            field_name=processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_u_wind_file_name)
        this_u_wind_matrix_m_s01 = processed_narr_io.read_fields_from_file(
            this_u_wind_file_name)[0][0, ...]
        this_u_wind_matrix_m_s01 = general_utils.fill_nans(
            this_u_wind_matrix_m_s01)

        this_v_wind_file_name = processed_narr_io.find_file_for_one_time(
            top_directory_name=top_narr_directory_name,
            field_name=processed_narr_io.V_WIND_GRID_RELATIVE_NAME,
            pressure_level_mb=pressure_level_mb,
            valid_time_unix_sec=valid_times_unix_sec[i])

        print 'Reading data from: "{0:s}"...'.format(this_v_wind_file_name)
        this_v_wind_matrix_m_s01 = processed_narr_io.read_fields_from_file(
            this_v_wind_file_name)[0][0, ...]
        this_v_wind_matrix_m_s01 = general_utils.fill_nans(
            this_v_wind_matrix_m_s01)

        this_tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
            thermal_field_matrix_kelvins=this_wet_bulb_theta_matrix_kelvins,
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)
        this_tfp_matrix_kelvins_m02[narr_mask_matrix == 0] = 0.

        this_proj_velocity_matrix_m_s01 = nfa.project_wind_to_thermal_gradient(
            u_matrix_grid_relative_m_s01=this_u_wind_matrix_m_s01,
            v_matrix_grid_relative_m_s01=this_v_wind_matrix_m_s01,
            thermal_field_matrix_kelvins=this_wet_bulb_theta_matrix_kelvins,
            x_spacing_metres=x_spacing_metres,
            y_spacing_metres=y_spacing_metres)

        this_locating_var_matrix_m01_s01 = nfa.get_locating_variable(
            tfp_matrix_kelvins_m02=this_tfp_matrix_kelvins_m02,
            projected_velocity_matrix_m_s01=this_proj_velocity_matrix_m_s01)

        this_predicted_label_matrix = nfa.get_front_types(
            locating_var_matrix_m01_s01=this_locating_var_matrix_m01_s01,
            warm_front_percentile=warm_front_percentile,
            cold_front_percentile=cold_front_percentile)

        this_predicted_label_matrix = front_utils.close_frontal_image(
            ternary_image_matrix=this_predicted_label_matrix,
            num_iterations=num_closing_iters)

        this_prediction_file_name = nfa.find_gridded_prediction_file(
            directory_name=output_dir_name,
            first_valid_time_unix_sec=valid_times_unix_sec[i],
            last_valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing gridded predictions to file: "{0:s}"...\n'.format(
            this_prediction_file_name)

        nfa.write_gridded_predictions(
            pickle_file_name=this_prediction_file_name,
            predicted_label_matrix=numpy.expand_dims(
                this_predicted_label_matrix, axis=0),
            valid_times_unix_sec=valid_times_unix_sec[[i]],
            narr_mask_matrix=narr_mask_matrix)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        randomize_times=bool(getattr(
            INPUT_ARG_OBJECT, RANDOMIZE_TIMES_ARG_NAME)),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        warm_front_percentile=getattr(INPUT_ARG_OBJECT, WF_PERCENTILE_ARG_NAME),
        cold_front_percentile=getattr(INPUT_ARG_OBJECT, CF_PERCENTILE_ARG_NAME),
        num_closing_iters=getattr(INPUT_ARG_OBJECT, NUM_CLOSING_ITERS_ARG_NAME),
        pressure_level_mb=getattr(INPUT_ARG_OBJECT, PRESSURE_LEVEL_ARG_NAME),
        top_narr_directory_name=getattr(
            INPUT_ARG_OBJECT, NARR_DIRECTORY_ARG_NAME),
        narr_mask_file_name=getattr(INPUT_ARG_OBJECT, NARR_MASK_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME))
