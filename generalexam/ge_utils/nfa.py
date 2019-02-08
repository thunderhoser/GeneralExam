"""Implements NFA (numerical frontal analysis) methods.

--- REFERENCES ---

Renard, R., and L. Clarke, 1965: "Experiments in numerical objective frontal
    analysis". Monthly Weather Review, 93 (9), 547-556.
"""

import pickle
import os.path
import numpy
from scipy.ndimage.filters import gaussian_filter
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6

DEFAULT_FRONT_PERCENTILE = 97.
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

PREDICTED_LABELS_KEY = 'predicted_label_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
NARR_MASK_KEY = 'narr_mask_matrix'
PRESSURE_LEVEL_KEY = 'pressure_level_mb'
SMOOTHING_RADIUS_KEY = 'smoothing_radius_pixels'
CUTOFF_RADIUS_KEY = 'cutoff_radius_pixels'
WF_PERCENTILE_KEY = 'warm_front_percentile'
CF_PERCENTILE_KEY = 'cold_front_percentile'
NUM_CLOSING_ITERS_KEY = 'num_closing_iters'

CLASS_PROBABILITIES_KEY = 'class_probability_matrix'
MODEL_DIRECTORIES_KEY = 'prediction_dir_name_by_model'
MODEL_WEIGHTS_KEY = 'model_weights'

ENSEMBLE_FILE_KEYS = [
    CLASS_PROBABILITIES_KEY, VALID_TIMES_KEY, NARR_MASK_KEY,
    MODEL_DIRECTORIES_KEY, MODEL_WEIGHTS_KEY
]


def _get_2d_gradient(field_matrix, x_spacing_metres, y_spacing_metres):
    """Computes gradient of 2-D field at each point

    M = number of rows in grid
    N = number of columns in grid

    :param field_matrix: M-by-N numpy array with values in field.
    :param x_spacing_metres: Spacing between grid points in adjacent columns.
    :param y_spacing_metres: Spacing between grid points in adjacent rows.
    :return: x_gradient_matrix_m01: M-by-N numpy array with x-component of
        gradient vector at each grid point.  Units are (units of `field_matrix`)
        per metre.
    :return: y_gradient_matrix_m01: Same but for y-component of gradient.
    """

    y_gradient_matrix_m01, x_gradient_matrix_m01 = numpy.gradient(
        field_matrix, edge_order=1)

    x_gradient_matrix_m01 = x_gradient_matrix_m01 / x_spacing_metres
    y_gradient_matrix_m01 = y_gradient_matrix_m01 / y_spacing_metres
    return x_gradient_matrix_m01, y_gradient_matrix_m01


def gaussian_smooth_2d_field(
        field_matrix, standard_deviation_pixels, cutoff_radius_pixels):
    """Applies Gaussian smoother to 2-D field.

    M = number of rows in grid
    N = number of columns in grid

    :param field_matrix: M-by-N numpy array with values in field.
    :param standard_deviation_pixels: Standard deviation of Gaussian kernel
        (pixels).
    :param cutoff_radius_pixels: Cutoff radius of Gaussian kernel (pixels).
    :return: field_matrix: Smoothed version of input.
    """

    error_checking.assert_is_numpy_array_without_nan(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)
    error_checking.assert_is_greater(standard_deviation_pixels, 0.)
    error_checking.assert_is_greater(
        cutoff_radius_pixels, standard_deviation_pixels)

    return gaussian_filter(
        input=field_matrix, sigma=standard_deviation_pixels, order=0,
        mode='reflect', truncate=cutoff_radius_pixels)


def get_thermal_front_param(
        thermal_field_matrix_kelvins, x_spacing_metres, y_spacing_metres):
    """Computes thermal front parameter (TFP) at each grid point.

    TFP is defined in Renard and Clarke (1965).

    M = number of rows in grid
    N = number of columns in grid

    :param thermal_field_matrix_kelvins: M-by-N numpy array with values of
        thermal variable.  This can be any thermal variable ([potential]
        temperature, wet-bulb [potential] temperature, equivalent [potential]
        temperature, etc.).
    :param x_spacing_metres: Spacing between grid points in adjacent columns.
    :param y_spacing_metres: Spacing between grid points in adjacent rows.
    :return: tfp_matrix_kelvins_m02: M-by-N numpy array with TFP at each grid
        point. Units are Kelvins per m^2.
    """

    error_checking.assert_is_numpy_array_without_nan(
        thermal_field_matrix_kelvins)
    error_checking.assert_is_greater_numpy_array(
        thermal_field_matrix_kelvins, 0.)
    error_checking.assert_is_numpy_array(
        thermal_field_matrix_kelvins, num_dimensions=2)

    error_checking.assert_is_greater(x_spacing_metres, 0.)
    error_checking.assert_is_greater(y_spacing_metres, 0.)

    x_grad_matrix_kelvins_m01, y_grad_matrix_kelvins_m01 = _get_2d_gradient(
        field_matrix=thermal_field_matrix_kelvins,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)

    grad_magnitude_matrix_kelvins_m01 = numpy.sqrt(
        x_grad_matrix_kelvins_m01 ** 2 + y_grad_matrix_kelvins_m01 ** 2)
    (x_grad_grad_matrix_kelvins_m02, y_grad_grad_matrix_kelvins_m02
    ) = _get_2d_gradient(
        field_matrix=grad_magnitude_matrix_kelvins_m01,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)

    first_matrix = (
        -x_grad_grad_matrix_kelvins_m02 *
        x_grad_matrix_kelvins_m01 / grad_magnitude_matrix_kelvins_m01)
    first_matrix[numpy.isnan(first_matrix)] = 0.

    second_matrix = (
        -y_grad_grad_matrix_kelvins_m02 *
        y_grad_matrix_kelvins_m01 / grad_magnitude_matrix_kelvins_m01)
    second_matrix[numpy.isnan(second_matrix)] = 0.

    return first_matrix + second_matrix


def project_wind_to_thermal_gradient(
        u_matrix_grid_relative_m_s01, v_matrix_grid_relative_m_s01,
        thermal_field_matrix_kelvins, x_spacing_metres, y_spacing_metres):
    """At each grid point, projects wind to direction of thermal gradient.

    M = number of rows in grid
    N = number of columns in grid

    :param u_matrix_grid_relative_m_s01: M-by-N numpy array of grid-relative
        u-wind (in the direction of increasing column number, or towards the
        right).  Units are metres per second.
    :param v_matrix_grid_relative_m_s01: M-by-N numpy array of grid-relative
        v-wind (in the direction of increasing row number, or towards the
        bottom).
    :param thermal_field_matrix_kelvins: See doc for `get_thermal_front_param`.
    :param x_spacing_metres: Same.
    :param y_spacing_metres: Same.
    :return: projected_velocity_matrix_m_s01: M-by-N numpy array with wind
        velocity in direction of thermal gradient.  Positive (negative) values
        mean that the wind is blowing towards warmer (cooler) air.
    """

    error_checking.assert_is_numpy_array_without_nan(
        u_matrix_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(
        u_matrix_grid_relative_m_s01, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(
        v_matrix_grid_relative_m_s01)
    error_checking.assert_is_numpy_array(
        v_matrix_grid_relative_m_s01,
        exact_dimensions=numpy.array(u_matrix_grid_relative_m_s01.shape))

    error_checking.assert_is_numpy_array_without_nan(
        thermal_field_matrix_kelvins)
    error_checking.assert_is_greater_numpy_array(
        thermal_field_matrix_kelvins, 0.)
    error_checking.assert_is_numpy_array(
        thermal_field_matrix_kelvins,
        exact_dimensions=numpy.array(u_matrix_grid_relative_m_s01.shape))

    x_grad_matrix_kelvins_m01, y_grad_matrix_kelvins_m01 = _get_2d_gradient(
        field_matrix=thermal_field_matrix_kelvins,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres)
    y_grad_matrix_kelvins_m01 = y_grad_matrix_kelvins_m01
    grad_magnitude_matrix_kelvins_m01 = numpy.sqrt(
        x_grad_matrix_kelvins_m01 ** 2 + y_grad_matrix_kelvins_m01 ** 2)

    first_matrix = (
        u_matrix_grid_relative_m_s01 *
        x_grad_matrix_kelvins_m01 / grad_magnitude_matrix_kelvins_m01)
    first_matrix[numpy.isnan(first_matrix)] = 0.

    second_matrix = (
        v_matrix_grid_relative_m_s01 *
        y_grad_matrix_kelvins_m01 / grad_magnitude_matrix_kelvins_m01)
    second_matrix[numpy.isnan(second_matrix)] = 0.

    return first_matrix + second_matrix


def get_locating_variable(
        tfp_matrix_kelvins_m02, projected_velocity_matrix_m_s01):
    """Computes locating variable at each grid point.

    The "locating variable" is the product of the absolute TFP (thermal front
    parameter) and projected wind velocity (in the direction of the thermal
    gradient).  Large positive values indicate the presence of a cold front,
    while large negative values indicate the presence of a warm front.

    M = number of rows in grid
    N = number of columns in grid

    :param tfp_matrix_kelvins_m02: M-by-N numpy array created by
        `get_thermal_front_param`.
    :param projected_velocity_matrix_m_s01: M-by-N numpy array created by
        `project_wind_to_thermal_gradient`.
    :return: locating_var_matrix_m01_s01: M-by-N numpy array with locating
        variable (units of m^-1 s^-1) at each grid point.
    """

    error_checking.assert_is_numpy_array_without_nan(tfp_matrix_kelvins_m02)
    error_checking.assert_is_numpy_array(
        tfp_matrix_kelvins_m02, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(
        projected_velocity_matrix_m_s01)
    error_checking.assert_is_numpy_array(
        projected_velocity_matrix_m_s01,
        exact_dimensions=numpy.array(tfp_matrix_kelvins_m02.shape))

    return (
        numpy.absolute(tfp_matrix_kelvins_m02) * projected_velocity_matrix_m_s01
    )


def get_front_types(locating_var_matrix_m01_s01,
                    warm_front_percentile=DEFAULT_FRONT_PERCENTILE,
                    cold_front_percentile=DEFAULT_FRONT_PERCENTILE):
    """Infers front type at each grid cell.

    M = number of rows in grid
    N = number of columns in grid

    :param locating_var_matrix_m01_s01: M-by-N numpy array created by
        `get_locating_variable`.
    :param warm_front_percentile: Used to locate warm fronts.  For grid cell
        [i, j] to be considered part of a warm front, its locating value must be
        <= the [q]th percentile of all non-positive values in the grid, where
        q = `100 - warm_front_percentile`.
    :param cold_front_percentile: Used to locate cold fronts.  For grid cell
        [i, j] to be considered part of a cold front, its locating value must be
        >= the [q]th percentile of all non-negative values in the grid, where
        q = `cold_front_percentile`.
    :return: predicted_label_matrix: M-by-N numpy array, where the value at each
        grid cell is from the list `front_utils.VALID_INTEGER_IDS`.
    """

    error_checking.assert_is_numpy_array_without_nan(
        locating_var_matrix_m01_s01)
    error_checking.assert_is_numpy_array(
        locating_var_matrix_m01_s01, num_dimensions=2)

    error_checking.assert_is_greater(warm_front_percentile, 0.)
    error_checking.assert_is_less_than(warm_front_percentile, 100.)
    error_checking.assert_is_greater(cold_front_percentile, 0.)
    error_checking.assert_is_less_than(cold_front_percentile, 100.)

    warm_front_threshold_m01_s01 = numpy.percentile(
        locating_var_matrix_m01_s01[locating_var_matrix_m01_s01 <= 0],
        100 - warm_front_percentile)
    cold_front_threshold_m01_s01 = numpy.percentile(
        locating_var_matrix_m01_s01[locating_var_matrix_m01_s01 >= 0],
        cold_front_percentile)

    predicted_label_matrix = numpy.full(
        locating_var_matrix_m01_s01.shape, front_utils.NO_FRONT_INTEGER_ID,
        dtype=int)
    predicted_label_matrix[
        locating_var_matrix_m01_s01 <= warm_front_threshold_m01_s01
    ] = front_utils.WARM_FRONT_INTEGER_ID
    predicted_label_matrix[
        locating_var_matrix_m01_s01 >= cold_front_threshold_m01_s01
    ] = front_utils.COLD_FRONT_INTEGER_ID

    return predicted_label_matrix


def find_prediction_file(
        directory_name, first_valid_time_unix_sec, last_valid_time_unix_sec,
        ensembled=False, raise_error_if_missing=True):
    """Finds Pickle file with gridded predictions.

    :param directory_name: Name of directory.
    :param first_valid_time_unix_sec: First time in file.
    :param last_valid_time_unix_sec: Last time in file.
    :param ensembled: Boolean flag.  If True, file should contain ensembled
        probabilistic predictions, written by `write_gridded_prediction_file`.
        If False, should contain non-ensembled deterministic predictions,
        written by `write_ensembled_predictions`.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(first_valid_time_unix_sec)
    error_checking.assert_is_integer(last_valid_time_unix_sec)
    error_checking.assert_is_geq(
        last_valid_time_unix_sec, first_valid_time_unix_sec)

    error_checking.assert_is_boolean(ensembled)
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/{1:s}_predictions_{2:s}-{3:s}.p'.format(
        directory_name,
        'ensembled' if ensembled else 'gridded',
        time_conversion.unix_sec_to_string(
            first_valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES),
        time_conversion.unix_sec_to_string(
            last_valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES)
    )

    if not os.path.isfile(prediction_file_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_gridded_predictions(
        pickle_file_name, predicted_label_matrix, valid_times_unix_sec,
        narr_mask_matrix, pressure_level_mb, smoothing_radius_pixels,
        cutoff_radius_pixels, warm_front_percentile, cold_front_percentile,
        num_closing_iters):
    """Writes gridded predictions to Pickle file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param pickle_file_name: Path to output file.
    :param predicted_label_matrix: T-by-M-by-N numpy array, where the value at
        each grid cell is from the list `front_utils.VALID_INTEGER_IDS`.
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param pressure_level_mb: Pressure level (millibars).
    :param narr_mask_matrix: M-by-N numpy array of integers (0 or 1).
        If narr_mask_matrix[i, j] = 0, TFP was set to 0 for grid cell [i, j].
        Thus, any predicted front at grid cell [i, j] is only a result of binary
        closing (expanding frontal regions from nearby grid cells).
    :param smoothing_radius_pixels: See doc for `gaussian_smooth_2d_field`.
    :param cutoff_radius_pixels: Same.
    :param warm_front_percentile: See doc for `get_front_types`.
    :param cold_front_percentile: Same.
    :param num_closing_iters: See doc for `front_utils.close_frontal_image`.
    """

    ml_utils.check_narr_mask(narr_mask_matrix)

    error_checking.assert_is_integer_numpy_array(predicted_label_matrix)
    error_checking.assert_is_numpy_array(
        predicted_label_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(
        predicted_label_matrix[0, ...],
        exact_dimensions=numpy.array(narr_mask_matrix.shape))

    error_checking.assert_is_geq_numpy_array(
        predicted_label_matrix, numpy.min(front_utils.VALID_INTEGER_IDS))
    error_checking.assert_is_leq_numpy_array(
        predicted_label_matrix, numpy.max(front_utils.VALID_INTEGER_IDS))

    num_times = predicted_label_matrix.shape[0]
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=numpy.array([num_times]))

    metadata_dict = {
        VALID_TIMES_KEY: valid_times_unix_sec,
        NARR_MASK_KEY: narr_mask_matrix,
        PRESSURE_LEVEL_KEY: pressure_level_mb,
        SMOOTHING_RADIUS_KEY: smoothing_radius_pixels,
        CUTOFF_RADIUS_KEY: cutoff_radius_pixels,
        WF_PERCENTILE_KEY: warm_front_percentile,
        CF_PERCENTILE_KEY: cold_front_percentile,
        NUM_CLOSING_ITERS_KEY: num_closing_iters
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(predicted_label_matrix, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_gridded_predictions(pickle_file_name):
    """Reads gridded predictions from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: predicted_label_matrix: See doc for `write_gridded_predictions`.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['valid_times_unix_sec']: See doc for
        `write_gridded_predictions`.
    metadata_dict['narr_mask_matrix']: Same.
    metadata_dict['pressure_level_mb']: Same.
    metadata_dict['smoothing_radius_pixels']: Same.
    metadata_dict['cutoff_radius_pixels']: Same.
    metadata_dict['warm_front_percentile']: Same.
    metadata_dict['cold_front_percentile']: Same.
    metadata_dict['num_closing_iters']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    predicted_label_matrix = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return predicted_label_matrix, metadata_dict


def check_ensemble_metadata(prediction_dir_name_by_model, model_weights):
    """Checks metadata for ensemble of NFA models.

    N = number of models in ensemble

    :param prediction_dir_name_by_model: length-N list of paths to input
        directories.  prediction_dir_name_by_model[j] should contain
        deterministic predictions for [j]th model.
    :param model_weights: length-N numpy array of model weights (must sum to
        1.0).
    """

    error_checking.assert_is_geq_numpy_array(model_weights, 0.)
    error_checking.assert_is_leq_numpy_array(model_weights, 1.)
    error_checking.assert_is_geq(numpy.sum(model_weights), 1. - TOLERANCE)
    error_checking.assert_is_leq(numpy.sum(model_weights), 1. + TOLERANCE)

    num_models = len(model_weights)
    error_checking.assert_is_geq(num_models, 2)

    these_expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(prediction_dir_name_by_model),
        exact_dimensions=these_expected_dim)


def write_ensembled_predictions(
        pickle_file_name, class_probability_matrix, valid_times_unix_sec,
        narr_mask_matrix, prediction_dir_name_by_model, model_weights):
    """Writes ensembled predictions to Pickle file.

    An "ensembled prediction" is an ensemble of gridded predictions from two or
    more NFA models.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of classes

    :param pickle_file_name: Path to output file.
    :param class_probability_matrix: T-by-M-by-N-by-C numpy array of class
        probabilities.
    :param valid_times_unix_sec: length-T numpy array of time steps.
    :param narr_mask_matrix: See doc for `write_gridded_predictions`.
    :param prediction_dir_name_by_model: See doc for `check_ensemble_metadata`.
    :param model_weights: Same.
    """

    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)
    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=4)

    ml_utils.check_narr_mask(narr_mask_matrix)

    these_expected_dim = numpy.array(
        class_probability_matrix.shape[1:3], dtype=int)
    error_checking.assert_is_numpy_array(
        narr_mask_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)

    num_times = class_probability_matrix.shape[0]
    these_expected_dim = numpy.array([num_times], dtype=int)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec, exact_dimensions=these_expected_dim)

    check_ensemble_metadata(
        prediction_dir_name_by_model=prediction_dir_name_by_model,
        model_weights=model_weights)

    ensemble_dict = {
        CLASS_PROBABILITIES_KEY: class_probability_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec,
        NARR_MASK_KEY: narr_mask_matrix,
        MODEL_DIRECTORIES_KEY: prediction_dir_name_by_model,
        MODEL_WEIGHTS_KEY: model_weights
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(ensemble_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_ensembled_predictions(pickle_file_name):
    """Reads ensembled predictions from Pickle file.

    An "ensembled prediction" is an ensemble of gridded predictions from two or
    more NFA models.

    :param pickle_file_name: Path to input file.
    :return: ensemble_dict: Dictionary with the following keys.
    ensemble_dict['class_probability_matrix']: See doc for
        `write_ensembled_predictions`.
    ensemble_dict['valid_times_unix_sec']: Same.
    ensemble_dict['narr_mask_matrix']: Same.
    ensemble_dict['prediction_dir_name_by_model']: Same.
    ensemble_dict['model_weights']: Same.

    :raises: ValueError: if any required keys are not found in the dictionary.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    ensemble_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(ENSEMBLE_FILE_KEYS) - set(ensemble_dict.keys()))
    if len(missing_keys) == 0:
        return ensemble_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
