"""Implements NFA (numerical frontal analysis) methods.

--- REFERENCES ---

Renard, R., and L. Clarke, 1965: "Experiments in numerical objective frontal
    analysis". Monthly Weather Review, 93 (9), 547-556.
"""

import numpy
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

DEFAULT_FRONT_PERCENTILE = 99.


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
        < the [q]th percentile of all non-positive values in the grid, where
        q = `100 - warm_front_percentile`.
    :param cold_front_percentile: Used to locate cold fronts.  For grid cell
        [i, j] to be considered part of a cold front, its locating value must be
        > the [q]th percentile of all non-negative values in the grid, where
        q = `cold_front_percentile`.
    :return: ternary_image_matrix: See doc for
        `front_utils.dilate_ternary_narr_image`.
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

    ternary_image_matrix = numpy.full(
        locating_var_matrix_m01_s01.shape, front_utils.NO_FRONT_INTEGER_ID,
        dtype=int)
    ternary_image_matrix[
        locating_var_matrix_m01_s01 <= warm_front_threshold_m01_s01
    ] = front_utils.WARM_FRONT_INTEGER_ID
    ternary_image_matrix[
        locating_var_matrix_m01_s01 >= cold_front_threshold_m01_s01
    ] = front_utils.COLD_FRONT_INTEGER_ID

    return ternary_image_matrix
