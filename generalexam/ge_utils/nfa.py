"""Implements NFA (numerical frontal analysis) methods.

--- REFERENCES ---

Renard, R., and L. Clarke, 1965: "Experiments in numerical objective frontal
    analysis". Monthly Weather Review, 93 (9), 547-556.
"""

import numpy
from gewittergefahr.gg_utils import error_checking


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


def get_wind_along_thermal_gradient(
        u_matrix_grid_relative_m_s01, v_matrix_grid_relative_m_s01,
        thermal_field_matrix_kelvins, x_spacing_metres, y_spacing_metres):
    """At each grid point, computes wind speed in direction of thermal gradient.

    M = number of rows in grid
    N = number of columns in grid

    :param u_matrix_grid_relative_m_s01: M-by-N numpy array of grid-relative
        u-wind (in the direction of increasing column number, or towards the
        right).  Units are metres per second.
    :param v_matrix_grid_relative_m_s01: M-by-N numpy array of grid-relative
        v-wind (in the direction of decreasing row number, or towards the top).
        Units are metres per second.
    :param thermal_field_matrix_kelvins: See doc for `get_thermal_front_param`.
    :param x_spacing_metres: Same.
    :return: along_grad_velocity_matrix_m_s01: M-by-N numpy array of wind speed
        (metres per second) along thermal gradient.
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
    y_grad_matrix_kelvins_m01 = -1 * y_grad_matrix_kelvins_m01
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
