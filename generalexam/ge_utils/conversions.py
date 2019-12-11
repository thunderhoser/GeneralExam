"""Methods for converting atmospheric variables."""

import numpy
# from sharppy.sharptab import thermo
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import thermo_utils

ZERO_CELSIUS_IN_KELVINS = 273.15
PASCALS_TO_MILLIBARS = 0.01


def dewpoint_to_wet_bulb_temperature(
        dewpoints_kelvins, temperatures_kelvins, total_pressures_pascals):
    """Converts one or more dewpoints to wet-bulb temperatures.

    :param dewpoints_kelvins: numpy array of dewpoints (K).
    :param temperatures_kelvins: equivalent-size numpy array of air temperatures
        (K).
    :param total_pressures_pascals: equivalent-size numpy array of total air
        pressures (K).
    :return: wet_bulb_temperatures_kelvins: equivalent-size numpy array of wet-
        bulb temperatures (K).
    """

    error_checking.assert_is_real_numpy_array(dewpoints_kelvins)
    error_checking.assert_is_real_numpy_array(temperatures_kelvins)
    error_checking.assert_is_real_numpy_array(total_pressures_pascals)

    orig_dimensions = numpy.array(dewpoints_kelvins.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        temperatures_kelvins, exact_dimensions=orig_dimensions)
    error_checking.assert_is_numpy_array(
        total_pressures_pascals, exact_dimensions=orig_dimensions)

    dewpoints_1d_celsius = -ZERO_CELSIUS_IN_KELVINS + numpy.ravel(
        dewpoints_kelvins)
    temperatures_1d_celsius = -ZERO_CELSIUS_IN_KELVINS + numpy.ravel(
        temperatures_kelvins)
    total_pressures_1d_millibars = PASCALS_TO_MILLIBARS * numpy.ravel(
        total_pressures_pascals)

    nan_flags = numpy.logical_or(
        numpy.isnan(dewpoints_1d_celsius), numpy.isnan(temperatures_1d_celsius))
    nan_flags = numpy.logical_or(
        nan_flags, numpy.isnan(total_pressures_1d_millibars))

    num_points = len(dewpoints_1d_celsius)
    wet_bulb_temperatures_1d_celsius = numpy.full(num_points, numpy.nan)

    for i in range(num_points):
        if nan_flags[i]:
            continue

        wet_bulb_temperatures_1d_celsius[i] = thermo_utils.wetbulb(
            p=total_pressures_1d_millibars[i], t=temperatures_1d_celsius[i],
            td=dewpoints_1d_celsius[i]
        )

    return ZERO_CELSIUS_IN_KELVINS + numpy.reshape(
        wet_bulb_temperatures_1d_celsius, tuple(orig_dimensions.tolist()))
