"""Unit tests for conversions.py."""

import unittest
import numpy
from generalexam.ge_utils import conversions

TOLERANCE = 1e-5

DEWPOINT_MATRIX_KELVINS = numpy.array(
    [[-42.9, 6.5, numpy.nan],
     [-46.8, 26.7, -100]]
) + conversions.ZERO_CELSIUS_IN_KELVINS
TEMPERATURE_MATRIX_KELVINS = numpy.array(
    [[-35.9, 36.4, -50],
     [-46.1, 30.9, -80]]
) + conversions.ZERO_CELSIUS_IN_KELVINS

PRESSURE_MATRIX_PASCALS = numpy.array([[89930, 88710, 85000],
                                       [93450, 93450, numpy.nan]])

WET_BULB_TEMP_MATRIX_KELVINS = numpy.array(
    [[-36.265742, 17.101542, numpy.nan],
     [-46.125458, 27.675687, numpy.nan]]
) + conversions.ZERO_CELSIUS_IN_KELVINS


class ConversionsTests(unittest.TestCase):
    """Each method is a unit test for conversions.py."""

    def test_dewpoint_to_wet_bulb_temperature(self):
        """Ensures correct output from dewpoint_to_wet_bulb_temperature."""

        this_wet_bulb_temp_matrix_kelvins = (
            conversions.dewpoint_to_wet_bulb_temperature(
                dewpoints_kelvins=DEWPOINT_MATRIX_KELVINS,
                temperatures_kelvins=TEMPERATURE_MATRIX_KELVINS,
                total_pressures_pascals=PRESSURE_MATRIX_PASCALS)
        )

        self.assertTrue(numpy.allclose(
            this_wet_bulb_temp_matrix_kelvins, WET_BULB_TEMP_MATRIX_KELVINS,
            atol=TOLERANCE, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
