"""Unit tests for permutation.py."""

import unittest
import numpy
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import permutation

SURFACE_PRESSURE_MB = predictor_utils.DUMMY_SURFACE_PRESSURE_MB

PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME, predictor_utils.PRESSURE_NAME,
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.HEIGHT_NAME
]

PRESSURE_LEVELS_MB = numpy.array([
    SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB,
    SURFACE_PRESSURE_MB, SURFACE_PRESSURE_MB,
    950, 950, 900
], dtype=int)

NICE_PREDICTOR_NAMES = [
    'Surface temperature', 'Surface specific humidity',
    r'Surface $u$-wind', r'Surface $v$-wind', 'Surface pressure',
    '950-mb temperature', '950-mb specific humidity',
    '900-mb geopotential height'
]


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_get_nice_predictor_names(self):
        """Ensures correct output from get_nice_predictor_names."""

        these_nice_predictor_names = permutation.get_nice_predictor_names(
            predictor_names=PREDICTOR_NAMES,
            pressure_levels_mb=PRESSURE_LEVELS_MB)

        self.assertTrue(these_nice_predictor_names == NICE_PREDICTOR_NAMES)


if __name__ == '__main__':
    unittest.main()
