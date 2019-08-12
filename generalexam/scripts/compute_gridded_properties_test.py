"""Unit tests for compute_gridded_properties.py."""

import copy
import unittest
import numpy
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.scripts import compute_gridded_properties as gridded_props

TOLERANCE = 1e-6

LABEL_MATRIX = numpy.array([
    [0, 0, 1, 1, 1, 1],
    [2, 2, 1, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0]
], dtype=int)

WF_LENGTH_MATRIX_METRES = numpy.array([
    [0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=float)

WF_AREA_MATRIX_M2 = copy.deepcopy(WF_LENGTH_MATRIX_METRES)

WF_LENGTH_MATRIX_METRES[WF_LENGTH_MATRIX_METRES == 0] = numpy.nan
WF_LENGTH_MATRIX_METRES[WF_LENGTH_MATRIX_METRES == 1] = (
    numpy.sqrt(10) * gridded_props.GRID_SPACING_METRES
)
WF_LENGTH_MATRIX_METRES = numpy.expand_dims(WF_LENGTH_MATRIX_METRES, axis=0)

WF_AREA_MATRIX_M2[WF_AREA_MATRIX_M2 == 0] = numpy.nan
WF_AREA_MATRIX_M2[WF_AREA_MATRIX_M2 == 1] = (
    5 * gridded_props.GRID_SPACING_METRES ** 2
)
WF_AREA_MATRIX_M2 = numpy.expand_dims(WF_AREA_MATRIX_M2, axis=0)

CF_LENGTH_MATRIX_METRES = numpy.array([
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
], dtype=float)

CF_AREA_MATRIX_M2 = copy.deepcopy(CF_LENGTH_MATRIX_METRES)

CF_LENGTH_MATRIX_METRES[CF_LENGTH_MATRIX_METRES == 0] = numpy.nan
CF_LENGTH_MATRIX_METRES[CF_LENGTH_MATRIX_METRES == 1] = (
    numpy.sqrt(5) * gridded_props.GRID_SPACING_METRES
)
CF_LENGTH_MATRIX_METRES = numpy.expand_dims(CF_LENGTH_MATRIX_METRES, axis=0)

CF_AREA_MATRIX_M2[CF_AREA_MATRIX_M2 == 0] = numpy.nan
CF_AREA_MATRIX_M2[CF_AREA_MATRIX_M2 == 1] = (
    4 * gridded_props.GRID_SPACING_METRES ** 2
)
CF_AREA_MATRIX_M2 = numpy.expand_dims(CF_AREA_MATRIX_M2, axis=0)


class ComputeGriddedPropertiesTests(unittest.TestCase):
    """Each method is a unit test for compute_gridded_properties.py."""

    def test_compute_properties_one_time(self):
        """Ensures correct output from _compute_properties_one_time."""

        this_property_dict = gridded_props._compute_properties_one_time(
            LABEL_MATRIX)

        self.assertTrue(numpy.allclose(
            this_property_dict[climo_utils.WARM_FRONT_LENGTHS_KEY],
            WF_LENGTH_MATRIX_METRES, rtol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_property_dict[climo_utils.WARM_FRONT_AREAS_KEY],
            WF_AREA_MATRIX_M2, rtol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_property_dict[climo_utils.COLD_FRONT_LENGTHS_KEY],
            CF_LENGTH_MATRIX_METRES, rtol=TOLERANCE, equal_nan=True
        ))

        self.assertTrue(numpy.allclose(
            this_property_dict[climo_utils.COLD_FRONT_AREAS_KEY],
            CF_AREA_MATRIX_M2, rtol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
