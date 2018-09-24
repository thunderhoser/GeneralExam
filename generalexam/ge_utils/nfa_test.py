"""Unit tests for nfa.py."""

import unittest
import numpy
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import nfa

TOLERANCE = 1e-6

# The following constants are used to test _get_2d_gradient.
TOY_FIELD_MATRIX = numpy.array(
    [[220, 222, 225, 226, 224.5, 224],
     [223, 225, 227, 228, 229, 230],
     [223, 226, 229, 230, 230.5, 228],
     [228, 232, 236, 235, 233, 233]]
)

TOY_X_SPACING_METRES = 10.
TOY_Y_SPACING_METRES = 5.

TOY_X_GRADIENT_MATRIX_M01 = numpy.array(
    [[2, 2.5, 2, -0.25, -1, -0.5],
     [2, 2, 1.5, 1, 1, 1],
     [3, 3, 2, 0.75, -1, -2.5],
     [4, 4, 1.5, -1.5, -1, 0]]
) / TOY_X_SPACING_METRES

TOY_Y_GRADIENT_MATRIX_M01 = numpy.array(
    [[3, 3, 2, 2, 4.5, 6],
     [1.5, 2, 2, 2, 3, 2],
     [2.5, 3.5, 4.5, 3.5, 2, 1.5],
     [5, 6, 7.0, 5, 2.5, 5]]
) / TOY_Y_SPACING_METRES

# The following constants are used to test get_thermal_front_param.
THERMAL_X_SPACING_METRES = 1.
THERMAL_Y_SPACING_METRES = 1.

THERMAL_MATRIX_KELVINS_NO_Y_GRAD = numpy.array(
    [[210, 213, 216, 219, 222, 225],
     [210, 213, 216, 219, 222, 225],
     [210, 213, 216, 219, 222, 225],
     [210, 213, 216, 219, 222, 225]], dtype=float)

TFP_MATRIX_KELVINS_M02_NO_Y_GRAD = numpy.full(
    THERMAL_MATRIX_KELVINS_NO_Y_GRAD.shape, 0.)

THERMAL_MATRIX_KELVINS_NO_X_GRAD = numpy.array(
    [[210, 210, 210, 210, 210, 210],
     [212, 212, 212, 212, 212, 212],
     [214, 214, 214, 214, 214, 214],
     [216, 216, 216, 216, 216, 216]], dtype=float)

TFP_MATRIX_KELVINS_M02_NO_X_GRAD = numpy.full(
    THERMAL_MATRIX_KELVINS_NO_X_GRAD.shape, 0.)

THERMAL_MATRIX_KELVINS = numpy.array(
    [[210, 213, 216, 219, 222, 225],
     [214, 217, 220, 223, 226, 229],
     [216, 219, 222, 225, 228, 231],
     [218, 221, 224, 227, 230, 233]], dtype=float)

THIS_X_GRAD_MATRIX_KELVINS_M01 = numpy.array(
    [[3, 3, 3, 3, 3, 3],
     [3, 3, 3, 3, 3, 3],
     [3, 3, 3, 3, 3, 3],
     [3, 3, 3, 3, 3, 3]], dtype=float)

THIS_Y_GRAD_MATRIX_KELVINS_M01 = numpy.array(
    [[4, 4, 4, 4, 4, 4],
     [3, 3, 3, 3, 3, 3],
     [2, 2, 2, 2, 2, 2],
     [2, 2, 2, 2, 2, 2]], dtype=float)

ROOT18 = numpy.sqrt(18.)
ROOT13 = numpy.sqrt(13.)

THIS_GRAD_MAGNITUDE_MATRIX_KELVINS_M01 = numpy.array(
    [[5, 5, 5, 5, 5, 5],
     [ROOT18, ROOT18, ROOT18, ROOT18, ROOT18, ROOT18],
     [ROOT13, ROOT13, ROOT13, ROOT13, ROOT13, ROOT13],
     [ROOT13, ROOT13, ROOT13, ROOT13, ROOT13, ROOT13]]
)

THIS_GRAD1 = ROOT18 - 5
THIS_GRAD2 = (ROOT13 - 5) / 2
THIS_GRAD3 = (ROOT13 - ROOT18) / 2

THIS_Y_GRAD_GRAD_MATRIX_KELVINS_M02 = numpy.array(
    [[THIS_GRAD1, THIS_GRAD1, THIS_GRAD1, THIS_GRAD1, THIS_GRAD1, THIS_GRAD1],
     [THIS_GRAD2, THIS_GRAD2, THIS_GRAD2, THIS_GRAD2, THIS_GRAD2, THIS_GRAD2],
     [THIS_GRAD3, THIS_GRAD3, THIS_GRAD3, THIS_GRAD3, THIS_GRAD3, THIS_GRAD3],
     [0, 0, 0, 0, 0, 0]], dtype=float)

TFP_MATRIX_KELVINS_M02 = (
    -1 * THIS_Y_GRAD_GRAD_MATRIX_KELVINS_M02 *
    THIS_Y_GRAD_MATRIX_KELVINS_M01 / THIS_GRAD_MAGNITUDE_MATRIX_KELVINS_M01)

# The following constants are used to test project_wind_to_thermal_gradient.
U_MATRIX_GRID_RELATIVE_M_S01 = numpy.array(
    [[0, 5, 10, 15, 20, 20],
     [0, 5, 10, 15, 20, 20],
     [0, 5, 10, 15, 20, 20],
     [0, 5, 10, 15, 20, 20]], dtype=float)

V_MATRIX_GRID_RELATIVE_M_S01 = numpy.array(
    [[20, 20, 20, 20, 20, 20],
     [10, 10, 10, 10, 10, 10],
     [5, 5, 5, 5, 5, 5],
     [5, 5, 5, 3, 2, 0]], dtype=float)

THIS_MATRIX1 = (
    U_MATRIX_GRID_RELATIVE_M_S01 *
    THIS_X_GRAD_MATRIX_KELVINS_M01 / THIS_GRAD_MAGNITUDE_MATRIX_KELVINS_M01)

THIS_MATRIX2 = (
    V_MATRIX_GRID_RELATIVE_M_S01 *
    THIS_Y_GRAD_MATRIX_KELVINS_M01 / THIS_GRAD_MAGNITUDE_MATRIX_KELVINS_M01)

ALONG_GRAD_VELOCITY_MATRIX_M_S01 = THIS_MATRIX1 + THIS_MATRIX2

# The following constants are used to test get_front_types.
LOCATING_VAR_MATRIX_M01_S01 = numpy.array(
    [[-8, -8, 1, -3, -7, -6],
     [-5, 5, -1, 2, 8, 14],
     [7, 17, 15, 7, 5, 8],
     [8, 8, 23, 23, 18, 21]], dtype=float)

WARM_FRONT_PERCENTILE = 80.
COLD_FRONT_PERCENTILE = 90.

NO_FRONT_ID = front_utils.NO_FRONT_INTEGER_ID + 0
WARM_FRONT_ID = front_utils.WARM_FRONT_INTEGER_ID + 0
COLD_FRONT_ID = front_utils.COLD_FRONT_INTEGER_ID + 0

TERNARY_IMAGE_MATRIX = numpy.array([
    [WARM_FRONT_ID, WARM_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID,
     NO_FRONT_ID],
    [NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID,
     NO_FRONT_ID],
    [NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID, NO_FRONT_ID,
     NO_FRONT_ID],
    [NO_FRONT_ID, NO_FRONT_ID, COLD_FRONT_ID, COLD_FRONT_ID, NO_FRONT_ID,
     NO_FRONT_ID]
])


class NfaTests(unittest.TestCase):
    """Each method is a unit test for nfa.py."""

    def test_get_2d_gradient(self):
        """Ensures correct output from _get_2d_gradient."""

        (this_x_gradient_matrix_m01, this_y_gradient_matrix_m01
        ) = nfa._get_2d_gradient(
            field_matrix=TOY_FIELD_MATRIX,
            x_spacing_metres=TOY_X_SPACING_METRES,
            y_spacing_metres=TOY_Y_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_x_gradient_matrix_m01, TOY_X_GRADIENT_MATRIX_M01,
            atol=TOLERANCE))
        self.assertTrue(numpy.allclose(
            this_y_gradient_matrix_m01, TOY_Y_GRADIENT_MATRIX_M01,
            atol=TOLERANCE))

    def test_get_thermal_front_param_no_y_grad(self):
        """Ensures correct output from get_thermal_front_param.

        In this case the thermal field has zero gradient in the y-direction.
        """

        this_tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
            thermal_field_matrix_kelvins=THERMAL_MATRIX_KELVINS_NO_Y_GRAD,
            x_spacing_metres=THERMAL_X_SPACING_METRES,
            y_spacing_metres=THERMAL_Y_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_tfp_matrix_kelvins_m02, TFP_MATRIX_KELVINS_M02_NO_Y_GRAD,
            atol=TOLERANCE))

    def test_get_thermal_front_param_no_x_grad(self):
        """Ensures correct output from get_thermal_front_param.

        In this case the thermal field has zero gradient in the x-direction.
        """

        this_tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
            thermal_field_matrix_kelvins=THERMAL_MATRIX_KELVINS_NO_X_GRAD,
            x_spacing_metres=THERMAL_X_SPACING_METRES,
            y_spacing_metres=THERMAL_Y_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_tfp_matrix_kelvins_m02, TFP_MATRIX_KELVINS_M02_NO_X_GRAD,
            atol=TOLERANCE))

    def test_get_thermal_front_param_both_grads(self):
        """Ensures correct output from get_thermal_front_param.

        In this case the thermal field has non-zero gradients in both the x- and
        y-direction.
        """

        this_tfp_matrix_kelvins_m02 = nfa.get_thermal_front_param(
            thermal_field_matrix_kelvins=THERMAL_MATRIX_KELVINS,
            x_spacing_metres=THERMAL_X_SPACING_METRES,
            y_spacing_metres=THERMAL_Y_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_tfp_matrix_kelvins_m02, TFP_MATRIX_KELVINS_M02,
            atol=TOLERANCE))

    def test_project_wind_to_thermal_gradient(self):
        """Ensures correct output from project_wind_to_thermal_gradient."""

        this_velocity_matrix_m_s01 = nfa.project_wind_to_thermal_gradient(
            u_matrix_grid_relative_m_s01=U_MATRIX_GRID_RELATIVE_M_S01,
            v_matrix_grid_relative_m_s01=V_MATRIX_GRID_RELATIVE_M_S01,
            thermal_field_matrix_kelvins=THERMAL_MATRIX_KELVINS,
            x_spacing_metres=THERMAL_X_SPACING_METRES,
            y_spacing_metres=THERMAL_Y_SPACING_METRES)

        self.assertTrue(numpy.allclose(
            this_velocity_matrix_m_s01, ALONG_GRAD_VELOCITY_MATRIX_M_S01,
            atol=TOLERANCE))

    def test_get_front_types(self):
        """Ensures correct output from get_front_types."""

        this_ternary_image_matrix = nfa.get_front_types(
            locating_var_matrix_m01_s01=LOCATING_VAR_MATRIX_M01_S01,
            warm_front_percentile=WARM_FRONT_PERCENTILE,
            cold_front_percentile=COLD_FRONT_PERCENTILE)

        self.assertTrue(numpy.array_equal(
            this_ternary_image_matrix, TERNARY_IMAGE_MATRIX))


if __name__ == '__main__':
    unittest.main()
