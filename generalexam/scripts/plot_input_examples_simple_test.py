"""Unit tests for plot_input_examples_simple.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.scripts import plot_input_examples_simple as plot_examples

TOLERANCE = 1e-6

FIRST_TEMP_MATRIX_KELVINS = numpy.full((17, 33), 260.)
SECOND_TEMP_MATRIX_KELVINS = numpy.full((17, 33), 240.)
FIRST_HEIGHT_MATRIX_MS_S02 = numpy.full((17, 33), 1000.)
SECOND_HEIGHT_MATRIX_MS_S02 = numpy.full((17, 33), 500.)
FIRST_PRESSURE_MATRIX_PA = numpy.full((17, 33), 97500.)
SECOND_PRESSURE_MATRIX_PA = numpy.full((17, 33), 96500.)
FIRST_DEWPOINT_MATRIX_KELVINS = numpy.full((17, 33), 250.)
SECOND_DEWPOINT_MATRIX_KELVINS = numpy.full((17, 33), 230.)
FIRST_SPEC_HUMIDITY_MATRIX_KG_KG01 = numpy.full((17, 33), 1e-3)
SECOND_SPEC_HUMIDITY_MATRIX_KG_KG01 = numpy.full((17, 33), 1e-4)
FIRST_U_WIND_MATRIX_M_S01 = numpy.full((17, 33), 10.)
SECOND_U_WIND_MATRIX_M_S01 = numpy.full((17, 33), -10.)
FIRST_V_WIND_MATRIX_M_S01 = numpy.full((17, 33), 5.)
SECOND_V_WIND_MATRIX_M_S01 = numpy.full((17, 33), -5.)
FIRST_THETA_W_MATRIX_KELVINS = numpy.full((17, 33), 270.)
SECOND_THETA_W_MATRIX_KELVINS = numpy.full((17, 33), 250.)

THIS_EXAMPLE1_MATRIX = numpy.stack((
    FIRST_TEMP_MATRIX_KELVINS, FIRST_HEIGHT_MATRIX_MS_S02,
    FIRST_PRESSURE_MATRIX_PA, FIRST_DEWPOINT_MATRIX_KELVINS,
    FIRST_SPEC_HUMIDITY_MATRIX_KG_KG01, FIRST_U_WIND_MATRIX_M_S01,
    FIRST_V_WIND_MATRIX_M_S01, FIRST_THETA_W_MATRIX_KELVINS
), axis=-1)

THIS_EXAMPLE2_MATRIX = numpy.stack((
    SECOND_TEMP_MATRIX_KELVINS, SECOND_HEIGHT_MATRIX_MS_S02,
    SECOND_PRESSURE_MATRIX_PA, SECOND_DEWPOINT_MATRIX_KELVINS,
    SECOND_SPEC_HUMIDITY_MATRIX_KG_KG01, SECOND_U_WIND_MATRIX_M_S01,
    SECOND_V_WIND_MATRIX_M_S01, SECOND_THETA_W_MATRIX_KELVINS
), axis=-1)

THIS_PREDICTOR_MATRIX = numpy.stack(
    (THIS_EXAMPLE1_MATRIX, THIS_EXAMPLE2_MATRIX), axis=0
)

PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.HEIGHT_NAME,
    predictor_utils.PRESSURE_NAME, predictor_utils.DEWPOINT_NAME,
    predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME,
    predictor_utils.WET_BULB_THETA_NAME
]

PRESSURE_LEVELS_MB = numpy.full(
    THIS_PREDICTOR_MATRIX.shape[-1], 1000, dtype=int
)

ROW_INDICES = numpy.array([50, 100], dtype=int)
COLUMN_INDICES = numpy.array([100, 200], dtype=int)

EXAMPLE_DICT_ORIG_UNITS = {
    examples_io.PREDICTOR_MATRIX_KEY: THIS_PREDICTOR_MATRIX,
    examples_io.ROW_INDICES_KEY: ROW_INDICES,
    examples_io.COLUMN_INDICES_KEY: COLUMN_INDICES,
    examples_io.PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
    examples_io.PRESSURE_LEVELS_KEY: PRESSURE_LEVELS_MB
}

FIRST_TEMP_MATRIX_CELSIUS = numpy.full((17, 33), -13.15)
SECOND_TEMP_MATRIX_CELSIUS = numpy.full((17, 33), -33.15)
FIRST_HEIGHT_MATRIX_METRES = numpy.full((17, 33), 1000. / 9.80665)
SECOND_HEIGHT_MATRIX_METRES = numpy.full((17, 33), 500. / 9.80665)
FIRST_PRESSURE_MATRIX_MB = numpy.full((17, 33), 975.)
SECOND_PRESSURE_MATRIX_MB = numpy.full((17, 33), 965.)
FIRST_DEWPOINT_MATRIX_CELSIUS = numpy.full((17, 33), -23.15)
SECOND_DEWPOINT_MATRIX_CELSIUS = numpy.full((17, 33), -43.15)
FIRST_SPEC_HUMIDITY_MATRIX_G_KG01 = numpy.full((17, 33), 1.)
SECOND_SPEC_HUMIDITY_MATRIX_G_KG01 = numpy.full((17, 33), 0.1)
FIRST_THETA_W_MATRIX_CELSIUS = numpy.full((17, 33), -3.15)
SECOND_THETA_W_MATRIX_CELSIUS = numpy.full((17, 33), -23.15)

THIS_EXAMPLE1_MATRIX = numpy.stack((
    FIRST_TEMP_MATRIX_CELSIUS, FIRST_HEIGHT_MATRIX_METRES,
    FIRST_PRESSURE_MATRIX_MB, FIRST_DEWPOINT_MATRIX_CELSIUS,
    FIRST_SPEC_HUMIDITY_MATRIX_G_KG01, FIRST_U_WIND_MATRIX_M_S01,
    FIRST_V_WIND_MATRIX_M_S01, FIRST_THETA_W_MATRIX_CELSIUS
), axis=-1)

THIS_EXAMPLE2_MATRIX = numpy.stack((
    SECOND_TEMP_MATRIX_CELSIUS, SECOND_HEIGHT_MATRIX_METRES,
    SECOND_PRESSURE_MATRIX_MB, SECOND_DEWPOINT_MATRIX_CELSIUS,
    SECOND_SPEC_HUMIDITY_MATRIX_G_KG01, SECOND_U_WIND_MATRIX_M_S01,
    SECOND_V_WIND_MATRIX_M_S01, SECOND_THETA_W_MATRIX_CELSIUS
), axis=-1)

THIS_PREDICTOR_MATRIX = numpy.stack(
    (THIS_EXAMPLE1_MATRIX, THIS_EXAMPLE2_MATRIX), axis=0
)

EXAMPLE_DICT_NEW_UNITS = {
    examples_io.PREDICTOR_MATRIX_KEY: THIS_PREDICTOR_MATRIX,
    examples_io.ROW_INDICES_KEY: ROW_INDICES,
    examples_io.COLUMN_INDICES_KEY: COLUMN_INDICES,
    examples_io.PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
    examples_io.PRESSURE_LEVELS_KEY: PRESSURE_LEVELS_MB
}

NARR_LAT_MATRIX_DEG, NARR_LNG_MATRIX_DEG = (
    nwp_model_utils.get_latlng_grid_point_matrices(
        model_name=nwp_model_utils.NARR_MODEL_NAME,
        grid_name=nwp_model_utils.NAME_OF_221GRID)
)

NARR_COSINE_MATRIX, NARR_SINE_MATRIX = nwp_model_utils.get_wind_rotation_angles(
    latitudes_deg=NARR_LAT_MATRIX_DEG, longitudes_deg=NARR_LNG_MATRIX_DEG,
    model_name=nwp_model_utils.NARR_MODEL_NAME)

FIRST_U_WIND_MATRIX_M_S01, FIRST_V_WIND_MATRIX_M_S01 = (
    nwp_model_utils.rotate_winds_to_earth_relative(
        u_winds_grid_relative_m_s01=FIRST_U_WIND_MATRIX_M_S01,
        v_winds_grid_relative_m_s01=FIRST_V_WIND_MATRIX_M_S01,
        rotation_angle_cosines=NARR_COSINE_MATRIX[42:59, 84:117],
        rotation_angle_sines=NARR_SINE_MATRIX[42:59, 84:117]
    )
)

SECOND_U_WIND_MATRIX_M_S01, SECOND_V_WIND_MATRIX_M_S01 = (
    nwp_model_utils.rotate_winds_to_earth_relative(
        u_winds_grid_relative_m_s01=SECOND_U_WIND_MATRIX_M_S01,
        v_winds_grid_relative_m_s01=SECOND_V_WIND_MATRIX_M_S01,
        rotation_angle_cosines=NARR_COSINE_MATRIX[92:109, 184:217],
        rotation_angle_sines=NARR_SINE_MATRIX[92:109, 184:217]
    )
)

THIS_EXAMPLE1_MATRIX = numpy.stack((
    FIRST_TEMP_MATRIX_CELSIUS, FIRST_HEIGHT_MATRIX_METRES,
    FIRST_PRESSURE_MATRIX_MB, FIRST_DEWPOINT_MATRIX_CELSIUS,
    FIRST_SPEC_HUMIDITY_MATRIX_G_KG01, FIRST_U_WIND_MATRIX_M_S01,
    FIRST_V_WIND_MATRIX_M_S01, FIRST_THETA_W_MATRIX_CELSIUS
), axis=-1)

THIS_EXAMPLE2_MATRIX = numpy.stack((
    SECOND_TEMP_MATRIX_CELSIUS, SECOND_HEIGHT_MATRIX_METRES,
    SECOND_PRESSURE_MATRIX_MB, SECOND_DEWPOINT_MATRIX_CELSIUS,
    SECOND_SPEC_HUMIDITY_MATRIX_G_KG01, SECOND_U_WIND_MATRIX_M_S01,
    SECOND_V_WIND_MATRIX_M_S01, SECOND_THETA_W_MATRIX_CELSIUS
), axis=-1)

THIS_PREDICTOR_MATRIX = numpy.stack(
    (THIS_EXAMPLE1_MATRIX, THIS_EXAMPLE2_MATRIX), axis=0
)

EXAMPLE_DICT_AFTER_ROTATION = {
    examples_io.PREDICTOR_MATRIX_KEY: THIS_PREDICTOR_MATRIX,
    examples_io.ROW_INDICES_KEY: ROW_INDICES,
    examples_io.COLUMN_INDICES_KEY: COLUMN_INDICES,
    examples_io.PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
    examples_io.PRESSURE_LEVELS_KEY: PRESSURE_LEVELS_MB
}

INTEGER_ARRAY_KEYS = [
    examples_io.ROW_INDICES_KEY, examples_io.COLUMN_INDICES_KEY,
    examples_io.PRESSURE_LEVELS_KEY
]
FLOAT_ARRAY_KEYS = [examples_io.PREDICTOR_MATRIX_KEY]


def _compare_example_dicts(first_example_dict, second_example_dict):
    """Computes two dictionaries with examples.

    :param first_example_dict: See doc for
        `learning_examples_io.create_examples`.
    :param second_example_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_example_dict.keys())
    second_keys = list(second_example_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key in FLOAT_ARRAY_KEYS:
            if not numpy.allclose(first_example_dict[this_key],
                                  second_example_dict[this_key],
                                  atol=TOLERANCE):
                return False

        elif this_key in INTEGER_ARRAY_KEYS:
            if not numpy.array_equal(
                    first_example_dict[this_key], second_example_dict[this_key]
            ):
                return False

        else:
            if first_example_dict[this_key] != second_example_dict[this_key]:
                return False

    return True


class PlotInputExamplesSimpleTests(unittest.TestCase):
    """Each method is a unit test for plot_input_examples_simple.py."""

    def test_convert_units(self):
        """Ensures correct output from _convert_units."""

        num_examples = EXAMPLE_DICT_ORIG_UNITS[
            examples_io.PREDICTOR_MATRIX_KEY].shape[0]

        this_example_dict = copy.deepcopy(EXAMPLE_DICT_ORIG_UNITS)

        for i in range(num_examples):
            this_example_dict = plot_examples._convert_units(
                example_dict=this_example_dict, example_index=i)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_NEW_UNITS
        ))

    def test_rotate_winds(self):
        """Ensures correct output from _rotate_winds."""

        num_examples = EXAMPLE_DICT_NEW_UNITS[
            examples_io.PREDICTOR_MATRIX_KEY].shape[0]

        this_example_dict = copy.deepcopy(EXAMPLE_DICT_NEW_UNITS)

        for i in range(num_examples):
            this_example_dict = plot_examples._rotate_winds(
                example_dict=this_example_dict, example_index=i,
                narr_cosine_matrix=NARR_COSINE_MATRIX,
                narr_sine_matrix=NARR_SINE_MATRIX
            )[0]

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_AFTER_ROTATION
        ))


if __name__ == '__main__':
    unittest.main()
