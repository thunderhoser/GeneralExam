"""Unit tests for predictor_io.py"""

import copy
import unittest
import numpy
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils

TOLERANCE = 1e-6
MIN_OROGRAPHIC_HEIGHT_M_ASL = -500.

# The following constants are used to test _subset_channels.
ALL_FIELD_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME,
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    predictor_utils.V_WIND_GRID_RELATIVE_NAME
]
ALL_PRESSURE_LEVELS_MB = numpy.array(
    [1000, 1000, 1000, 1000, 850, 850, 850, 850], dtype=int
)

VALID_TIMES_UNIX_SEC = numpy.array([0, 1, 2, 3], dtype=int)
THESE_DIMENSIONS = (len(VALID_TIMES_UNIX_SEC), 277, 349, len(ALL_FIELD_NAMES))
MAIN_DATA_MATRIX = numpy.random.uniform(
    low=MIN_OROGRAPHIC_HEIGHT_M_ASL - 2, high=MIN_OROGRAPHIC_HEIGHT_M_ASL - 1,
    size=THESE_DIMENSIONS
)

MAIN_PREDICTOR_DICT = {
    predictor_utils.FIELD_NAMES_KEY: ALL_FIELD_NAMES,
    predictor_utils.PRESSURE_LEVELS_KEY: ALL_PRESSURE_LEVELS_MB,
    predictor_utils.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    predictor_utils.DATA_MATRIX_KEY: MAIN_DATA_MATRIX,
    predictor_utils.LATITUDES_KEY: None,
    predictor_utils.LONGITUDES_KEY: None
}

FIRST_FIELD_NAMES = [
    predictor_utils.V_WIND_GRID_RELATIVE_NAME, predictor_utils.TEMPERATURE_NAME,
    predictor_utils.TEMPERATURE_NAME
]
FIRST_PRESSURE_LEVELS_MB = numpy.array([1000, 850, 1000], dtype=int)
FIRST_METADATA_ONLY_FLAG = True

FIRST_PREDICTOR_DICT = copy.deepcopy(MAIN_PREDICTOR_DICT)
FIRST_PREDICTOR_DICT[predictor_utils.FIELD_NAMES_KEY] = FIRST_FIELD_NAMES
FIRST_PREDICTOR_DICT[predictor_utils.PRESSURE_LEVELS_KEY] = (
    FIRST_PRESSURE_LEVELS_MB
)

SECOND_FIELD_NAMES = copy.deepcopy(FIRST_FIELD_NAMES)
SECOND_PRESSURE_LEVELS_MB = FIRST_PRESSURE_LEVELS_MB + 0
SECOND_METADATA_ONLY_FLAG = False

SECOND_PREDICTOR_DICT = copy.deepcopy(FIRST_PREDICTOR_DICT)
SECOND_PREDICTOR_DICT[predictor_utils.DATA_MATRIX_KEY] = (
    SECOND_PREDICTOR_DICT[predictor_utils.DATA_MATRIX_KEY][..., [3, 4, 0]]
)

DUMMY_PRESSURES_MB = numpy.array(
    [predictor_utils.DUMMY_SURFACE_PRESSURE_MB], dtype=int
)
THIRD_FIELD_NAMES = [predictor_utils.HEIGHT_NAME] + FIRST_FIELD_NAMES
THIRD_PRESSURE_LEVELS_MB = numpy.concatenate(
    (DUMMY_PRESSURES_MB, FIRST_PRESSURE_LEVELS_MB), axis=0
)
THIRD_METADATA_ONLY_FLAG = True

THIRD_PREDICTOR_DICT = copy.deepcopy(MAIN_PREDICTOR_DICT)
THIRD_PREDICTOR_DICT[predictor_utils.FIELD_NAMES_KEY] = (
    FIRST_FIELD_NAMES + [predictor_utils.HEIGHT_NAME]
)
THIRD_PREDICTOR_DICT[predictor_utils.PRESSURE_LEVELS_KEY] = numpy.concatenate(
    (FIRST_PRESSURE_LEVELS_MB, DUMMY_PRESSURES_MB), axis=0
)

FOURTH_FIELD_NAMES = copy.deepcopy(THIRD_FIELD_NAMES)
FOURTH_PRESSURE_LEVELS_MB = THIRD_PRESSURE_LEVELS_MB + 0
FOURTH_METADATA_ONLY_FLAG = False
FOURTH_PREDICTOR_DICT = copy.deepcopy(THIRD_PREDICTOR_DICT)

THESE_DIMENSIONS = MAIN_DATA_MATRIX.shape[:-1] + (1,)
NEW_DATA_MATRIX = numpy.random.uniform(
    low=MIN_OROGRAPHIC_HEIGHT_M_ASL, high=MIN_OROGRAPHIC_HEIGHT_M_ASL + 1,
    size=THESE_DIMENSIONS
)
FOURTH_PREDICTOR_DICT[predictor_utils.DATA_MATRIX_KEY] = numpy.concatenate((
    THIRD_PREDICTOR_DICT[predictor_utils.DATA_MATRIX_KEY][..., [3, 4, 0]],
    NEW_DATA_MATRIX
), axis=-1)

# The following constants are used to test find_file.
TOP_DIRECTORY_NAME = 'stuff'
VALID_TIME_UNIX_SEC = 65804414745  # 040545 UTC 5 Apr 4055
FILE_NAME = 'stuff/405504/era5_processed_4055040504.nc'


def _compare_predictor_dicts(first_dict, second_dict):
    """Compares dictionaries with predictors.

    :param first_dict: First dictionary.
    :param second_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    integer_array_keys = [
        predictor_utils.PRESSURE_LEVELS_KEY, predictor_utils.VALID_TIMES_KEY
    ]

    for this_key in first_keys:
        if this_key == predictor_utils.DATA_MATRIX_KEY:
            if not numpy.allclose(
                    first_dict[this_key], second_dict[this_key], atol=TOLERANCE
            ):
                return False

        elif this_key in integer_array_keys:
            if not numpy.array_equal(
                    first_dict[this_key], second_dict[this_key]
            ):
                return False

        else:
            if first_dict[this_key] != second_dict[this_key]:
                return False

    return True


class PredictorIoTests(unittest.TestCase):
    """Each method is a unit test for predictor_io.py."""

    def test_subset_channels_first(self):
        """Ensures correct output from _subset_channels.

        In this case, using first set of inputs.
        """

        this_predictor_dict = predictor_io._subset_channels(
            predictor_dict=copy.deepcopy(MAIN_PREDICTOR_DICT),
            metadata_only=FIRST_METADATA_ONLY_FLAG,
            field_names=FIRST_FIELD_NAMES,
            pressure_levels_mb=FIRST_PRESSURE_LEVELS_MB
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, FIRST_PREDICTOR_DICT
        ))

    def test_subset_channels_second(self):
        """Ensures correct output from _subset_channels.

        In this case, using second set of inputs.
        """

        this_predictor_dict = predictor_io._subset_channels(
            predictor_dict=copy.deepcopy(MAIN_PREDICTOR_DICT),
            metadata_only=SECOND_METADATA_ONLY_FLAG,
            field_names=SECOND_FIELD_NAMES,
            pressure_levels_mb=SECOND_PRESSURE_LEVELS_MB
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, SECOND_PREDICTOR_DICT
        ))

    def test_subset_channels_third(self):
        """Ensures correct output from _subset_channels.

        In this case, using third set of inputs.
        """

        this_predictor_dict = predictor_io._subset_channels(
            predictor_dict=copy.deepcopy(MAIN_PREDICTOR_DICT),
            metadata_only=THIRD_METADATA_ONLY_FLAG,
            field_names=THIRD_FIELD_NAMES,
            pressure_levels_mb=THIRD_PRESSURE_LEVELS_MB
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, THIRD_PREDICTOR_DICT
        ))

    def test_subset_channels_fourth(self):
        """Ensures correct output from _subset_channels.

        In this case, using fourth set of inputs.
        """

        actual_predictor_dict = predictor_io._subset_channels(
            predictor_dict=copy.deepcopy(MAIN_PREDICTOR_DICT),
            metadata_only=FOURTH_METADATA_ONLY_FLAG,
            field_names=FOURTH_FIELD_NAMES,
            pressure_levels_mb=FOURTH_PRESSURE_LEVELS_MB
        )

        actual_predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.minimum(
            actual_predictor_dict[predictor_utils.DATA_MATRIX_KEY],
            MIN_OROGRAPHIC_HEIGHT_M_ASL
        )

        expected_predictor_dict = copy.deepcopy(FOURTH_PREDICTOR_DICT)
        expected_predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.minimum(
            expected_predictor_dict[predictor_utils.DATA_MATRIX_KEY],
            MIN_OROGRAPHIC_HEIGHT_M_ASL
        )

        self.assertTrue(_compare_predictor_dicts(
            actual_predictor_dict, expected_predictor_dict
        ))

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = predictor_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == FILE_NAME)


if __name__ == '__main__':
    unittest.main()
