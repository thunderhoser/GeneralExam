"""Unit tests for learning_examples_io.py."""

import unittest
import numpy
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import learning_examples_io as examples_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6

# The following constants are used to test _shrink_predictor_grid.
LARGE_2D_MATRIX = numpy.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28],
    [29, 30, 31, 32, 33, 34, 35]
], dtype=float)

LARGE_3D_MATRIX = numpy.stack(
    (LARGE_2D_MATRIX, LARGE_2D_MATRIX - 10, LARGE_2D_MATRIX + 10), axis=-1
)

LARGE_PREDICTOR_MATRIX = numpy.stack(
    (LARGE_3D_MATRIX, LARGE_3D_MATRIX + 100), axis=0
)

NUM_HALF_ROWS_SMALL = 1
NUM_HALF_COLUMNS_SMALL = 2

SMALL_2D_MATRIX = numpy.array(
    [[9, 10, 11, 12, 13],
     [16, 17, 18, 19, 20],
     [23, 24, 25, 26, 27]], dtype=float)

SMALL_3D_MATRIX = numpy.stack(
    (SMALL_2D_MATRIX, SMALL_2D_MATRIX - 10, SMALL_2D_MATRIX + 10), axis=-1
)

SMALL_PREDICTOR_MATRIX = numpy.stack(
    (SMALL_3D_MATRIX, SMALL_3D_MATRIX + 100), axis=0)

# The following constants are used to test find_file, _file_name_to_times, and
# _file_name_to_batch_number.
TOP_DIRECTORY_NAME = 'poop'
FIRST_VALID_TIME_UNIX_SEC = -84157200  # 2300 UTC 2 May 1967
LAST_VALID_TIME_UNIX_SEC = -84146400  # 0200 UTC 3 May 1967
NON_SHUFFLED_FILE_NAME = 'poop/downsized_3d_examples_1967050223-1967050302.nc'

BATCH_NUMBER = 1234
SHUFFLED_FILE_NAME = (
    'poop/batches0001000-0001999/downsized_3d_examples_batch0001234.nc')

# The following constants are used to test subset_examples.
PREDICTOR_MATRIX_FIELD1 = numpy.full((16, 32), 0.)

PREDICTOR_MATRIX_EXAMPLE1 = numpy.stack((
    PREDICTOR_MATRIX_FIELD1, PREDICTOR_MATRIX_FIELD1 + 1,
    PREDICTOR_MATRIX_FIELD1 + 2, PREDICTOR_MATRIX_FIELD1 + 3
), axis=-1)

THIS_PREDICTOR_MATRIX = numpy.stack((
    PREDICTOR_MATRIX_EXAMPLE1 - 3, PREDICTOR_MATRIX_EXAMPLE1 - 2,
    PREDICTOR_MATRIX_EXAMPLE1 - 1, PREDICTOR_MATRIX_EXAMPLE1 + 1,
    PREDICTOR_MATRIX_EXAMPLE1 + 2, PREDICTOR_MATRIX_EXAMPLE1 + 3
), axis=0)

THIS_TARGET_MATRIX = numpy.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0]
], dtype=float)

THESE_TIMES_UNIX_SEC = numpy.array([0, 0, 0, 10800, 10800, 10800], dtype=int)
THESE_ROW_INDICES = numpy.array([0, 50, 100, 0, 50, 100], dtype=int)
THESE_COLUMN_INDICES = numpy.array([0, 10, 20, 0, 10, 20], dtype=int)

PREDICTOR_NAMES = [
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME,
    predictor_utils.TEMPERATURE_NAME, predictor_utils.SPECIFIC_HUMIDITY_NAME
]

PRESSURE_LEVELS_MB = numpy.array([
    predictor_utils.DUMMY_SURFACE_PRESSURE_MB,
    predictor_utils.DUMMY_SURFACE_PRESSURE_MB, 850, 850
], dtype=int)

DILATION_DISTANCE_METRES = 50000.
NARR_MASK_MATRIX = None
NORMALIZATION_TYPE_STRING = ml_utils.Z_SCORE_STRING

THIS_MEAN_MATRIX = numpy.array([
    [290, 0.01, 281, 0.007],
    [290, 0.01, 281, 0.007],
    [290, 0.01, 281, 0.007],
    [295, 0.012, 282, 0.008],
    [295, 0.012, 282, 0.008],
    [295, 0.012, 282, 0.008]
], dtype=float)

THIS_STDEV_MATRIX = numpy.array([
    [5, 0.005, 3, 0.003],
    [5, 0.005, 3, 0.003],
    [5, 0.005, 3, 0.003],
    [6, 0.007, 3.1, 0.004],
    [6, 0.007, 3.1, 0.004],
    [6, 0.007, 3.1, 0.004]
], dtype=float)

LARGE_EXAMPLE_DICT = {
    examples_io.PREDICTOR_MATRIX_KEY: THIS_PREDICTOR_MATRIX,
    examples_io.TARGET_MATRIX_KEY: THIS_TARGET_MATRIX,
    examples_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    examples_io.ROW_INDICES_KEY: THESE_ROW_INDICES,
    examples_io.COLUMN_INDICES_KEY: THESE_COLUMN_INDICES,
    examples_io.PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
    examples_io.PRESSURE_LEVELS_KEY: PRESSURE_LEVELS_MB,
    examples_io.DILATION_DISTANCE_KEY: DILATION_DISTANCE_METRES,
    examples_io.MASK_MATRIX_KEY: NARR_MASK_MATRIX,
    examples_io.NORMALIZATION_TYPE_KEY: NORMALIZATION_TYPE_STRING,
    examples_io.FIRST_NORM_PARAM_KEY: THIS_MEAN_MATRIX,
    examples_io.SECOND_NORM_PARAM_KEY: THIS_STDEV_MATRIX
}

DESIRED_EXAMPLE_INDICES = numpy.array([5, 0], dtype=int)

THIS_PREDICTOR_MATRIX = numpy.stack((
    PREDICTOR_MATRIX_EXAMPLE1 + 3, PREDICTOR_MATRIX_EXAMPLE1 - 3
), axis=0)

THIS_TARGET_MATRIX = numpy.array([
    [1, 0, 0],
    [1, 0, 0]
], dtype=float)

THESE_TIMES_UNIX_SEC = numpy.array([10800, 0], dtype=int)
THESE_ROW_INDICES = numpy.array([100, 0], dtype=int)
THESE_COLUMN_INDICES = numpy.array([20, 0], dtype=int)

THIS_MEAN_MATRIX = numpy.array([
    [295, 0.012, 282, 0.008],
    [290, 0.01, 281, 0.007]
], dtype=float)

THIS_STDEV_MATRIX = numpy.array([
    [6, 0.007, 3.1, 0.004],
    [5, 0.005, 3, 0.003]
], dtype=float)

SMALL_EXAMPLE_DICT = {
    examples_io.PREDICTOR_MATRIX_KEY: THIS_PREDICTOR_MATRIX,
    examples_io.TARGET_MATRIX_KEY: THIS_TARGET_MATRIX,
    examples_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    examples_io.ROW_INDICES_KEY: THESE_ROW_INDICES,
    examples_io.COLUMN_INDICES_KEY: THESE_COLUMN_INDICES,
    examples_io.PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
    examples_io.PRESSURE_LEVELS_KEY: PRESSURE_LEVELS_MB,
    examples_io.DILATION_DISTANCE_KEY: DILATION_DISTANCE_METRES,
    examples_io.MASK_MATRIX_KEY: NARR_MASK_MATRIX,
    examples_io.NORMALIZATION_TYPE_KEY: NORMALIZATION_TYPE_STRING,
    examples_io.FIRST_NORM_PARAM_KEY: THIS_MEAN_MATRIX,
    examples_io.SECOND_NORM_PARAM_KEY: THIS_STDEV_MATRIX
}

INTEGER_ARRAY_KEYS = [
    examples_io.VALID_TIMES_KEY, examples_io.ROW_INDICES_KEY,
    examples_io.COLUMN_INDICES_KEY, examples_io.PRESSURE_LEVELS_KEY
]

FLOAT_ARRAY_KEYS = [
    examples_io.PREDICTOR_MATRIX_KEY, examples_io.TARGET_MATRIX_KEY,
    examples_io.FIRST_NORM_PARAM_KEY, examples_io.SECOND_NORM_PARAM_KEY
]

FLOAT_KEYS = [examples_io.DILATION_DISTANCE_KEY]

# The following constants are used to test create_example_ids.
ID_STRINGS_LARGE_DICT = [
    'time0000000000_row000_column000', 'time0000000000_row050_column010',
    'time0000000000_row100_column020', 'time0000010800_row000_column000',
    'time0000010800_row050_column010', 'time0000010800_row100_column020'
]


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

        elif this_key in FLOAT_KEYS:
            if not numpy.isclose(first_example_dict[this_key],
                                 second_example_dict[this_key],
                                 atol=TOLERANCE):
                return False

        else:
            if first_example_dict[this_key] != second_example_dict[this_key]:
                return False

    return True


class LearningExamplesIoTests(unittest.TestCase):
    """Each method is a unit test for learning_examples_io.py."""

    def test_shrink_predictor_grid(self):
        """Ensures correct output from _shrink_predictor_grid."""

        this_predictor_matrix = examples_io._shrink_predictor_grid(
            predictor_matrix=LARGE_PREDICTOR_MATRIX + 0.,
            num_half_rows=NUM_HALF_ROWS_SMALL,
            num_half_columns=NUM_HALF_COLUMNS_SMALL)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, SMALL_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_file_name_to_times_shuffled(self):
        """Ensures correct output from _file_name_to_times.

        In this case the file contains temporally shuffled data.
        """

        with self.assertRaises(ValueError):
            examples_io._file_name_to_times(SHUFFLED_FILE_NAME)

    def test_file_name_to_times_non_shuffled(self):
        """Ensures correct output from _file_name_to_times.

        In this case the file does *not* contain temporally shuffled data.
        """

        this_first_time_unix_sec, this_last_time_unix_sec = (
            examples_io._file_name_to_times(NON_SHUFFLED_FILE_NAME)
        )

        self.assertTrue(this_first_time_unix_sec == FIRST_VALID_TIME_UNIX_SEC)
        self.assertTrue(this_last_time_unix_sec == LAST_VALID_TIME_UNIX_SEC)

    def test_file_name_to_batch_number_shuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the file contains temporally shuffled data.
        """

        this_batch_number = examples_io._file_name_to_batch_number(
            SHUFFLED_FILE_NAME)
        self.assertTrue(this_batch_number == BATCH_NUMBER)

    def test_file_name_to_batch_number_non_shuffled(self):
        """Ensures correct output from _file_name_to_batch_number.

        In this case the file does *not* contain temporally shuffled data.
        """

        with self.assertRaises(ValueError):
            examples_io._file_name_to_batch_number(NON_SHUFFLED_FILE_NAME)

    def test_find_file_shuffled(self):
        """Ensures correct output from find_file.

        In this case the file contains temporally shuffled data.
        """

        this_file_name = examples_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=True,
            batch_number=BATCH_NUMBER, raise_error_if_missing=False)

        self.assertTrue(this_file_name == SHUFFLED_FILE_NAME)

    def test_find_file_non_shuffled(self):
        """Ensures correct output from find_file.

        In this case the file does *not* contain temporally shuffled data.
        """

        this_file_name = examples_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME, shuffled=False,
            first_valid_time_unix_sec=FIRST_VALID_TIME_UNIX_SEC,
            last_valid_time_unix_sec=LAST_VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == NON_SHUFFLED_FILE_NAME)

    def test_subset_examples(self):
        """Ensures correct output from subset_examples."""

        this_example_dict = examples_io.subset_examples(
            example_dict=LARGE_EXAMPLE_DICT,
            desired_indices=DESIRED_EXAMPLE_INDICES)

        self.assertTrue(_compare_example_dicts(
            this_example_dict, SMALL_EXAMPLE_DICT
        ))

    def test_create_example_ids(self):
        """Ensures correct output from create_example_ids."""

        these_id_strings = examples_io.create_example_ids(
            valid_times_unix_sec=
            LARGE_EXAMPLE_DICT[examples_io.VALID_TIMES_KEY],
            row_indices=LARGE_EXAMPLE_DICT[examples_io.ROW_INDICES_KEY],
            column_indices=LARGE_EXAMPLE_DICT[examples_io.COLUMN_INDICES_KEY]
        )

        self.assertTrue(these_id_strings == ID_STRINGS_LARGE_DICT)


if __name__ == '__main__':
    unittest.main()
