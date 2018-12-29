"""IO methods for training and on-the-fly validation.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples.  Examples may be either downsized or full-size.  See
    below for a definition of each.
K = number of classes (possible target values).  See below for the definition of
    "target value".
M = number of spatial rows per example
N = number of spatial columns per example
T = number of predictor times per example (number of images per sequence)
C = number of channels (predictor variables) per example

--- DEFINITIONS ---

A "downsized" example covers only a subset of the NARR grid, while a full-size
example covers the entire NARR grid.

The dimensions of a 3-D example are M x N x C (only one predictor time).

The dimensions of a 4-D example are M x N x T x C.

NF = no front
WF = warm front
CF = cold front

Target variable = label at one pixel.  For a downsized example, there is only
one target variable (the label at the center pixel).  For a full-size example,
there are M*N target variables (the label at each pixel).
"""

import copy
import glob
import os.path
from random import shuffle
import numpy
import keras
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import processed_narr_io
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6
LARGE_INTEGER = int(1e10)

TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9][0-2][0-9]'
BATCH_NUMBER_REGEX = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
NUM_BATCHES_PER_DIRECTORY = 1000

HOURS_TO_SECONDS = 3600
NARR_TIME_INTERVAL_SECONDS = HOURS_TO_SECONDS * nwp_model_utils.get_time_steps(
    nwp_model_utils.NARR_MODEL_NAME)[1]

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
TARGET_TIMES_KEY = 'target_times_unix_sec'
ROW_INDICES_KEY = 'row_indices'
COLUMN_INDICES_KEY = 'column_indices'
PREDICTOR_NAMES_KEY = 'narr_predictor_names'
NARR_MASK_KEY = 'narr_mask_matrix'

PRESSURE_LEVEL_KEY = 'pressure_level_mb'
DILATION_DISTANCE_KEY = 'dilation_distance_metres'
NARR_ROW_DIMENSION_KEY = 'narr_row'
NARR_COLUMN_DIMENSION_KEY = 'narr_column'
EXAMPLE_DIMENSION_KEY = 'example'
EXAMPLE_ROW_DIMENSION_KEY = 'example_row'
EXAMPLE_COLUMN_DIMENSION_KEY = 'example_column'
PREDICTOR_DIMENSION_KEY = 'predictor_variable'
CHARACTER_DIMENSION_KEY = 'predictor_variable_char'
CLASS_DIMENSION_KEY = 'class'

FIRST_NORM_PARAM_KEY = 'first_normalization_param_matrix'
SECOND_NORM_PARAM_KEY = 'second_normalization_param_matrix'

MAIN_KEYS = [
    PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY, TARGET_TIMES_KEY, ROW_INDICES_KEY,
    COLUMN_INDICES_KEY, FIRST_NORM_PARAM_KEY, SECOND_NORM_PARAM_KEY
]
OPTIONAL_KEYS = [FIRST_NORM_PARAM_KEY, SECOND_NORM_PARAM_KEY]


def _file_name_to_target_times(downsized_3d_file_name):
    """Parses file name for target times.

    :param downsized_3d_file_name: See doc for `find_downsized_3d_example_file`.
    :return: first_target_time_unix_sec: First target time in file.
    :return: last_target_time_unix_sec: Last target time in file.
    :raises: ValueError: if target times cannot be parsed from file name.
    """

    pathless_file_name = os.path.split(downsized_3d_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    target_time_strings = extensionless_file_name.split(
        'downsized_3d_examples_')[-1].split('-')

    first_target_time_unix_sec = time_conversion.string_to_unix_sec(
        target_time_strings[0], TIME_FORMAT)
    last_target_time_unix_sec = time_conversion.string_to_unix_sec(
        target_time_strings[-1], TIME_FORMAT)

    return first_target_time_unix_sec, last_target_time_unix_sec


def _file_name_to_batch_number(downsized_3d_file_name):
    """Parses file name for batch number.

    :param downsized_3d_file_name: See doc for `find_downsized_3d_example_file`.
    :return: batch_number: Integer.
    :raises: ValueError: if batch number cannot be parsed from file name.
    """

    pathless_file_name = os.path.split(downsized_3d_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    return int(extensionless_file_name.split('downsized_3d_examples_batch')[-1])


def _decrease_example_size(predictor_matrix, num_half_rows, num_half_columns):
    """Decreases the grid size for each example.

    M = original number of rows per example
    N = original number of columns per example
    m = new number of rows per example
    n = new number of columns per example

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor images.
    :param num_half_rows: Determines number of rows returned for each example.
        Examples will be cropped so that the center of the original image is the
        center of the new image.  If `num_half_rows`, examples will not be
        cropped.
    :param num_half_columns: Same but for columns.
    :return: predictor_matrix: E-by-m-by-n-by-C numpy array of predictor images.
    """

    if num_half_rows is not None:
        error_checking.assert_is_integer(num_half_rows)
        error_checking.assert_is_greater(num_half_rows, 0)

        center_row_index = int(
            numpy.floor(float(predictor_matrix.shape[1]) / 2)
        )
        first_row_index = center_row_index - num_half_rows
        last_row_index = center_row_index + num_half_rows
        predictor_matrix = predictor_matrix[
            :, first_row_index:(last_row_index + 1), ...
        ]

    if num_half_columns is not None:
        error_checking.assert_is_integer(num_half_columns)
        error_checking.assert_is_greater(num_half_columns, 0)

        center_column_index = int(
            numpy.floor(float(predictor_matrix.shape[2]) / 2)
        )
        first_column_index = center_column_index - num_half_columns
        last_column_index = center_column_index + num_half_columns
        predictor_matrix = predictor_matrix[
            :, :, first_column_index:(last_column_index + 1), ...
        ]

    return predictor_matrix


def find_input_files_for_3d_examples(
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb):
    """Finds input files for 3-D examples.

    These files do not *contain* 3-D examples, but they may be used to *create*
    3-D examples on the fly.

    Q = number of target times

    :param first_target_time_unix_sec: First target time.  This method will find
        files for all target times from `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param top_narr_directory_name: Name of top-level directory with NARR data.
        Files therein will be found by
        `processed_narr_io.find_file_for_one_time` and read by
        `processed_narr_io.read_fields_from_file`.
    :param top_frontal_grid_dir_name: Name of top-level directory with target
        values (grids of front labels).  Files therein will be found by
        `fronts_io.find_file_for_one_time` and read by
        `fronts_io.read_narr_grids_from_file`.
    :param narr_predictor_names: length-C list with names of predictor
        variables.  Each must be accepted by
        `processed_narr_io.check_field_name`.
    :param pressure_level_mb: Pressure level (millibars) for predictors.
    :return: narr_file_name_matrix: Q-by-C numpy array of paths to predictor
        files.
    :return: frontal_grid_file_names: length-Q list of paths to target files.
    """

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(target_times_unix_sec)

    num_target_times = len(target_times_unix_sec)
    num_predictors = len(narr_predictor_names)

    frontal_grid_file_names = [''] * num_target_times
    narr_file_name_matrix = numpy.full(
        (num_target_times, num_predictors), '', dtype=numpy.object)

    for i in range(num_target_times):
        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(frontal_grid_file_names[i]):
            frontal_grid_file_names[i] = ''
            continue

        for j in range(num_predictors):
            narr_file_name_matrix[i, j] = (
                processed_narr_io.find_file_for_one_time(
                    top_directory_name=top_narr_directory_name,
                    field_name=narr_predictor_names[j],
                    pressure_level_mb=pressure_level_mb,
                    valid_time_unix_sec=target_times_unix_sec[i],
                    raise_error_if_missing=True)
            )

    keep_time_flags = numpy.array(
        [f != '' for f in frontal_grid_file_names], dtype=bool)
    keep_time_indices = numpy.where(keep_time_flags)[0]

    return (narr_file_name_matrix[keep_time_indices, ...],
            [frontal_grid_file_names[i] for i in keep_time_indices])


def find_input_files_for_4d_examples(
        first_target_time_unix_sec, last_target_time_unix_sec,
        num_lead_time_steps, predictor_time_step_offsets,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb):
    """Finds input files for 4-D examples.

    These files do not *contain* 4-D examples, but they may be used to *create*
    4-D examples on the fly.

    Q = number of target times

    :param first_target_time_unix_sec: See doc for
        `find_input_files_for_3d_examples`.
    :param last_target_time_unix_sec: Same.
    :param num_lead_time_steps: Number of time steps between target time and
        latest possible predictor time.
    :param predictor_time_step_offsets: length-T numpy array of offsets between
        predictor time and latest possible predictor time (target time minus
        lead time).
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :return: narr_file_name_matrix: Q-by-T-by-C numpy array of paths to
        predictor files.
    :return: frontal_grid_file_names: length-Q list of paths to target files.
    """

    error_checking.assert_is_integer_numpy_array(predictor_time_step_offsets)
    error_checking.assert_is_numpy_array(
        predictor_time_step_offsets, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(predictor_time_step_offsets, 0)

    predictor_time_step_offsets = numpy.unique(
        predictor_time_step_offsets)[::-1]

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)
    numpy.random.shuffle(target_times_unix_sec)

    num_target_times = len(target_times_unix_sec)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)

    frontal_grid_file_names = [''] * num_target_times
    narr_file_name_matrix = numpy.full(
        (num_target_times, num_predictor_times_per_example, num_predictors), '',
        dtype=numpy.object)

    for i in range(num_target_times):
        frontal_grid_file_names[i] = fronts_io.find_file_for_one_time(
            top_directory_name=top_frontal_grid_dir_name,
            file_type=fronts_io.GRIDDED_FILE_TYPE,
            valid_time_unix_sec=target_times_unix_sec[i],
            raise_error_if_missing=False)

        if not os.path.isfile(frontal_grid_file_names[i]):
            frontal_grid_file_names[i] = ''
            continue

        this_last_time_unix_sec = target_times_unix_sec[i] - (
            num_lead_time_steps * NARR_TIME_INTERVAL_SECONDS)
        these_narr_times_unix_sec = this_last_time_unix_sec - (
            predictor_time_step_offsets * NARR_TIME_INTERVAL_SECONDS)

        for j in range(num_predictor_times_per_example):
            for k in range(num_predictors):
                narr_file_name_matrix[i, j, k] = (
                    processed_narr_io.find_file_for_one_time(
                        top_directory_name=top_narr_directory_name,
                        field_name=narr_predictor_names[k],
                        pressure_level_mb=pressure_level_mb,
                        valid_time_unix_sec=these_narr_times_unix_sec[j],
                        raise_error_if_missing=True)
                )

    keep_time_flags = numpy.array(
        [f != '' for f in frontal_grid_file_names], dtype=bool)
    keep_time_indices = numpy.where(keep_time_flags)[0]

    return (narr_file_name_matrix[keep_time_indices, ...],
            [frontal_grid_file_names[i] for i in keep_time_indices])


def downsized_3d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        class_fractions, num_rows_in_half_grid, num_columns_in_half_grid,
        narr_mask_matrix=None):
    """Generates downsized 3-D examples from raw files.

    :param num_examples_per_batch: Number of examples per batch.
    :param num_examples_per_target_time: Number of examples (target pixels) per
        target time.
    :param first_target_time_unix_sec: First target time.  Examples will be
        randomly drawn from the period `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Dilation distance.  Will be used to dilate
        WF and CF labels, which effectively creates a distance buffer around
        each front, thus accounting for spatial uncertainty in front placement.
    :param class_fractions: List of downsampling fractions.  Must have length 3,
        where the elements are (NF, WF, CF).  The sum of all fractions must be
        1.0.
    :param num_rows_in_half_grid: Number of rows in half-grid for each example.
        Actual number of rows will be 2 * `num_rows_in_half_grid` + 1.
    :param num_columns_in_half_grid: Same but for columns.
    :param narr_mask_matrix: See doc for
        `machine_learning_utils.check_narr_mask`.  If narr_mask_matrix[i, j]
        = 0, cell [i, j] in the full NARR grid will never be used as the center
        of a downsized example.  If you do not want masking, leave this alone.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: E-by-K numpy array of target values.  All values are
        0 or 1, but the array type is "float64".  Columns are mutually exclusive
        and collectively exhaustive, so the sum across each row is 1.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_target_time)
    error_checking.assert_is_geq(num_examples_per_target_time, 2)

    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    time_index = 0
    num_times_in_memory = 0
    num_times_needed_in_memory = int(
        numpy.ceil(float(num_examples_per_batch) / num_examples_per_target_time)
    )

    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_times_in_memory < num_times_needed_in_memory:
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[time_index, j])

                this_field_predictor_matrix = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[time_index, j])
                )[0]
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix)
                )

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[time_index])

            time_index += 1
            if time_index >= num_times:
                time_index = 0

            this_full_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_full_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_full_predictor_matrix)

            this_full_target_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_full_target_matrix = ml_utils.binarize_front_images(
                    this_full_target_matrix)

            if num_classes == 2:
                this_full_target_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_full_target_matrix,
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_full_target_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_full_target_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = copy.deepcopy(
                    this_full_predictor_matrix)
                full_target_matrix = copy.deepcopy(this_full_target_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_full_predictor_matrix), axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_full_target_matrix), axis=0)

            num_times_in_memory = full_target_matrix.shape[0]

        print 'Creating downsized 3-D examples...'
        sampled_target_point_dict = ml_utils.sample_target_points(
            target_matrix=full_target_matrix, class_fractions=class_fractions,
            num_points_to_sample=num_examples_per_batch,
            mask_matrix=narr_mask_matrix)

        (downsized_predictor_matrix, target_values
        ) = ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=sampled_target_point_dict,
            verbose=False)[:2]

        numpy.random.shuffle(batch_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = target_values[batch_indices]

        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        actual_class_fractions = numpy.sum(target_matrix, axis=0)
        print 'Fraction of examples in each class: {0:s}'.format(
            str(actual_class_fractions))

        full_predictor_matrix = None
        full_target_matrix = None
        num_times_in_memory = 0

        yield (downsized_predictor_matrix, target_matrix)


def quick_downsized_3d_example_gen(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, top_input_dir_name, narr_predictor_names,
        num_classes, num_rows_in_half_grid, num_columns_in_half_grid):
    """Generates downsized 3-D examples from processed files.

    These "processed files" are created by `write_downsized_3d_examples`.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param top_input_dir_name: Name of top-level directory for files with
        downsized 3-D examples.  Files therein will be found by
        `find_downsized_3d_example_file` (with `shuffled == True`) and read by
        `read_downsized_3d_examples`.
    :param narr_predictor_names: See doc for `downsized_3d_example_generator`.
    :param num_classes: Number of target classes (2 or 3).
    :param num_rows_in_half_grid: See doc for `downsized_3d_example_generator`.
    :param num_columns_in_half_grid: Same.
    :return: predictor_matrix: See doc for `downsized_3d_example_generator`.
    :return: target_matrix: Same.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    example_file_names = find_downsized_3d_example_files(
        top_directory_name=top_input_dir_name, shuffled=True,
        first_batch_number=0, last_batch_number=LARGE_INTEGER)
    shuffle(example_file_names)

    num_files = len(example_file_names)
    file_index = 0
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print 'Reading data from: "{0:s}"...'.format(
                example_file_names[file_index])
            this_example_dict = read_downsized_3d_examples(
                netcdf_file_name=example_file_names[file_index],
                predictor_names_to_keep=narr_predictor_names,
                num_half_rows_to_keep=num_rows_in_half_grid,
                num_half_columns_to_keep=num_columns_in_half_grid,
                first_time_to_keep_unix_sec=first_target_time_unix_sec,
                last_time_to_keep_unix_sec=last_target_time_unix_sec)

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            this_num_examples = len(this_example_dict[TARGET_TIMES_KEY])
            if this_num_examples == 0:
                continue

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = this_example_dict[
                    PREDICTOR_MATRIX_KEY] + 0.
                full_target_matrix = this_example_dict[TARGET_MATRIX_KEY] + 0
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_example_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_example_dict[TARGET_MATRIX_KEY]),
                    axis=0)

            num_examples_in_memory = full_target_matrix.shape[0]

        predictor_matrix = full_predictor_matrix[batch_indices, ...].astype(
            'float32')
        target_matrix = full_target_matrix[batch_indices, ...].astype('float64')

        if num_classes == 2:
            target_values = numpy.argmax(target_matrix, axis=1)
            target_matrix = keras.utils.to_categorical(
                target_values, num_classes)

        actual_class_fractions = numpy.sum(target_matrix, axis=0)
        print 'Number of examples in each class: {0:s}'.format(
            str(actual_class_fractions))

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None

        yield (predictor_matrix, target_matrix)


def downsized_4d_example_generator(
        num_examples_per_batch, num_examples_per_target_time,
        first_target_time_unix_sec, last_target_time_unix_sec,
        num_lead_time_steps, predictor_time_step_offsets,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_distance_metres,
        class_fractions, num_rows_in_half_grid, num_columns_in_half_grid,
        narr_mask_matrix=None):
    """Generates downsized 4-D examples from raw files.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param num_examples_per_target_time: Same.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param num_lead_time_steps: See doc for `find_input_files_for_4d_examples`.
    :param predictor_time_step_offsets: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param class_fractions: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :param narr_mask_matrix: Same.
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        values.
    :return: target_matrix: See doc for `downsized_3d_example_generator`.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 10)
    error_checking.assert_is_integer(num_examples_per_target_time)
    error_checking.assert_is_geq(num_examples_per_target_time, 2)

    error_checking.assert_is_numpy_array(class_fractions, num_dimensions=1)
    num_classes = len(class_fractions)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_4d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        predictor_time_step_offsets=predictor_time_step_offsets,
        num_lead_time_steps=num_lead_time_steps,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_target_times = len(frontal_grid_file_names)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_times_in_memory = 0
    num_times_needed_in_memory = int(
        numpy.ceil(float(num_examples_per_batch) / num_examples_per_target_time)
    )

    full_predictor_matrix = None
    full_target_matrix = None

    while True:
        while num_times_in_memory < num_times_needed_in_memory:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            for i in range(num_predictor_times_per_example):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])

                    this_field_predictor_matrix = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j])
                    )[0]
                    this_field_predictor_matrix = (
                        ml_utils.fill_nans_in_predictor_images(
                            this_field_predictor_matrix)
                    )

                    tuple_of_3d_predictor_matrices += (
                        this_field_predictor_matrix,)

                tuple_of_4d_predictor_matrices += (
                    ml_utils.stack_predictor_variables(
                        tuple_of_3d_predictor_matrices),
                )

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            this_full_predictor_matrix = ml_utils.stack_time_steps(
                tuple_of_4d_predictor_matrices)
            this_full_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_full_predictor_matrix)

            this_full_target_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_full_predictor_matrix.shape[1],
                num_columns_per_image=this_full_predictor_matrix.shape[2])

            if num_classes == 2:
                this_full_target_matrix = ml_utils.binarize_front_images(
                    this_full_target_matrix)

            if num_classes == 2:
                this_full_target_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_full_target_matrix,
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_full_target_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_full_target_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = copy.deepcopy(
                    this_full_predictor_matrix)
                full_target_matrix = copy.deepcopy(this_full_target_matrix)
            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix, this_full_predictor_matrix), axis=0)
                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_full_target_matrix), axis=0)

            num_times_in_memory = full_target_matrix.shape[0]

        print 'Creating downsized 4-D examples...'
        sampled_target_point_dict = ml_utils.sample_target_points(
            target_matrix=full_target_matrix, class_fractions=class_fractions,
            num_points_to_sample=num_examples_per_batch,
            mask_matrix=narr_mask_matrix)

        (downsized_predictor_matrix, target_values
        ) = ml_utils.downsize_grids_around_selected_points(
            predictor_matrix=full_predictor_matrix,
            target_matrix=full_target_matrix,
            num_rows_in_half_window=num_rows_in_half_grid,
            num_columns_in_half_window=num_columns_in_half_grid,
            target_point_dict=sampled_target_point_dict,
            verbose=False)[:2]

        numpy.random.shuffle(batch_indices)
        downsized_predictor_matrix = downsized_predictor_matrix[
            batch_indices, ...].astype('float32')
        target_values = target_values[batch_indices]

        target_matrix = keras.utils.to_categorical(target_values, num_classes)
        actual_class_fractions = numpy.sum(target_matrix, axis=0)
        print 'Fraction of examples in each class: {0:s}'.format(
            str(actual_class_fractions))

        full_predictor_matrix = None
        full_target_matrix = None
        num_times_in_memory = 0

        yield (downsized_predictor_matrix, target_matrix)


def full_size_3d_example_generator(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes):
    """Generates full-size 3-D examples from raw files.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param num_classes: Same.
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: target_matrix: E-by-M-by-N numpy array of target values.  Each
        value is an integer from the list `front_utils.VALID_INTEGER_IDS`.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_target_times = len(frontal_grid_file_names)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_examples_in_memory = 0

    predictor_matrix = None
    target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print '\n'
            tuple_of_predictor_matrices = ()

            for j in range(num_predictors):
                print 'Reading data from: "{0:s}"...'.format(
                    narr_file_name_matrix[target_time_index, j])

                this_field_predictor_matrix = (
                    processed_narr_io.read_fields_from_file(
                        narr_file_name_matrix[target_time_index, j])
                )[0]
                this_field_predictor_matrix = (
                    ml_utils.fill_nans_in_predictor_images(
                        this_field_predictor_matrix)
                )

                tuple_of_predictor_matrices += (this_field_predictor_matrix,)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            this_predictor_matrix = ml_utils.stack_predictor_variables(
                tuple_of_predictor_matrices)
            this_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_predictor_matrix)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_predictor_matrix.shape[1],
                num_columns_per_image=this_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_predictor_matrix)
            this_frontal_grid_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

            if target_matrix is None or target_matrix.size == 0:
                predictor_matrix = copy.deepcopy(this_predictor_matrix)
                target_matrix = copy.deepcopy(this_frontal_grid_matrix)
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0)
                target_matrix = numpy.concatenate(
                    (target_matrix, this_frontal_grid_matrix), axis=0)

            num_examples_in_memory = target_matrix.shape[0]

        predictor_matrix_to_return = predictor_matrix[
            batch_indices, ...].astype('float32')
        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_matrix[batch_indices, ...] > 0))

        target_matrix_to_return = keras.utils.to_categorical(
            target_matrix[batch_indices, ...], num_classes)
        target_matrix_to_return = numpy.reshape(
            target_matrix_to_return, target_matrix.shape + (num_classes,))

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_matrix = numpy.delete(target_matrix, batch_indices, axis=0)
        num_examples_in_memory = target_matrix.shape[0]

        yield (predictor_matrix_to_return, target_matrix_to_return)


def full_size_4d_example_generator(
        num_examples_per_batch, first_target_time_unix_sec,
        last_target_time_unix_sec, num_lead_time_steps,
        predictor_time_step_offsets, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, num_classes):
    """Generates full-size 4-D examples from raw files.

    :param num_examples_per_batch: See doc for `downsized_3d_example_generator`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param num_lead_time_steps: See doc for `find_input_files_for_4d_examples`.
    :param predictor_time_step_offsets: Same.
    :param top_narr_directory_name: Same.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param num_classes: Same.
    :return: predictor_matrix: E-by-M-by-N-by-T-by-C numpy array of predictor
        values.
    :return: target_matrix: See doc for `full_size_3d_example_generator`.
    """

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_integer(num_classes)
    error_checking.assert_is_geq(num_classes, 2)
    error_checking.assert_is_leq(num_classes, 3)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_4d_examples(
        first_target_time_unix_sec=first_target_time_unix_sec,
        last_target_time_unix_sec=last_target_time_unix_sec,
        num_lead_time_steps=num_lead_time_steps,
        predictor_time_step_offsets=predictor_time_step_offsets,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    num_target_times = len(frontal_grid_file_names)
    num_predictor_times_per_example = len(predictor_time_step_offsets)
    num_predictors = len(narr_predictor_names)
    batch_indices = numpy.linspace(
        0, num_examples_per_batch - 1, num=num_examples_per_batch, dtype=int)

    target_time_index = 0
    num_examples_in_memory = 0

    predictor_matrix = None
    target_matrix = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print '\n'
            tuple_of_4d_predictor_matrices = ()

            for i in range(num_predictor_times_per_example):
                tuple_of_3d_predictor_matrices = ()

                for j in range(num_predictors):
                    print 'Reading data from: "{0:s}"...'.format(
                        narr_file_name_matrix[target_time_index, i, j])

                    this_field_predictor_matrix = (
                        processed_narr_io.read_fields_from_file(
                            narr_file_name_matrix[target_time_index, i, j])
                    )[0]
                    this_field_predictor_matrix = (
                        ml_utils.fill_nans_in_predictor_images(
                            this_field_predictor_matrix)
                    )

                    tuple_of_3d_predictor_matrices += (
                        this_field_predictor_matrix,)

                tuple_of_4d_predictor_matrices += (
                    ml_utils.stack_predictor_variables(
                        tuple_of_3d_predictor_matrices),
                )

            this_predictor_matrix = ml_utils.stack_time_steps(
                tuple_of_4d_predictor_matrices)

            print 'Reading data from: "{0:s}"...'.format(
                frontal_grid_file_names[target_time_index])
            this_frontal_grid_table = fronts_io.read_narr_grids_from_file(
                frontal_grid_file_names[target_time_index])

            target_time_index += 1
            if target_time_index >= num_target_times:
                target_time_index = 0

            this_predictor_matrix, _ = ml_utils.normalize_predictors(
                predictor_matrix=this_predictor_matrix)

            this_frontal_grid_matrix = ml_utils.front_table_to_images(
                frontal_grid_table=this_frontal_grid_table,
                num_rows_per_image=this_predictor_matrix.shape[1],
                num_columns_per_image=this_predictor_matrix.shape[2])

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.binarize_front_images(
                    this_frontal_grid_matrix)

            this_predictor_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_predictor_matrix)
            this_frontal_grid_matrix = ml_utils.subset_narr_grid_for_fcn_input(
                this_frontal_grid_matrix)

            if num_classes == 2:
                this_frontal_grid_matrix = ml_utils.dilate_binary_target_images(
                    target_matrix=this_frontal_grid_matrix,
                    dilation_distance_metres=dilation_distance_metres,
                    verbose=False)
            else:
                this_frontal_grid_matrix = (
                    ml_utils.dilate_ternary_target_images(
                        target_matrix=this_frontal_grid_matrix,
                        dilation_distance_metres=dilation_distance_metres,
                        verbose=False)
                )

            if target_matrix is None or target_matrix.size == 0:
                predictor_matrix = copy.deepcopy(this_predictor_matrix)
                target_matrix = copy.deepcopy(this_frontal_grid_matrix)
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0)
                target_matrix = numpy.concatenate(
                    (target_matrix, this_frontal_grid_matrix), axis=0)

            num_examples_in_memory = target_matrix.shape[0]

        predictor_matrix_to_return = predictor_matrix[
            batch_indices, ...].astype('float32')
        print 'Fraction of examples with a front = {0:.4f}'.format(
            numpy.mean(target_matrix[batch_indices, ...] > 0))

        target_matrix_to_return = keras.utils.to_categorical(
            target_matrix[batch_indices, ...], num_classes)
        target_matrix_to_return = numpy.reshape(
            target_matrix_to_return, target_matrix.shape + (num_classes,))

        predictor_matrix = numpy.delete(predictor_matrix, batch_indices, axis=0)
        target_matrix = numpy.delete(target_matrix, batch_indices, axis=0)
        num_examples_in_memory = target_matrix.shape[0]

        yield (predictor_matrix_to_return, target_matrix_to_return)


def prep_downsized_3d_examples_to_write(
        target_time_unix_sec, max_num_examples, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names,
        pressure_level_mb, dilation_distance_metres, class_fractions,
        num_rows_in_half_grid, num_columns_in_half_grid,
        normalization_type_string=ml_utils.Z_SCORE_STRING,
        narr_mask_matrix=None):
    """Prepares downsized 3-D examples for writing to a file.

    :param target_time_unix_sec: Target time.
    :param max_num_examples: Maximum number of examples.
    :param top_narr_directory_name: See doc for
        `find_input_files_for_3d_examples`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: See doc for
        `downsized_3d_example_generator`.
    :param class_fractions: length-3 numpy array of sampling fractions for
        target variable.  These are the fractions of (NF, WF, CF) examples,
        respectively, to be created.
    :param num_rows_in_half_grid: See doc for
        `downsized_3d_example_generator`.
    :param num_columns_in_half_grid: Same.
    :param normalization_type_string: Normalization type (see
        `machine_learning_utils.normalize_predictors` for details).
    :param narr_mask_matrix: See doc for
        `downsized_3d_example_generator`.

    :return: example_dict: Dictionary with the following keys.
    example_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        images.
    example_dict['target_matrix']: E-by-K numpy array of Boolean labels (all 0
        or 1, but technically the type is "float64").
    example_dict['target_times_unix_sec']: length-E numpy array of target times
        (these will all be the same).
    example_dict['row_indices']: length-E numpy array with NARR row at the
        center of each example.
    example_dict['column_indices']: length-E numpy array with NARR column at the
        center of each example.
    example_dict['first_normalization_param_matrix']: E-by-C numpy array with
        values of first normalization parameter (either minimum or mean value).
    example_dict['second_normalization_param_matrix']: E-by-C numpy array with
        values of second normalization parameter (either max value or standard
        deviation).
    """

    error_checking.assert_is_numpy_array(
        class_fractions, exact_dimensions=numpy.array([3]))
    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    (narr_file_name_matrix, frontal_grid_file_names
    ) = find_input_files_for_3d_examples(
        first_target_time_unix_sec=target_time_unix_sec,
        last_target_time_unix_sec=target_time_unix_sec,
        top_narr_directory_name=top_narr_directory_name,
        top_frontal_grid_dir_name=top_frontal_grid_dir_name,
        narr_predictor_names=narr_predictor_names,
        pressure_level_mb=pressure_level_mb)

    if len(frontal_grid_file_names) == 0:
        return None

    narr_file_names = narr_file_name_matrix[0, ...]
    frontal_grid_file_name = frontal_grid_file_names[0]

    tuple_of_predictor_matrices = ()
    num_predictors = len(narr_predictor_names)

    for j in range(num_predictors):
        print 'Reading data from: "{0:s}"...'.format(narr_file_names[j])

        this_field_matrix = processed_narr_io.read_fields_from_file(
            narr_file_names[j])[0]
        this_field_matrix = ml_utils.fill_nans_in_predictor_images(
            this_field_matrix)
        tuple_of_predictor_matrices += (this_field_matrix,)

    predictor_matrix = ml_utils.stack_predictor_variables(
        tuple_of_predictor_matrices)
    predictor_matrix, normalization_dict = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix,
        normalization_type_string=normalization_type_string)

    print 'Reading data from: "{0:s}"...'.format(frontal_grid_file_name)
    frontal_grid_table = fronts_io.read_narr_grids_from_file(
        frontal_grid_file_name)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=frontal_grid_table,
        num_rows_per_image=predictor_matrix.shape[1],
        num_columns_per_image=predictor_matrix.shape[2])

    target_matrix = ml_utils.dilate_ternary_target_images(
        target_matrix=target_matrix,
        dilation_distance_metres=dilation_distance_metres, verbose=False)

    sampled_target_point_dict = ml_utils.sample_target_points(
        target_matrix=target_matrix, class_fractions=class_fractions,
        num_points_to_sample=max_num_examples, mask_matrix=narr_mask_matrix)
    if sampled_target_point_dict is None:
        return None

    (predictor_matrix, target_values, _, row_indices, column_indices
    ) = ml_utils.downsize_grids_around_selected_points(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        num_rows_in_half_window=num_rows_in_half_grid,
        num_columns_in_half_window=num_columns_in_half_grid,
        target_point_dict=sampled_target_point_dict, verbose=False)

    target_matrix = keras.utils.to_categorical(target_values, 3)
    actual_class_fractions = numpy.sum(target_matrix, axis=0)
    print 'Fraction of examples in each class: {0:s}'.format(
        str(actual_class_fractions))

    example_dict = {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_MATRIX_KEY: target_matrix,
        TARGET_TIMES_KEY:
            numpy.full(target_matrix.shape[0], target_time_unix_sec, dtype=int),
        ROW_INDICES_KEY: row_indices,
        COLUMN_INDICES_KEY: column_indices
    }

    if normalization_type_string == ml_utils.MINMAX_STRING:
        example_dict.update({
            FIRST_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MIN_VALUE_MATRIX_KEY],
            SECOND_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MAX_VALUE_MATRIX_KEY]
        })
    else:
        example_dict.update({
            FIRST_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MEAN_VALUE_MATRIX_KEY],
            SECOND_NORM_PARAM_KEY:
                normalization_dict[ml_utils.STDEV_MATRIX_KEY]
        })

    return example_dict


def find_downsized_3d_example_file(
        top_directory_name, shuffled=False, first_target_time_unix_sec=None,
        last_target_time_unix_sec=None, batch_number=None,
        raise_error_if_missing=True):
    """Finds file with downsized 3-D examples.

    :param top_directory_name: Name of top-level directory for files with
        downsized 3-D examples.
    :param shuffled: Boolean flag.  If examples in the file were shuffled by
        shuffle_downsized_3d_files.py, this should be True.
    :param first_target_time_unix_sec: [used iff `shuffled == False`]
        First target time in file.
    :param last_target_time_unix_sec: [used iff `shuffled == False`]
        Last target time in file.
    :param batch_number: [used iff `shuffled == True`] Batch number (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: downsized_3d_file_name: Path to file with downsized 3-D examples.
        If file is missing and `raise_error_if_missing = False`, this is the
        *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(shuffled)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if shuffled:
        error_checking.assert_is_integer(batch_number)
        error_checking.assert_is_geq(batch_number, 0)

        first_batch_number = int(number_rounding.floor_to_nearest(
            batch_number, NUM_BATCHES_PER_DIRECTORY))
        last_batch_number = first_batch_number + NUM_BATCHES_PER_DIRECTORY - 1

        downsized_3d_file_name = (
            '{0:s}/batches{1:07d}-{2:07d}/downsized_3d_examples_batch{3:07d}.nc'
        ).format(top_directory_name, first_batch_number, last_batch_number,
                 batch_number)
    else:
        error_checking.assert_is_integer(first_target_time_unix_sec)
        error_checking.assert_is_integer(last_target_time_unix_sec)
        error_checking.assert_is_geq(
            last_target_time_unix_sec, first_target_time_unix_sec)

        downsized_3d_file_name = (
            '{0:s}/downsized_3d_examples_{1:s}-{2:s}.nc'
        ).format(
            top_directory_name,
            time_conversion.unix_sec_to_string(
                first_target_time_unix_sec, TIME_FORMAT),
            time_conversion.unix_sec_to_string(
                last_target_time_unix_sec, TIME_FORMAT)
        )

    if raise_error_if_missing and not os.path.isfile(downsized_3d_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            downsized_3d_file_name)
        raise ValueError(error_string)

    return downsized_3d_file_name


def find_downsized_3d_example_files(
        top_directory_name, shuffled=False, first_target_time_unix_sec=None,
        last_target_time_unix_sec=None, first_batch_number=None,
        last_batch_number=None):
    """Finds many files with downsized 3-D examples.

    :param top_directory_name: See doc for `find_downsized_3d_example_file`.
    :param shuffled: Same.
    :param first_target_time_unix_sec: [used iff `shuffled == False`]
        First target time.
    :param last_target_time_unix_sec: [used iff `shuffled == False`]
        Last target time.
    :param first_batch_number: [used iff `shuffled == True`] First batch number.
    :param last_batch_number: [used iff `shuffled == True`] Last batch number.
    :return: downsized_3d_file_names: 1-D list of file paths.
    :raises: ValueError: if no files are found.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(shuffled)

    if shuffled:
        error_checking.assert_is_integer(first_batch_number)
        error_checking.assert_is_integer(last_batch_number)
        error_checking.assert_is_geq(first_batch_number, 0)
        error_checking.assert_is_geq(last_batch_number, first_batch_number)

        downsized_3d_file_pattern = (
            '{0:s}/batches{1:s}-{1:s}/downsized_3d_examples_batch{1:s}.nc'
        ).format(top_directory_name, BATCH_NUMBER_REGEX)
    else:
        error_checking.assert_is_integer(first_target_time_unix_sec)
        error_checking.assert_is_integer(last_target_time_unix_sec)
        error_checking.assert_is_geq(
            last_target_time_unix_sec, first_target_time_unix_sec)

        downsized_3d_file_pattern = (
            '{0:s}/downsized_3d_examples_{1:s}-{1:s}.nc'
        ).format(top_directory_name, TIME_FORMAT_REGEX, TIME_FORMAT_REGEX)

    downsized_3d_file_names = glob.glob(downsized_3d_file_pattern)
    if len(downsized_3d_file_names) == 0:
        error_string = 'Cannot find any files with the pattern: "{0:s}"'.format(
            downsized_3d_file_pattern)
        raise ValueError(error_string)

    if shuffled:
        batch_numbers = numpy.array(
            [_file_name_to_batch_number(f) for f in downsized_3d_file_names],
            dtype=int)
        good_indices = numpy.where(numpy.logical_and(
            batch_numbers >= first_batch_number,
            batch_numbers <= last_batch_number
        ))[0]

        if len(good_indices) == 0:
            error_string = (
                'Cannot find any files with batch number in [{0:d}, {1:d}].'
            ).format(first_batch_number, last_batch_number)
            raise ValueError(error_string)

    else:
        downsized_3d_file_names.sort()
        file_start_times_unix_sec = numpy.array(
            [_file_name_to_target_times(f)[0] for f in downsized_3d_file_names],
            dtype=int)
        file_end_times_unix_sec = numpy.array(
            [_file_name_to_target_times(f)[1] for f in downsized_3d_file_names],
            dtype=int)

        good_indices = numpy.where(numpy.invert(numpy.logical_or(
            file_start_times_unix_sec > last_target_time_unix_sec,
            file_end_times_unix_sec < first_target_time_unix_sec
        )))[0]

        if len(good_indices) == 0:
            error_string = (
                'Cannot find any files with target time from {0:s} to {1:s}.'
            ).format(
                time_conversion.unix_sec_to_string(
                    first_target_time_unix_sec, TIME_FORMAT),
                time_conversion.unix_sec_to_string(
                    last_target_time_unix_sec, TIME_FORMAT)
            )
            raise ValueError(error_string)

    return [downsized_3d_file_names[i] for i in good_indices]


def write_downsized_3d_examples(
        netcdf_file_name, example_dict, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, narr_mask_matrix=None, append_to_file=False):
    """Writes downsized 3-D examples to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param example_dict: Dictionary created by
        `prep_downsized_3d_examples_to_write`.
    :param narr_predictor_names: See doc for
        `prep_downsized_3d_examples_to_write`.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param narr_mask_matrix: Same.
    :param append_to_file: Boolean flag.  If True, this method will append to an
        existing file.  If False, will create a new file, overwriting the
        existing file if necessary.
    """

    # Check input args.
    error_checking.assert_is_integer(pressure_level_mb)
    error_checking.assert_is_greater(pressure_level_mb, 0)
    error_checking.assert_is_geq(dilation_distance_metres, 0.)
    error_checking.assert_is_boolean(append_to_file)

    num_predictors = example_dict[PREDICTOR_MATRIX_KEY].shape[3]

    error_checking.assert_is_string_list(narr_predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(narr_predictor_names),
        exact_dimensions=numpy.array([num_predictors]))

    num_narr_rows, num_narr_columns = nwp_model_utils.get_grid_dimensions(
        model_name=nwp_model_utils.NARR_MODEL_NAME)
    if narr_mask_matrix is None:
        narr_mask_matrix = numpy.full(
            (num_narr_rows, num_narr_columns), 1, dtype=int)

    ml_utils.check_narr_mask(narr_mask_matrix)

    # Do other stuff.
    if append_to_file:
        netcdf_dataset = netCDF4.Dataset(
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET')

        orig_predictor_names = netCDF4.chartostring(
            netcdf_dataset.variables[PREDICTOR_NAMES_KEY][:]
        )
        orig_predictor_names = [str(s) for s in orig_predictor_names]
        assert orig_predictor_names == narr_predictor_names

        orig_pressure_level_mb = int(
            getattr(netcdf_dataset, PRESSURE_LEVEL_KEY)
        )
        assert orig_pressure_level_mb == pressure_level_mb

        orig_dilation_distance_metres = getattr(
            netcdf_dataset, DILATION_DISTANCE_KEY)
        assert numpy.isclose(orig_dilation_distance_metres,
                             dilation_distance_metres, atol=TOLERANCE)

        orig_narr_mask_matrix = numpy.array(
            netcdf_dataset.variables[NARR_MASK_KEY][:], dtype=int)
        assert numpy.array_equal(orig_narr_mask_matrix, narr_mask_matrix)

        num_examples_orig = len(
            numpy.array(netcdf_dataset.variables[TARGET_TIMES_KEY][:])
        )
        num_examples_to_add = len(example_dict[TARGET_TIMES_KEY])

        for this_key in MAIN_KEYS:
            netcdf_dataset.variables[this_key][
                num_examples_orig:(num_examples_orig + num_examples_to_add),
                ...
            ] = example_dict[this_key]

        netcdf_dataset.close()
        return

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    netcdf_dataset = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    netcdf_dataset.setncattr(PRESSURE_LEVEL_KEY, int(pressure_level_mb))
    netcdf_dataset.setncattr(DILATION_DISTANCE_KEY, dilation_distance_metres)

    num_classes = example_dict[TARGET_MATRIX_KEY].shape[1]
    num_rows_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[1]
    num_columns_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[2]

    num_predictor_chars = 1
    for m in range(num_predictors):
        num_predictor_chars = max(
            [num_predictor_chars, len(narr_predictor_names[m])]
        )

    netcdf_dataset.createDimension(NARR_ROW_DIMENSION_KEY, num_narr_rows)
    netcdf_dataset.createDimension(NARR_COLUMN_DIMENSION_KEY, num_narr_columns)
    netcdf_dataset.createDimension(EXAMPLE_DIMENSION_KEY, None)
    netcdf_dataset.createDimension(
        EXAMPLE_ROW_DIMENSION_KEY, num_rows_per_example)
    netcdf_dataset.createDimension(
        EXAMPLE_COLUMN_DIMENSION_KEY, num_columns_per_example)
    netcdf_dataset.createDimension(PREDICTOR_DIMENSION_KEY, num_predictors)
    netcdf_dataset.createDimension(CHARACTER_DIMENSION_KEY, num_predictor_chars)
    netcdf_dataset.createDimension(CLASS_DIMENSION_KEY, num_classes)

    string_type = 'S{0:d}'.format(num_predictor_chars)
    predictor_names_as_char_array = netCDF4.stringtochar(numpy.array(
        narr_predictor_names, dtype=string_type
    ))

    netcdf_dataset.createVariable(
        PREDICTOR_NAMES_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIMENSION_KEY, CHARACTER_DIMENSION_KEY)
    )
    netcdf_dataset.variables[PREDICTOR_NAMES_KEY][:] = numpy.array(
        predictor_names_as_char_array)

    netcdf_dataset.createVariable(
        NARR_MASK_KEY, datatype=numpy.int32,
        dimensions=(NARR_ROW_DIMENSION_KEY, NARR_COLUMN_DIMENSION_KEY)
    )
    netcdf_dataset.variables[NARR_MASK_KEY][:] = narr_mask_matrix

    netcdf_dataset.createVariable(
        PREDICTOR_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ROW_DIMENSION_KEY,
                    EXAMPLE_COLUMN_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    netcdf_dataset.variables[PREDICTOR_MATRIX_KEY][:] = example_dict[
        PREDICTOR_MATRIX_KEY]

    netcdf_dataset.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    netcdf_dataset.variables[TARGET_MATRIX_KEY][:] = example_dict[
        TARGET_MATRIX_KEY]

    netcdf_dataset.createVariable(
        TARGET_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[TARGET_TIMES_KEY][:] = example_dict[
        TARGET_TIMES_KEY]

    netcdf_dataset.createVariable(
        ROW_INDICES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[ROW_INDICES_KEY][:] = example_dict[ROW_INDICES_KEY]

    netcdf_dataset.createVariable(
        COLUMN_INDICES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    netcdf_dataset.variables[COLUMN_INDICES_KEY][:] = example_dict[
        COLUMN_INDICES_KEY]

    netcdf_dataset.createVariable(
        FIRST_NORM_PARAM_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    netcdf_dataset.variables[FIRST_NORM_PARAM_KEY][:] = example_dict[
        FIRST_NORM_PARAM_KEY]

    netcdf_dataset.createVariable(
        SECOND_NORM_PARAM_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    netcdf_dataset.variables[SECOND_NORM_PARAM_KEY][:] = example_dict[
        SECOND_NORM_PARAM_KEY]

    netcdf_dataset.close()


def read_downsized_3d_examples(
        netcdf_file_name, metadata_only=False, predictor_names_to_keep=None,
        num_half_rows_to_keep=None, num_half_columns_to_keep=None,
        first_time_to_keep_unix_sec=None, last_time_to_keep_unix_sec=None):
    """Reads downsized 3-D examples from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param metadata_only: Boolean flag.  If True, will return only metadata
        (everything except predictor and target matrices).
    :param predictor_names_to_keep: 1-D list with names of predictor variables
        to keep (each name must be accepted by `check_field_name`).  If
        `predictor_names_to_keep is None`, all predictors in the file will be
        returned.
    :param num_half_rows_to_keep: [used iff `metadata_only == False`]
        Determines number of rows to keep for each example.  Examples will be
        cropped so that the center of the original image is the center of the
        new image.  If `num_half_rows_to_keep is None`, examples will not be
        cropped.
    :param num_half_columns_to_keep: [used iff `metadata_only == False`]
        Same but for columns.
    :param first_time_to_keep_unix_sec: Will throw out earlier target times.
    :param last_time_to_keep_unix_sec: Will throw out later target times.
    :return: example_dict: Dictionary with the following keys.
        `first_normalization_param_matrix` and
        `second_normalization_param_matrix` may not be present (present only in
        newer files).

    example_dict['predictor_matrix']: See doc for
        `prep_downsized_3d_examples_to_write`.
    example_dict['target_matrix']: Same.
    example_dict['target_times_unix_sec']: Same.
    example_dict['row_indices']: Same.
    example_dict['column_indices']: Same.
    example_dict['first_normalization_param_matrix']: Same.
    example_dict['second_normalization_param_matrix']: Same.
    example_dict['predictor_names_to_keep']: See doc for
        `write_downsized_3d_examples`.
    example_dict['pressure_level_mb']: Same.
    example_dict['dilation_distance_metres']: Same.
    example_dict['narr_mask_matrix']: Same.
    """

    # Check input args.
    if predictor_names_to_keep is not None:
        error_checking.assert_is_numpy_array(
            numpy.array(predictor_names_to_keep), num_dimensions=1)

    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = int(1e11)

    error_checking.assert_is_boolean(metadata_only)
    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

    # Do other stuff.
    netcdf_dataset = netcdf_io.open_netcdf(netcdf_file_name)

    narr_predictor_names = netCDF4.chartostring(
        netcdf_dataset.variables[PREDICTOR_NAMES_KEY][:]
    )
    narr_predictor_names = [str(s) for s in narr_predictor_names]
    if predictor_names_to_keep is None:
        predictor_names_to_keep = narr_predictor_names + []

    target_times_unix_sec = numpy.array(
        netcdf_dataset.variables[TARGET_TIMES_KEY][:], dtype=int)
    row_indices = numpy.array(
        netcdf_dataset.variables[ROW_INDICES_KEY][:], dtype=int)
    column_indices = numpy.array(
        netcdf_dataset.variables[COLUMN_INDICES_KEY][:], dtype=int)

    found_normalization_params = (
        FIRST_NORM_PARAM_KEY in netcdf_dataset.variables or
        SECOND_NORM_PARAM_KEY in netcdf_dataset.variables
    )

    if found_normalization_params:
        first_normalization_param_matrix = numpy.array(
            netcdf_dataset.variables[FIRST_NORM_PARAM_KEY][:])
        second_normalization_param_matrix = numpy.array(
            netcdf_dataset.variables[SECOND_NORM_PARAM_KEY][:])

    if not metadata_only:
        predictor_matrix = numpy.array(
            netcdf_dataset.variables[PREDICTOR_MATRIX_KEY][:])
        target_matrix = numpy.array(
            netcdf_dataset.variables[TARGET_MATRIX_KEY][:])

        these_indices = numpy.array(
            [narr_predictor_names.index(p) for p in predictor_names_to_keep],
            dtype=int)
        predictor_matrix = predictor_matrix[..., these_indices]
        predictor_matrix = _decrease_example_size(
            predictor_matrix=predictor_matrix,
            num_half_rows=num_half_rows_to_keep,
            num_half_columns=num_half_columns_to_keep)

    indices_to_keep = numpy.where(numpy.logical_and(
        target_times_unix_sec >= first_time_to_keep_unix_sec,
        target_times_unix_sec <= last_time_to_keep_unix_sec
    ))[0]

    example_dict = {
        TARGET_TIMES_KEY: target_times_unix_sec[indices_to_keep],
        ROW_INDICES_KEY: row_indices[indices_to_keep],
        COLUMN_INDICES_KEY: column_indices[indices_to_keep],
        PREDICTOR_NAMES_KEY: predictor_names_to_keep,
        PRESSURE_LEVEL_KEY: int(getattr(netcdf_dataset, PRESSURE_LEVEL_KEY)),
        DILATION_DISTANCE_KEY: getattr(netcdf_dataset, DILATION_DISTANCE_KEY),
        NARR_MASK_KEY:
            numpy.array(netcdf_dataset.variables[NARR_MASK_KEY][:], dtype=int)
    }

    if found_normalization_params:
        example_dict = {
            FIRST_NORM_PARAM_KEY:
                first_normalization_param_matrix[indices_to_keep, ...],
            SECOND_NORM_PARAM_KEY:
                second_normalization_param_matrix[indices_to_keep, ...]
        }

    if not metadata_only:
        example_dict.update({
            PREDICTOR_MATRIX_KEY:
                predictor_matrix[indices_to_keep, ...].astype('float32'),
            TARGET_MATRIX_KEY:
                target_matrix[indices_to_keep, ...].astype('float64')
        })

    netcdf_dataset.close()
    return example_dict
