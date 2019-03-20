"""IO methods for learning examples.

A 'learning example' is a pre-processed set of predictor fields and target
value, ready for input to a machine-learning model.

--- NOTATION ---

The following letters will be used throughout this module.

E = number of examples
M = number of rows in grid
N = number of columns in grid
C = number of channels (predictor variables)
"""

import copy
import glob
import os.path
import warnings
import numpy
import netCDF4
from keras.utils import to_categorical
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_io import fronts_io
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

TOLERANCE = 1e-6
LARGE_INTEGER = int(1e11)

PATHLESS_FILE_NAME_PREFIX = 'downsized_3d_examples'

TIME_FORMAT = '%Y%m%d%H'
TIME_FORMAT_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9][0-2][0-9]'
BATCH_NUMBER_REGEX = '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
NUM_BATCHES_PER_DIRECTORY = 1000

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'target_times_unix_sec'
ROW_INDICES_KEY = 'row_indices'
COLUMN_INDICES_KEY = 'column_indices'
NORMALIZATION_TYPE_KEY = 'normalization_type_string'
PREDICTOR_NAMES_KEY = 'narr_predictor_names'
PRESSURE_LEVEL_KEY = 'pressure_level_mb'
DILATION_DISTANCE_KEY = 'dilation_distance_metres'
MASK_MATRIX_KEY = 'narr_mask_matrix'

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
    PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY, VALID_TIMES_KEY, ROW_INDICES_KEY,
    COLUMN_INDICES_KEY, FIRST_NORM_PARAM_KEY, SECOND_NORM_PARAM_KEY
]


def _file_name_to_times(example_file_name):
    """Parses valid times from file name.

    :param example_file_name: See doc for `find_file`.
    :return: first_valid_time_unix_sec: First time in file.
    :return: last_valid_time_unix_sec: Last time in file.
    """

    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    valid_time_strings = extensionless_file_name.split(
        '{0:s}_'.format(PATHLESS_FILE_NAME_PREFIX)
    )[-1].split('-')

    first_valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_strings[0], TIME_FORMAT)
    last_valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_strings[-1], TIME_FORMAT)

    return first_valid_time_unix_sec, last_valid_time_unix_sec


def _file_name_to_batch_number(example_file_name):
    """Parses batch number from file name.

    :param example_file_name: See doc for `find_file`.
    :return: batch_number: Integer.
    :raises: ValueError: if batch number cannot be parsed from file name.
    """

    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    batch_string = extensionless_file_name.split(
        '{0:s}_batch'.format(PATHLESS_FILE_NAME_PREFIX)
    )[-1]

    try:
        return int(batch_string)
    except ValueError:
        error_string = (
            'Batch number cannot be parsed from file name: "{0:s}"'
        ).format(example_file_name)

        raise ValueError(error_string)


def _shrink_predictor_grid(predictor_matrix, num_half_rows=None,
                           num_half_columns=None):
    """Shrinks predictor grid (by cropping around the center).

    M = original num rows in grid
    N = original num columns in grid
    m = final num rows in grid (after shrinking)
    n = final num columns in grid (after shrinking)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param num_half_rows: Number of rows in half-grid (on either side of center)
        after shrinking.  If `num_half_rows is None`, rows will not be cropped.
    :param num_half_columns: Same but for columns.
    :return: predictor_matrix: Same as input, except that dimensions are now
        E x m x n x C.
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


def create_examples(
        top_predictor_dir_name, top_gridded_front_dir_name, valid_time_unix_sec,
        predictor_names, pressure_level_mb, num_half_rows, num_half_columns,
        dilation_distance_metres, class_fractions, max_num_examples,
        normalization_type_string, narr_mask_matrix=None):
    """Creates examples.

    :param top_predictor_dir_name: Name of top-level directory with predictors.
        Files therein will be found by `predictor_io.find_file` and read by
        `predictor_io.read_file`.
    :param top_gridded_front_dir_name: Name of top-level directory with gridded
        front labels.  Files therein will be found by
        `fronts_io.find_gridded_file` and read by
        `fronts_io.read_grid_from_file`.
    :param valid_time_unix_sec: Valid time.
    :param predictor_names: 1-D list of predictor names (each must be accepted
        by `predictor_utils.check_field_name`).
    :param pressure_level_mb: Pressure level (millibars) for predictors.
    :param num_half_rows: Number of half-rows (on either side of center) for
        predictor grid.
    :param num_half_columns: Same but for columns.
    :param dilation_distance_metres: Dilation distance for gridded warm-front
        and cold-front labels.
    :param class_fractions: length-3 numpy array with downsampling fractions for
        the 3 classes (no front, warm front, cold front).
    :param max_num_examples: Max number of examples to create.
    :param normalization_type_string: Normalization method for predictors (see
        `machine_learning_utils.normalize_predictors`).
    :param narr_mask_matrix: See doc for
        `machine_learning_utils.check_narr_mask`.  Masked grid cells will not be
        used as the center of a learning example.  If this is None, no grid
        cells will be masked.
    :return: example_dict: Dictionary with the following keys.
    example_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    example_dict['target_matrix']: E-by-3 numpy array of target values (labels).
        All elements are 0 or 1, but the array type is "float64".
    example_dict['target_times_unix_sec']: length-E numpy array of valid times.
    example_dict['row_indices']: length-E numpy array of row indices in NARR
        grid.  If row_indices[i] = j, the center of the predictor grid for the
        [i]th example is the [j]th row of the NARR grid.
    example_dict['column_indices']: Same but for columns.
    example_dict['first_normalization_param_matrix']: E-by-C numpy array with
        values of first normalization param (either minimum or mean).
    example_dict['second_normalization_param_matrix']: E-by-C numpy array with
        values of second normalization param (either max or standard deviation).
    example_dict['normalization_type_string']: Same as input (metadata).
    example_dict['narr_predictor_names']: Same as input (metadata).
    example_dict['pressure_level_mb']: Same as input (metadata).
    example_dict['dilation_distance_metres']: Same as input (metadata).
    example_dict['narr_mask_matrix']: Same as input (metadata).
    """

    these_expected_dim = numpy.array([3], dtype=int)
    error_checking.assert_is_numpy_array(
        class_fractions, exact_dimensions=these_expected_dim)

    if narr_mask_matrix is not None:
        ml_utils.check_narr_mask(narr_mask_matrix)

    gridded_front_file_name = fronts_io.find_gridded_file(
        top_directory_name=top_gridded_front_dir_name,
        valid_time_unix_sec=valid_time_unix_sec, raise_error_if_missing=False)

    if not os.path.isfile(gridded_front_file_name):
        warning_string = (
            'POTENTIAL PROBLEM.  Cannot find file expected at: "{0:s}"'
        ).format(gridded_front_file_name)

        warnings.warn(warning_string)
        return None

    predictor_file_name = predictor_io.find_file(
        top_directory_name=top_predictor_dir_name,
        valid_time_unix_sec=valid_time_unix_sec, raise_error_if_missing=True)

    print 'Reading data from: "{0:s}"...'.format(predictor_file_name)
    predictor_dict = predictor_io.read_file(
        netcdf_file_name=predictor_file_name,
        pressure_levels_to_keep_mb=numpy.array([pressure_level_mb], dtype=int),
        field_names_to_keep=predictor_names)

    predictor_matrix = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY
    ][[0], ..., 0, :]

    if narr_mask_matrix is None:
        narr_mask_matrix = numpy.full(predictor_matrix.shape[1:3], 1, dtype=int)

    predictor_matrix, normalization_dict = ml_utils.normalize_predictors(
        predictor_matrix=predictor_matrix,
        normalization_type_string=normalization_type_string)

    print 'Reading data from: "{0:s}"...'.format(gridded_front_file_name)
    gridded_front_table = fronts_io.read_grid_from_file(
        gridded_front_file_name)

    target_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
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

    (predictor_matrix, target_values, time_indices, row_indices, column_indices
    ) = ml_utils.downsize_grids_around_selected_points(
        predictor_matrix=predictor_matrix, target_matrix=target_matrix,
        num_rows_in_half_window=num_half_rows,
        num_columns_in_half_window=num_half_columns,
        target_point_dict=sampled_target_point_dict, verbose=False)

    target_matrix = to_categorical(target_values, 3)
    actual_class_fractions = numpy.sum(target_matrix, axis=0)
    print 'Fraction of examples in each class: {0:s}'.format(
        str(actual_class_fractions)
    )

    example_dict = {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_MATRIX_KEY: target_matrix,
        VALID_TIMES_KEY:
            numpy.full(len(row_indices), valid_time_unix_sec, dtype=int),
        ROW_INDICES_KEY: row_indices,
        COLUMN_INDICES_KEY: column_indices,
        NORMALIZATION_TYPE_KEY: normalization_type_string,
        PREDICTOR_NAMES_KEY: predictor_names,
        PRESSURE_LEVEL_KEY: pressure_level_mb,
        DILATION_DISTANCE_KEY: dilation_distance_metres,
        MASK_MATRIX_KEY: narr_mask_matrix
    }

    if normalization_type_string == ml_utils.MINMAX_STRING:
        example_dict.update({
            FIRST_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MIN_VALUE_MATRIX_KEY][
                    time_indices, ...],
            SECOND_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MAX_VALUE_MATRIX_KEY][
                    time_indices, ...]
        })
    else:
        example_dict.update({
            FIRST_NORM_PARAM_KEY:
                normalization_dict[ml_utils.MEAN_VALUE_MATRIX_KEY][
                    time_indices, ...],
            SECOND_NORM_PARAM_KEY:
                normalization_dict[ml_utils.STDEV_MATRIX_KEY][time_indices, ...]
        })

    return example_dict


def find_file(
        top_directory_name, shuffled=False, first_valid_time_unix_sec=None,
        last_valid_time_unix_sec=None, batch_number=None,
        raise_error_if_missing=True):
    """Finds file with learning examples.

    :param top_directory_name: Name of top-level directory with example files.
    :param shuffled: Boolean flag.  If True, will look for a file with shuffled
        data (from random time steps).  If False, will look for a file with data
        from a continuous time period.
    :param first_valid_time_unix_sec: [used only if `shuffled == False`]
        First valid time in period.
    :param last_valid_time_unix_sec: [used only if `shuffled == False`]
        Last valid time in period.
    :param batch_number: [used only if `shuffled == True`] Batch number
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: example_file_name: Path to example file.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
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

        example_file_name = (
            '{0:s}/batches{1:07d}-{2:07d}/{3:s}_batch{4:07d}.nc'
        ).format(
            top_directory_name, first_batch_number, last_batch_number,
            PATHLESS_FILE_NAME_PREFIX, batch_number
        )
    else:
        error_checking.assert_is_integer(first_valid_time_unix_sec)
        error_checking.assert_is_integer(last_valid_time_unix_sec)
        error_checking.assert_is_geq(
            last_valid_time_unix_sec, first_valid_time_unix_sec)

        example_file_name = '{0:s}/{1:s}_{2:s}-{3:s}.nc'.format(
            top_directory_name, PATHLESS_FILE_NAME_PREFIX,
            time_conversion.unix_sec_to_string(
                first_valid_time_unix_sec, TIME_FORMAT),
            time_conversion.unix_sec_to_string(
                last_valid_time_unix_sec, TIME_FORMAT)
        )

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name)
        raise ValueError(error_string)

    return example_file_name


def find_many_files(
        top_directory_name, shuffled=False, first_valid_time_unix_sec=None,
        last_valid_time_unix_sec=None, first_batch_number=None,
        last_batch_number=None):
    """Finds many files with learning examples.

    :param top_directory_name: See doc for `find_file`.
    :param shuffled: Same.
    :param first_valid_time_unix_sec: Same.
    :param last_valid_time_unix_sec: Same.
    :param first_batch_number: [used only if `shuffled == True`]
        First batch number.
    :param last_batch_number: [used only if `shuffled == True`]
        Last batch number.
    :return: example_file_names: 1-D list of paths to example files.
    :raises: ValueError: if no files are found.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(shuffled)

    if shuffled:
        error_checking.assert_is_integer(first_batch_number)
        error_checking.assert_is_integer(last_batch_number)
        error_checking.assert_is_geq(first_batch_number, 0)
        error_checking.assert_is_geq(last_batch_number, first_batch_number)

        example_file_pattern = (
            '{0:s}/batches{1:s}-{1:s}/{2:s}_batch{1:s}.nc'
        ).format(top_directory_name, BATCH_NUMBER_REGEX,
                 PATHLESS_FILE_NAME_PREFIX)
    else:
        error_checking.assert_is_integer(first_valid_time_unix_sec)
        error_checking.assert_is_integer(last_valid_time_unix_sec)
        error_checking.assert_is_geq(
            last_valid_time_unix_sec, first_valid_time_unix_sec)

        example_file_pattern = '{0:s}/{1:s}_{2:s}-{2:s}.nc'.format(
            top_directory_name, PATHLESS_FILE_NAME_PREFIX, TIME_FORMAT_REGEX)

    example_file_names = glob.glob(example_file_pattern)

    if len(example_file_names) == 0:
        error_string = 'Cannot find any files with the pattern: "{0:s}"'.format(
            example_file_pattern)
        raise ValueError(error_string)

    if shuffled:
        batch_numbers = numpy.array(
            [_file_name_to_batch_number(f) for f in example_file_names],
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
        example_file_names.sort()

        file_start_times_unix_sec = numpy.array(
            [_file_name_to_times(f)[0] for f in example_file_names],
            dtype=int)
        file_end_times_unix_sec = numpy.array(
            [_file_name_to_times(f)[1] for f in example_file_names],
            dtype=int)

        good_indices = numpy.where(numpy.invert(numpy.logical_or(
            file_start_times_unix_sec > last_valid_time_unix_sec,
            file_end_times_unix_sec < first_valid_time_unix_sec
        )))[0]

        if len(good_indices) == 0:
            error_string = (
                'Cannot find any files with target time from {0:s} to {1:s}.'
            ).format(
                time_conversion.unix_sec_to_string(
                    first_valid_time_unix_sec, TIME_FORMAT),
                time_conversion.unix_sec_to_string(
                    last_valid_time_unix_sec, TIME_FORMAT)
            )
            raise ValueError(error_string)

    return [example_file_names[i] for i in good_indices]


def write_file(netcdf_file_name, example_dict, append_to_file=False):
    """Writes learning examples to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param example_dict: Dictionary created by `create_examples`.
    :param append_to_file: Boolean flag.  If True, will append to an existing
        file.  If False, will create a new one.
    """

    error_checking.assert_is_boolean(append_to_file)

    if append_to_file:
        dataset_object = netCDF4.Dataset(
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET')

        orig_pressure_level_mb = int(
            getattr(dataset_object, PRESSURE_LEVEL_KEY)
        )
        assert orig_pressure_level_mb == example_dict[PRESSURE_LEVEL_KEY]

        orig_dilation_distance_metres = getattr(
            dataset_object, DILATION_DISTANCE_KEY
        )
        assert numpy.isclose(
            orig_dilation_distance_metres,
            example_dict[DILATION_DISTANCE_KEY], atol=TOLERANCE
        )

        orig_norm_type_string = str(
            getattr(dataset_object, NORMALIZATION_TYPE_KEY)
        )
        assert orig_norm_type_string == example_dict[NORMALIZATION_TYPE_KEY]

        orig_predictor_names = netCDF4.chartostring(
            dataset_object.variables[PREDICTOR_NAMES_KEY][:]
        )
        orig_predictor_names = [str(s) for s in orig_predictor_names]
        assert orig_predictor_names == example_dict[PREDICTOR_NAMES_KEY]

        orig_narr_mask_matrix = numpy.array(
            dataset_object.variables[MASK_MATRIX_KEY][:], dtype=int
        )
        assert numpy.array_equal(
            orig_narr_mask_matrix, example_dict[MASK_MATRIX_KEY]
        )

        num_examples_orig = len(
            numpy.array(dataset_object.variables[VALID_TIMES_KEY][:])
        )
        num_examples_to_add = len(example_dict[VALID_TIMES_KEY])

        for this_key in MAIN_KEYS:
            dataset_object.variables[this_key][
                num_examples_orig:(num_examples_orig + num_examples_to_add),
                ...
            ] = example_dict[this_key]

        dataset_object.close()
        return

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    dataset_object.setncattr(
        PRESSURE_LEVEL_KEY, int(numpy.round(example_dict[PRESSURE_LEVEL_KEY]))
    )
    dataset_object.setncattr(
        DILATION_DISTANCE_KEY, example_dict[DILATION_DISTANCE_KEY]
    )
    dataset_object.setncattr(
        NORMALIZATION_TYPE_KEY, str(example_dict[NORMALIZATION_TYPE_KEY])
    )

    num_classes = example_dict[TARGET_MATRIX_KEY].shape[1]
    num_rows_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[1]
    num_columns_per_example = example_dict[PREDICTOR_MATRIX_KEY].shape[2]
    num_predictors = example_dict[PREDICTOR_MATRIX_KEY].shape[3]

    num_predictor_chars = 1
    for j in range(num_predictors):
        num_predictor_chars = max([
            num_predictor_chars, len(example_dict[PREDICTOR_NAMES_KEY][j])
        ])

    dataset_object.createDimension(
        NARR_ROW_DIMENSION_KEY, example_dict[MASK_MATRIX_KEY].shape[0]
    )
    dataset_object.createDimension(
        NARR_COLUMN_DIMENSION_KEY, example_dict[MASK_MATRIX_KEY].shape[1]
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, None)
    dataset_object.createDimension(
        EXAMPLE_ROW_DIMENSION_KEY, num_rows_per_example)
    dataset_object.createDimension(
        EXAMPLE_COLUMN_DIMENSION_KEY, num_columns_per_example)
    dataset_object.createDimension(PREDICTOR_DIMENSION_KEY, num_predictors)
    dataset_object.createDimension(CHARACTER_DIMENSION_KEY, num_predictor_chars)
    dataset_object.createDimension(CLASS_DIMENSION_KEY, num_classes)

    string_type = 'S{0:d}'.format(num_predictor_chars)
    predictor_names_as_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[PREDICTOR_NAMES_KEY], dtype=string_type
    ))

    dataset_object.createVariable(
        PREDICTOR_NAMES_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIMENSION_KEY, CHARACTER_DIMENSION_KEY)
    )
    dataset_object.variables[PREDICTOR_NAMES_KEY][:] = numpy.array(
        predictor_names_as_char_array)

    dataset_object.createVariable(
        MASK_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(NARR_ROW_DIMENSION_KEY, NARR_COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MASK_MATRIX_KEY][:] = example_dict[MASK_MATRIX_KEY]

    dataset_object.createVariable(
        PREDICTOR_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ROW_DIMENSION_KEY,
                    EXAMPLE_COLUMN_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    dataset_object.variables[PREDICTOR_MATRIX_KEY][:] = example_dict[
        PREDICTOR_MATRIX_KEY]

    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = example_dict[
        TARGET_MATRIX_KEY]

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    dataset_object.variables[VALID_TIMES_KEY][:] = example_dict[
        VALID_TIMES_KEY]

    dataset_object.createVariable(
        ROW_INDICES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY)
    dataset_object.variables[ROW_INDICES_KEY][:] = example_dict[ROW_INDICES_KEY]

    dataset_object.createVariable(
        COLUMN_INDICES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY)
    dataset_object.variables[COLUMN_INDICES_KEY][:] = example_dict[
        COLUMN_INDICES_KEY]

    dataset_object.createVariable(
        FIRST_NORM_PARAM_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    dataset_object.variables[FIRST_NORM_PARAM_KEY][:] = example_dict[
        FIRST_NORM_PARAM_KEY]

    dataset_object.createVariable(
        SECOND_NORM_PARAM_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY)
    )
    dataset_object.variables[SECOND_NORM_PARAM_KEY][:] = example_dict[
        SECOND_NORM_PARAM_KEY]

    dataset_object.close()


def read_file(
        netcdf_file_name, metadata_only=False, predictor_names_to_keep=False,
        num_half_rows_to_keep=None, num_half_columns_to_keep=None,
        first_time_to_keep_unix_sec=None, last_time_to_keep_unix_sec=None):
    """Reads learning examples from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param metadata_only: Boolean flag.  If True, will read only metadata
        (everything except predictor and target matrices).
    :param predictor_names_to_keep: 1-D list of predictors to keep.  If None,
        all predictors will be kept.
    :param num_half_rows_to_keep: Number of half-rows to keep in predictor
        grids.  If None, all rows will be kept.
    :param num_half_columns_to_keep: Same but for columns.
    :param first_time_to_keep_unix_sec: First valid time to keep.  If None, all
        valid times will be kept.
    :param last_time_to_keep_unix_sec: Last valid time to keep.  If None, all
        valid times will be kept.
    :return: example_dict: See doc for `create_examples`.
    """

    # Check input args.
    if first_time_to_keep_unix_sec is None:
        first_time_to_keep_unix_sec = 0
    if last_time_to_keep_unix_sec is None:
        last_time_to_keep_unix_sec = LARGE_INTEGER

    error_checking.assert_is_boolean(metadata_only)
    error_checking.assert_is_integer(first_time_to_keep_unix_sec)
    error_checking.assert_is_integer(last_time_to_keep_unix_sec)
    error_checking.assert_is_geq(
        last_time_to_keep_unix_sec, first_time_to_keep_unix_sec)

    # Read file.
    dataset_object = netcdf_io.open_netcdf(netcdf_file_name)

    valid_times_unix_sec = numpy.array(
        dataset_object.variables[VALID_TIMES_KEY][:], dtype=int)
    row_indices = numpy.array(
        dataset_object.variables[ROW_INDICES_KEY][:], dtype=int)
    column_indices = numpy.array(
        dataset_object.variables[COLUMN_INDICES_KEY][:], dtype=int)

    predictor_names = netCDF4.chartostring(
        dataset_object.variables[PREDICTOR_NAMES_KEY][:]
    )

    predictor_names = [str(s) for s in predictor_names]
    if predictor_names_to_keep is None:
        predictor_names_to_keep = copy.deepcopy(predictor_names)

    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names_to_keep), num_dimensions=1)

    predictor_indices = numpy.array(
        [predictor_names.index(p) for p in predictor_names_to_keep],
        dtype=int)

    found_normalization_params = (
        FIRST_NORM_PARAM_KEY in dataset_object.variables or
        SECOND_NORM_PARAM_KEY in dataset_object.variables
    )

    if found_normalization_params:
        if hasattr(dataset_object, NORMALIZATION_TYPE_KEY):
            normalization_type_string = str(getattr(
                dataset_object, NORMALIZATION_TYPE_KEY
            ))
        else:
            normalization_type_string = ml_utils.Z_SCORE_STRING

        first_normalization_param_matrix = numpy.array(
            dataset_object.variables[FIRST_NORM_PARAM_KEY][:]
        )
        second_normalization_param_matrix = numpy.array(
            dataset_object.variables[SECOND_NORM_PARAM_KEY][:]
        )

        first_normalization_param_matrix = first_normalization_param_matrix[
            ..., predictor_indices]
        second_normalization_param_matrix = second_normalization_param_matrix[
            ..., predictor_indices]
    else:
        normalization_type_string = None
        first_normalization_param_matrix = None
        second_normalization_param_matrix = None

    if metadata_only:
        predictor_matrix = None
        target_matrix = None
    else:
        predictor_matrix = numpy.array(
            dataset_object.variables[PREDICTOR_MATRIX_KEY][:]
        )
        target_matrix = numpy.array(
            dataset_object.variables[TARGET_MATRIX_KEY][:]
        )

        predictor_matrix = predictor_matrix[..., predictor_indices]
        predictor_matrix = _shrink_predictor_grid(
            predictor_matrix=predictor_matrix,
            num_half_rows=num_half_rows_to_keep,
            num_half_columns=num_half_columns_to_keep)

    example_indices = numpy.where(numpy.logical_and(
        valid_times_unix_sec >= first_time_to_keep_unix_sec,
        valid_times_unix_sec <= last_time_to_keep_unix_sec
    ))[0]

    example_dict = {
        VALID_TIMES_KEY: valid_times_unix_sec[example_indices],
        ROW_INDICES_KEY: row_indices[example_indices],
        COLUMN_INDICES_KEY: column_indices[example_indices],
        PREDICTOR_NAMES_KEY: predictor_names_to_keep,
        PRESSURE_LEVEL_KEY: int(getattr(dataset_object, PRESSURE_LEVEL_KEY)),
        DILATION_DISTANCE_KEY: getattr(dataset_object, DILATION_DISTANCE_KEY),
        MASK_MATRIX_KEY:
            numpy.array(dataset_object.variables[MASK_MATRIX_KEY][:], dtype=int)
    }

    if found_normalization_params:
        example_dict.update({
            NORMALIZATION_TYPE_KEY: normalization_type_string,
            FIRST_NORM_PARAM_KEY:
                first_normalization_param_matrix[example_indices, ...],
            SECOND_NORM_PARAM_KEY:
                second_normalization_param_matrix[example_indices, ...]
        })

    if not metadata_only:
        example_dict.update({
            PREDICTOR_MATRIX_KEY:
                predictor_matrix[example_indices, ...].astype('float32'),
            TARGET_MATRIX_KEY:
                target_matrix[example_indices, ...].astype('float64')
        })

    dataset_object.close()
    return example_dict
