"""IO methods for predictions (gridded probs and deterministic labels)."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.machine_learning import neigh_evaluation

NUM_CLASSES = 3
FILE_NAME_TIME_FORMAT = '%Y%m%d%H'

CLASS_PROBABILITIES_KEY = 'class_probability_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
MODEL_FILE_KEY = 'model_file_name'
DILATION_DISTANCE_KEY = 'target_dilation_distance_metres'
USED_ISOTONIC_KEY = 'used_isotonic'

PREDICTED_LABELS_KEY = 'predicted_label_matrix'
THRESHOLDS_KEY = 'prob_threshold_by_class'
MIN_REGION_LENGTH_KEY = 'min_region_length_metres'
REGION_BUFFER_DISTANCE_KEY = 'region_buffer_distance_metres'

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
CLASS_DIMENSION_KEY = 'class'


def _fill_probabilities(class_probability_matrix):
    """Fills missing class probabilities.

    For any grid cell with missing probabilities, this method assumes that there
    is no front.

    :param class_probability_matrix: numpy array of class probabilities.  The
        last axis should have length 3.  class_probability_matrix[..., k] should
        contain probabilities for the [k]th class.
    :return: class_probability_matrix: Same but with no missing values.
    """

    class_probability_matrix[..., front_utils.NO_FRONT_ENUM][
        numpy.isnan(class_probability_matrix[..., front_utils.NO_FRONT_ENUM])
    ] = 1.

    class_probability_matrix[numpy.isnan(class_probability_matrix)] = 0.

    return class_probability_matrix


def find_file(directory_name, first_time_unix_sec, last_time_unix_sec,
              raise_error_if_missing=False):
    """Finds file with gridded predictions (probs and maybe deterministic).

    :param directory_name: Name of directory.
    :param first_time_unix_sec: First time in file.
    :param last_time_unix_sec: Last time in file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: prediction_file_name: Path to prediction file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/predictions_{1:s}-{2:s}.nc'.format(
        directory_name,
        time_conversion.unix_sec_to_string(
            first_time_unix_sec, FILE_NAME_TIME_FORMAT),
        time_conversion.unix_sec_to_string(
            last_time_unix_sec, FILE_NAME_TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_probabilities(
        netcdf_file_name, class_probability_matrix, target_matrix,
        valid_times_unix_sec, model_file_name, target_dilation_distance_metres,
        used_isotonic):
    """Writes gridded probabilities to NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param class_probability_matrix: T-by-M-by-N-by-3 numpy array of gridded
        class probabilities.
    :param target_matrix: T-by-M-by-N numpy array of true labels (all in range
        0...2).
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param model_file_name: Path to model that generated the predictions
        (readable by `cnn.read_model`).
    :param target_dilation_distance_metres: Dilation distance for
        `target_matrix`.
    :param used_isotonic: Boolean flag.  True means that isotonic regression was
        used for probability calibration.
    """

    error_checking.assert_is_numpy_array(class_probability_matrix)
    class_probability_matrix = _fill_probabilities(class_probability_matrix)

    # Check input args.
    neigh_evaluation.check_gridded_predictions(
        prediction_matrix=class_probability_matrix, expect_probs=True)
    neigh_evaluation.check_gridded_predictions(
        prediction_matrix=target_matrix, expect_probs=False)

    error_checking.assert_is_numpy_array(
        target_matrix,
        exact_dimensions=numpy.array(
            class_probability_matrix.shape[:-1], dtype=int
        )
    )

    num_times = target_matrix.shape[0]

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec,
        exact_dimensions=numpy.array([num_times], dtype=int)
    )

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes.
    dataset_object.setncattr(MODEL_FILE_KEY, str(model_file_name))
    dataset_object.setncattr(USED_ISOTONIC_KEY, int(used_isotonic))
    dataset_object.setncattr(
        DILATION_DISTANCE_KEY, float(target_dilation_distance_metres)
    )

    # Set dimensions.
    dataset_object.createDimension(TIME_DIMENSION_KEY, num_times)
    dataset_object.createDimension(
        ROW_DIMENSION_KEY, class_probability_matrix.shape[1]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, class_probability_matrix.shape[2]
    )
    dataset_object.createDimension(
        CLASS_DIMENSION_KEY, class_probability_matrix.shape[3]
    )

    # Add variables.
    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=TIME_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = target_matrix

    dataset_object.createVariable(
        CLASS_PROBABILITIES_KEY, datatype=numpy.float32,
        dimensions=(TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
                    CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[CLASS_PROBABILITIES_KEY][:] = (
        class_probability_matrix
    )

    dataset_object.close()


def append_deterministic_labels(
        probability_file_name, predicted_label_matrix, prob_threshold_by_class,
        min_region_length_metres, region_buffer_distance_metres):
    """Appends deterministic labels to NetCDF file with probabilities.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param probability_file_name: Path to output file (existing file created by
        `write_probabilities`).
    :param predicted_label_matrix: T-by-M-by-N numpy array of predicted
        deterministic labels (all in range 0...2).
    :param prob_threshold_by_class: length-3 numpy array of class-probability
        thresholds.  If `neigh_evaluation.determinize_predictions_1threshold`
        was used to determinize predictions, the last 2 elements should be NaN.
        If `neigh_evaluation.determinize_predictions_2thresholds` was used, the
        first element should be NaN.
    :param min_region_length_metres: Minimum length used in
        `neigh_evaluation.remove_small_regions_one_time`.
    :param region_buffer_distance_metres: Buffer distance used in
        `neigh_evaluation.remove_small_regions_one_time`.
    """

    # Check input args.
    neigh_evaluation.check_gridded_predictions(
        prediction_matrix=predicted_label_matrix, expect_probs=False)

    error_checking.assert_is_numpy_array(
        prob_threshold_by_class,
        exact_dimensions=numpy.array([NUM_CLASSES], dtype=int)
    )

    error_checking.assert_is_geq_numpy_array(
        prob_threshold_by_class, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(
        prob_threshold_by_class, 1., allow_nan=True)

    if numpy.isnan(prob_threshold_by_class[0]):
        assert not numpy.any(numpy.isnan(prob_threshold_by_class[1:]))
    else:
        assert numpy.all(numpy.isnan(prob_threshold_by_class[1:]))

    prob_threshold_by_class[numpy.isnan(prob_threshold_by_class)] = -1.

    error_checking.assert_is_geq(min_region_length_metres, 0.)
    error_checking.assert_is_geq(region_buffer_distance_metres, 0.)

    # Open file and finish checking input args.
    dataset_object = netCDF4.Dataset(
        probability_file_name, 'a', format='NETCDF3_64BIT_OFFSET')

    # TODO(thunderhoser): This might not work.
    these_expected_dim = numpy.array(
        dataset_object.variables[TARGET_MATRIX_KEY][:].shape, dtype=int
    )
    error_checking.assert_is_numpy_array(
        predicted_label_matrix, exact_dimensions=these_expected_dim)

    # Append deterministic labels to file.
    dataset_object.setncattr(
        MIN_REGION_LENGTH_KEY, float(min_region_length_metres)
    )
    dataset_object.setncattr(
        REGION_BUFFER_DISTANCE_KEY, float(region_buffer_distance_metres)
    )

    dataset_object.createVariable(
        THRESHOLDS_KEY, datatype=numpy.float32, dimensions=CLASS_DIMENSION_KEY
    )
    dataset_object.variables[THRESHOLDS_KEY][:] = prob_threshold_by_class

    dataset_object.createVariable(
        PREDICTED_LABELS_KEY, datatype=numpy.int32,
        dimensions=(TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[PREDICTED_LABELS_KEY][:] = predicted_label_matrix

    dataset_object.close()


def read_file(netcdf_file_name, read_deterministic=False):
    """Reads probabilities (and maybe deterministic labels) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :param read_deterministic: Boolean flag.  If True, will read deterministic
        labels as well as probabilities.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['class_probability_matrix']: See doc for
        `write_probabilities`.
    prediction_dict['target_matrix']: Same.
    prediction_dict['valid_times_unix_sec']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['target_dilation_distance_metres']: Same.
    prediction_dict['used_isotonic']: Same.

    If `read_deterministic == True`, will also have the following keys.

    prediction_dict['predicted_label_matrix']: See doc for
        `append_deterministic_labels`.
    prediction_dict['prob_threshold_by_class']: Same.
    prediction_dict['min_region_length_metres']: Same.
    prediction_dict['region_buffer_distance_metres']: Same.
    """

    error_checking.assert_is_boolean(read_deterministic)
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        CLASS_PROBABILITIES_KEY: numpy.array(
            dataset_object.variables[CLASS_PROBABILITIES_KEY][:]
        ),
        TARGET_MATRIX_KEY: numpy.array(
            dataset_object.variables[TARGET_MATRIX_KEY][:], dtype=int
        ),
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:], dtype=int
        ),
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        USED_ISOTONIC_KEY: bool(getattr(dataset_object, USED_ISOTONIC_KEY)),
        DILATION_DISTANCE_KEY:
            float(getattr(dataset_object, DILATION_DISTANCE_KEY))
    }

    if not read_deterministic:
        dataset_object.close()
        return prediction_dict

    prediction_dict.update({
        PREDICTED_LABELS_KEY: numpy.array(
            dataset_object.variables[PREDICTED_LABELS_KEY][:], dtype=int
        ),
        THRESHOLDS_KEY: numpy.array(
            dataset_object.variables[THRESHOLDS_KEY][:]
        ),
        MIN_REGION_LENGTH_KEY:
            float(getattr(dataset_object, MIN_REGION_LENGTH_KEY)),
        REGION_BUFFER_DISTANCE_KEY:
            float(getattr(dataset_object, REGION_BUFFER_DISTANCE_KEY))
    })

    dataset_object.close()
    return prediction_dict
