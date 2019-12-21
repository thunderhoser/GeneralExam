"""IO methods for predictions (gridded probs and deterministic labels)."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.ge_utils import neigh_evaluation

CLIMO_TIME_INTERVAL_SEC = 10800
ACCEPTED_HOURS_FOR_CLIMO = numpy.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=int)

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

    year_string = time_conversion.unix_sec_to_string(first_time_unix_sec, '%Y')

    prediction_file_name = '{0:s}/{1:s}/predictions_{2:s}-{3:s}.nc'.format(
        directory_name, year_string,
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


def file_name_to_times(prediction_file_name):
    """Parses time period from name of prediction file.

    :param prediction_file_name: Path to input file.
    :return: first_time_unix_sec: First time in file.
    :return: last_time_unix_sec: Last time in file.
    """

    error_checking.assert_is_string(prediction_file_name)

    pathless_file_name = os.path.split(prediction_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    time_period_string = extensionless_file_name.split('_')[-1]
    time_strings = time_period_string.split('-')

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        time_strings[0], FILE_NAME_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        time_strings[1], FILE_NAME_TIME_FORMAT
    )

    return first_time_unix_sec, last_time_unix_sec


def find_files_for_climo(
        directory_name, first_time_unix_sec, last_time_unix_sec,
        hours_to_keep=None, months_to_keep=None):
    """Finds files with gridded predictions to be used for climatology.

    T = number of time steps found

    :param directory_name: Name of directory.
    :param first_time_unix_sec: First time in period.
    :param last_time_unix_sec: Last time in period.
    :param hours_to_keep: 1-D numpy array of hours to be used for climo (in
        range 0...23).
    :param months_to_keep: 1-D numpy array of months to be used for climo (in
        range 1...12).
    :return: prediction_file_names: length-T list of file paths.
    :return: valid_times_unix_sec: length-T numpy array of valid times.
    :raises: ValueError: if any value in `hours_to_keep` is not in the list
        `ACCEPTED_HOURS_FOR_CLIMO`.
    """

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=CLIMO_TIME_INTERVAL_SEC, include_endpoint=True)

    if hours_to_keep is not None:
        climo_utils.check_hours(hours_to_keep)
        these_flags = numpy.array(
            [h in ACCEPTED_HOURS_FOR_CLIMO for h in hours_to_keep], dtype=bool
        )

        if not numpy.all(these_flags):
            error_string = (
                '\n{0:s}\nAt least one hour (listed above) is not in the list '
                'of accepted hours (listed below).\n{1:s}'
            ).format(str(hours_to_keep), str(ACCEPTED_HOURS_FOR_CLIMO))

            raise ValueError(error_string)

        indices_to_keep = climo_utils.filter_by_hour(
            all_times_unix_sec=valid_times_unix_sec,
            hours_to_keep=hours_to_keep)

        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]

    if months_to_keep is not None:
        climo_utils.check_months(months_to_keep)

        indices_to_keep = climo_utils.filter_by_month(
            all_times_unix_sec=valid_times_unix_sec,
            months_to_keep=months_to_keep)

        valid_times_unix_sec = valid_times_unix_sec[indices_to_keep]

    prediction_file_names = [
        find_file(
            directory_name=directory_name,
            first_time_unix_sec=t, last_time_unix_sec=t,
            raise_error_if_missing=True)
        for t in valid_times_unix_sec
    ]

    return prediction_file_names, valid_times_unix_sec


def write_probabilities(
        netcdf_file_name, class_probability_matrix, valid_times_unix_sec,
        model_file_name, used_isotonic, target_matrix=None,
        target_dilation_distance_metres=None):
    """Writes gridded probabilities to NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    If `target_matrix is None`, this method will NOT write target values.

    :param netcdf_file_name: Path to output file.
    :param class_probability_matrix: T-by-M-by-N-by-3 numpy array of gridded
        class probabilities.
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param model_file_name: Path to model that generated the predictions
        (readable by `cnn.read_model`).
    :param used_isotonic: Boolean flag.  True means that isotonic regression was
        used for probability calibration.
    :param target_matrix: T-by-M-by-N numpy array of true labels (all in range
        0...2).
    :param target_dilation_distance_metres: Dilation distance for
        `target_matrix`.
    """

    error_checking.assert_is_numpy_array(class_probability_matrix)
    class_probability_matrix = _fill_probabilities(class_probability_matrix)

    # Check input args.
    neigh_evaluation.check_gridded_predictions(
        prediction_matrix=class_probability_matrix, expect_probs=True)

    if target_matrix is not None:
        neigh_evaluation.check_gridded_predictions(
            prediction_matrix=target_matrix, expect_probs=False)

        error_checking.assert_is_numpy_array(
            target_matrix,
            exact_dimensions=numpy.array(
                class_probability_matrix.shape[:-1], dtype=int
            )
        )

    num_times = class_probability_matrix.shape[0]

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

    if target_matrix is not None:
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

    if target_matrix is not None:
        dataset_object.createVariable(
            TARGET_MATRIX_KEY, datatype=numpy.int32, dimensions=(
                TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
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

    these_expected_dim = numpy.array(
        dataset_object.variables[CLASS_PROBABILITIES_KEY][:].shape[:-1],
        dtype=int
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
    prediction_dict['valid_times_unix_sec']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['used_isotonic']: Same.
    prediction_dict['target_matrix']: Same.
    prediction_dict['target_dilation_distance_metres']: Same.

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
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:], dtype=int
        ),
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        USED_ISOTONIC_KEY: bool(getattr(dataset_object, USED_ISOTONIC_KEY)),
    }

    if TARGET_MATRIX_KEY in dataset_object.variables:
        prediction_dict.update({
            TARGET_MATRIX_KEY: numpy.array(
                dataset_object.variables[TARGET_MATRIX_KEY][:], dtype=int
            ),
            DILATION_DISTANCE_KEY:
                float(getattr(dataset_object, DILATION_DISTANCE_KEY))
        })

    if not read_deterministic:
        dataset_object.close()
        return prediction_dict

    prob_threshold_by_class = numpy.array(
        dataset_object.variables[THRESHOLDS_KEY][:]
    )
    prob_threshold_by_class[prob_threshold_by_class < 0.] = numpy.nan

    prediction_dict.update({
        PREDICTED_LABELS_KEY: numpy.array(
            dataset_object.variables[PREDICTED_LABELS_KEY][:], dtype=int
        ),
        THRESHOLDS_KEY: prob_threshold_by_class,
        MIN_REGION_LENGTH_KEY:
            float(getattr(dataset_object, MIN_REGION_LENGTH_KEY)),
        REGION_BUFFER_DISTANCE_KEY:
            float(getattr(dataset_object, REGION_BUFFER_DISTANCE_KEY))
    })

    dataset_object.close()
    return prediction_dict
