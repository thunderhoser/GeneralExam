"""IO for ungridded predictions."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import evaluation_utils

TIME_FORMAT = '%Y%m%d%H'

CLASS_PROBABILITIES_KEY = 'class_probability_matrix'
OBSERVED_LABELS_KEY = 'observed_labels'
EXAMPLE_IDS_KEY = 'example_id_strings'
EXAMPLE_DIR_KEY = 'top_example_dir_name'
MODEL_FILE_KEY = 'model_file_name'
USED_ISOTONIC_KEY = 'used_isotonic'

EXAMPLE_DIMENSION_KEY = 'example'
CLASS_DIMENSION_KEY = 'class'
ID_CHAR_DIMENSION_KEY = 'example_id_char'


def find_file(directory_name, first_time_unix_sec, last_time_unix_sec,
              raise_error_if_missing=False):
    """Finds file with ungridded predictions.

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

    prediction_file_name = (
        '{0:s}/{1:s}/ungridded_predictions_{2:s}-{3:s}.nc'
    ).format(
        directory_name, year_string,
        time_conversion.unix_sec_to_string(first_time_unix_sec, TIME_FORMAT),
        time_conversion.unix_sec_to_string(last_time_unix_sec, TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name)
        raise ValueError(error_string)

    return prediction_file_name


def write_file(
        netcdf_file_name, class_probability_matrix, observed_labels,
        example_id_strings, top_example_dir_name, model_file_name,
        used_isotonic):
    """Writes ungridded predictions to NetCDF file.

    E = number of examples
    K = number of classes

    :param netcdf_file_name: Path to output file.
    :param class_probability_matrix: E-by-K numpy array of class probabilities.
    :param observed_labels: length-E numpy array of observed classes (integers
        from 0...[K - 1]).
    :param example_id_strings: length-E list of example IDs.
    :param top_example_dir_name: Name of top-level directory with examples used
        to generate predictions.  Files therein should be findable by
        `learning_examples_io.find_file` and readable by
        `learning_examples_io.read_file`.
    :param model_file_name: Path to model used to generate predictions (readable
        by `cnn.read_model`).
    :param used_isotonic: Boolean flag, indicating whether or not isotonic
        regression was used to calibrate probabilities in
        `class_probability_matrix`.
    """

    evaluation_utils.check_evaluation_pairs(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels)

    num_examples = len(observed_labels)
    these_expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings),
        exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_string_list(top_example_dir_name)
    error_checking.assert_is_string_list(model_file_name)
    error_checking.assert_is_boolean(used_isotonic)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes.
    dataset_object.setncattr(EXAMPLE_DIR_KEY, str(top_example_dir_name))
    dataset_object.setncattr(MODEL_FILE_KEY, str(model_file_name))
    dataset_object.setncattr(USED_ISOTONIC_KEY, int(used_isotonic))

    # Set dimensions.
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        CLASS_DIMENSION_KEY, class_probability_matrix.shape[1]
    )

    num_id_characters = max([
        len(s) for s in example_id_strings
    ])
    dataset_object.createDimension(ID_CHAR_DIMENSION_KEY, num_id_characters)

    # Add variables.
    dataset_object.createVariable(
        CLASS_PROBABILITIES_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, CLASS_DIMENSION_KEY)
    )
    dataset_object.variables[CLASS_PROBABILITIES_KEY][:] = (
        class_probability_matrix
    )

    dataset_object.createVariable(
        OBSERVED_LABELS_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[OBSERVED_LABELS_KEY][:] = observed_labels

    this_string_type = 'S{0:d}'.format(num_id_characters)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_type
    ))

    dataset_object.createVariable(
        EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, ID_CHAR_DIMENSION_KEY)
    )
    dataset_object.variables[EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array)

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads ungridded predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['class_probability_matrix']: See doc for `write_file`.
    prediction_dict['observed_labels']: Same.
    prediction_dict['example_id_strings']: Same.
    prediction_dict['top_example_dir_name']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['used_isotonic']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        CLASS_PROBABILITIES_KEY: numpy.array(
            dataset_object.variables[CLASS_PROBABILITIES_KEY][:]
        ),
        OBSERVED_LABELS_KEY: numpy.array(
            dataset_object.variables[OBSERVED_LABELS_KEY][:], dtype=int
        ),
        EXAMPLE_IDS_KEY: [
            str(s) for s in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        EXAMPLE_DIR_KEY: str(getattr(dataset_object, EXAMPLE_DIR_KEY)),
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        USED_ISOTONIC_KEY: bool(getattr(dataset_object, USED_ISOTONIC_KEY))
    }

    dataset_object.close()
    return prediction_dict
