"""Helper methods for Grad-CAM (gradient-weighted class-activation maps)."""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

PREDICTOR_MATRIX_KEY = 'denorm_predictor_matrix'
ACTIVN_MATRIX_KEY = 'class_activn_matrix'
GUIDED_ACTIVN_MATRIX_KEY = 'guided_class_activn_matrix'
EXAMPLE_IDS_KEY = 'example_id_strings'
MODEL_FILE_KEY = 'model_file_name'
TARGET_CLASS_KEY = 'target_class'
TARGET_LAYER_KEY = 'target_layer_name'

STANDARD_FILE_KEYS = [
    PREDICTOR_MATRIX_KEY, ACTIVN_MATRIX_KEY, GUIDED_ACTIVN_MATRIX_KEY,
    EXAMPLE_IDS_KEY, MODEL_FILE_KEY, TARGET_CLASS_KEY, TARGET_LAYER_KEY
]

MEAN_PREDICTOR_MATRIX_KEY = 'mean_denorm_predictor_matrix'
MEAN_ACTIVN_MATRIX_KEY = 'mean_activn_matrix'
MEAN_GUIDED_ACTIVN_MATRIX_KEY = 'mean_guided_activn_matrix'
NON_PMM_FILE_KEY = 'non_pmm_file_name'
PMM_MAX_PERCENTILE_KEY = 'pmm_max_percentile_level'

PMM_FILE_KEYS = [
    MEAN_PREDICTOR_MATRIX_KEY, MEAN_ACTIVN_MATRIX_KEY,
    MEAN_GUIDED_ACTIVN_MATRIX_KEY, MODEL_FILE_KEY, NON_PMM_FILE_KEY,
    PMM_MAX_PERCENTILE_KEY
]


def _check_in_and_out_matrices(
        predictor_matrix, num_examples=None, class_activn_matrix=None,
        guided_class_activn_matrix=None):
    """Error-checks input and output matrices.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictors.
    :param num_examples: E in the above discussion.  If you don't know the
        number of examples, leave this as None.
    :param class_activn_matrix: E-by-M-by-N numpy array of class activations.
    :param guided_class_activn_matrix: E-by-M-by-N numpy array of guided class
        activations.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)

    if num_examples is not None:
        these_expected_dim = numpy.array(
            (num_examples,) + predictor_matrix.shape[1:], dtype=int
        )
        error_checking.assert_is_numpy_array(
            predictor_matrix, exact_dimensions=these_expected_dim)

    if class_activn_matrix is None:
        return

    error_checking.assert_is_numpy_array_without_nan(class_activn_matrix)
    error_checking.assert_is_numpy_array(
        class_activn_matrix,
        exact_dimensions=numpy.array(predictor_matrix.shape[:-1], dtype=int)
    )

    error_checking.assert_is_numpy_array_without_nan(guided_class_activn_matrix)
    error_checking.assert_is_numpy_array(
        guided_class_activn_matrix,
        exact_dimensions=numpy.array(predictor_matrix.shape, dtype=int)
    )


def write_standard_file(
        pickle_file_name, denorm_predictor_matrix, class_activn_matrix,
        guided_class_activn_matrix, example_id_strings, model_file_name,
        target_class, target_layer_name):
    """Writes class-activation maps to Pickle file.

    E = number of examples

    :param pickle_file_name: Path to output file.
    :param denorm_predictor_matrix: See doc for `_check_in_and_out_matrices`.
    :param class_activn_matrix: Same.
    :param guided_class_activn_matrix: Same.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to model that created class-activation maps
        (readable by `cnn.read_model`).
    :param target_class: Target class.  Must be an integer from 0...(K - 1),
        where K = number of classes.
    :param target_layer_name: Name of target layer.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_integer(target_class)
    error_checking.assert_is_geq(target_class, 0)
    error_checking.assert_is_string(target_layer_name)

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    num_examples = len(example_id_strings)

    _check_in_and_out_matrices(
        predictor_matrix=denorm_predictor_matrix, num_examples=num_examples,
        class_activn_matrix=class_activn_matrix,
        guided_class_activn_matrix=guided_class_activn_matrix)

    gradcam_dict = {
        PREDICTOR_MATRIX_KEY: denorm_predictor_matrix,
        ACTIVN_MATRIX_KEY: class_activn_matrix,
        GUIDED_ACTIVN_MATRIX_KEY: guided_class_activn_matrix,
        EXAMPLE_IDS_KEY: example_id_strings,
        MODEL_FILE_KEY: model_file_name,
        TARGET_CLASS_KEY: target_class,
        TARGET_LAYER_KEY: target_layer_name
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def write_pmm_file(
        pickle_file_name, mean_denorm_predictor_matrix, mean_activn_matrix,
        mean_guided_activn_matrix, model_file_name, non_pmm_file_name,
        pmm_max_percentile_level):
    """Writes composite of class-activation maps to Pickle file.

    :param pickle_file_name: Path to output file.
    :param mean_denorm_predictor_matrix: See doc for
        `_check_in_and_out_matrices`.
    :param mean_activn_matrix: Same.
    :param mean_guided_activn_matrix: Same.
    :param model_file_name: Path to model that created class-activation maps
        (readable by `cnn.read_model`).
    :param non_pmm_file_name: Path to standard Grad-CAM file (with
        non-composited CAMs).
    :param pmm_max_percentile_level: Max percentile level for PMM.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(non_pmm_file_name)
    error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
    error_checking.assert_is_leq(pmm_max_percentile_level, 100.)

    _check_in_and_out_matrices(
        predictor_matrix=mean_denorm_predictor_matrix, num_examples=None,
        class_activn_matrix=mean_activn_matrix,
        guided_class_activn_matrix=mean_guided_activn_matrix)

    mean_gradcam_dict = {
        MEAN_PREDICTOR_MATRIX_KEY: mean_denorm_predictor_matrix,
        MEAN_ACTIVN_MATRIX_KEY: mean_activn_matrix,
        MEAN_GUIDED_ACTIVN_MATRIX_KEY: mean_guided_activn_matrix,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_gradcam_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads standard or composited class-activation maps from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: gradcam_dict: Has the following keys if not a composite...
    gradcam_dict['denorm_predictor_matrix']: See doc for
        `write_standard_file`.
    gradcam_dict['class_activn_matrix']: Same.
    gradcam_dict['guided_class_activn_matrix']: Same.
    gradcam_dict['example_id_strings']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['target_class']: Same.
    gradcam_dict['target_layer_name']: Same.

    ...or the following keys if composite...

    gradcam_dict['mean_denorm_predictor_matrix']: See doc for
        `write_pmm_file`.
    gradcam_dict['mean_activn_matrix']: Same.
    gradcam_dict['mean_guided_activn_matrix']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['non_pmm_file_name']: Same.
    gradcam_dict['pmm_max_percentile_level']: Same.

    :return: pmm_flag: Boolean flag.  True if `gradcam_dict` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    gradcam_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_PREDICTOR_MATRIX_KEY in gradcam_dict

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(gradcam_dict.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(gradcam_dict.keys())
        )

    if len(missing_keys) == 0:
        return gradcam_dict, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
