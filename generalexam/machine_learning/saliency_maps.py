"""IO methods for saliency maps created by GeneralExam models."""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as gg_saliency_maps

COMPONENT_TYPE_KEY = gg_saliency_maps.COMPONENT_TYPE_KEY
TARGET_CLASS_KEY = gg_saliency_maps.TARGET_CLASS_KEY
LAYER_NAME_KEY = gg_saliency_maps.LAYER_NAME_KEY
IDEAL_ACTIVATION_KEY = gg_saliency_maps.IDEAL_ACTIVATION_KEY
NEURON_INDICES_KEY = gg_saliency_maps.NEURON_INDICES_KEY
CHANNEL_INDEX_KEY = gg_saliency_maps.CHANNEL_INDEX_KEY

PREDICTOR_MATRIX_KEY = 'denorm_predictor_matrix'
SALIENCY_MATRIX_KEY = 'saliency_matrix'
EXAMPLE_IDS_KEY = 'example_id_strings'
MODEL_FILE_KEY = 'model_file_name'

STANDARD_FILE_KEYS = [
    PREDICTOR_MATRIX_KEY, SALIENCY_MATRIX_KEY, EXAMPLE_IDS_KEY, MODEL_FILE_KEY,
    COMPONENT_TYPE_KEY, TARGET_CLASS_KEY, LAYER_NAME_KEY, IDEAL_ACTIVATION_KEY,
    NEURON_INDICES_KEY, CHANNEL_INDEX_KEY
]

MEAN_PREDICTOR_MATRIX_KEY = 'mean_denorm_predictor_matrix'
MEAN_SALIENCY_MATRIX_KEY = 'mean_saliency_matrix'
NON_PMM_FILE_KEY = 'non_pmm_file_name'
PMM_MAX_PERCENTILE_KEY = 'pmm_max_percentile_leve'

PMM_FILE_KEYS = [
    MEAN_PREDICTOR_MATRIX_KEY, MEAN_SALIENCY_MATRIX_KEY, MODEL_FILE_KEY,
    NON_PMM_FILE_KEY, PMM_MAX_PERCENTILE_KEY
]


def _check_in_and_out_matrices(
        predictor_matrix, num_examples=None, saliency_matrix=None):
    """Error-checks input and output matrices.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictors.
    :param num_examples: E in the above discussion.  If you don't know the
        number of examples, leave this as None.
    :param saliency_matrix: E-by-M-by-N-by-C numpy array of saliency values.
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

    if saliency_matrix is None:
        return

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix,
        exact_dimensions=numpy.array(predictor_matrix.shape, dtype=int)
    )


def write_pmm_file(
        pickle_file_name, mean_denorm_predictor_matrix, mean_saliency_matrix,
        model_file_name, non_pmm_file_name, pmm_max_percentile_level):
    """Writes composite saliency map to Pickle file.

    The composite should be created by probability-matched means (PMM).

    :param pickle_file_name: Path to output file.
    :param mean_denorm_predictor_matrix: See doc for
        `_check_in_and_out_matrices`.
    :param mean_saliency_matrix: Same.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param non_pmm_file_name: Path to standard saliency file (with
        non-composited saliency maps).
    :param pmm_max_percentile_level: Max percentile level for PMM.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(non_pmm_file_name)
    error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
    error_checking.assert_is_leq(pmm_max_percentile_level, 100.)

    _check_in_and_out_matrices(
        predictor_matrix=mean_denorm_predictor_matrix, num_examples=None,
        saliency_matrix=mean_saliency_matrix)

    mean_saliency_dict = {
        MEAN_PREDICTOR_MATRIX_KEY: mean_denorm_predictor_matrix,
        MEAN_SALIENCY_MATRIX_KEY: mean_saliency_matrix,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def write_standard_file(
        pickle_file_name, denorm_predictor_matrix, saliency_matrix,
        example_id_strings, model_file_name, metadata_dict):
    """Writes saliency maps to Pickle file.

    E = number of examples

    :param pickle_file_name: Path to output file.
    :param denorm_predictor_matrix: See doc for `_check_in_and_out_matrices`.
    :param saliency_matrix: Same.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param metadata_dict: Dictionary created by `saliency_maps.check_metadata`
        in GewitterGefahr.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    num_examples = len(example_id_strings)

    _check_in_and_out_matrices(
        predictor_matrix=denorm_predictor_matrix, num_examples=num_examples,
        saliency_matrix=saliency_matrix)

    saliency_dict = {
        PREDICTOR_MATRIX_KEY: denorm_predictor_matrix,
        SALIENCY_MATRIX_KEY: saliency_matrix,
        EXAMPLE_IDS_KEY: example_id_strings,
        MODEL_FILE_KEY: model_file_name,
        COMPONENT_TYPE_KEY: metadata_dict[COMPONENT_TYPE_KEY],
        TARGET_CLASS_KEY: metadata_dict[TARGET_CLASS_KEY],
        LAYER_NAME_KEY: metadata_dict[LAYER_NAME_KEY],
        IDEAL_ACTIVATION_KEY: metadata_dict[IDEAL_ACTIVATION_KEY],
        NEURON_INDICES_KEY: metadata_dict[NEURON_INDICES_KEY],
        CHANNEL_INDEX_KEY: metadata_dict[CHANNEL_INDEX_KEY]
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(saliency_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads composite or non-composite saliency maps from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: saliency_dict: Has the following keys if not a composite...
    saliency_dict['denorm_predictor_matrix']: See doc for
        `write_standard_file`.
    saliency_dict['saliency_matrix']: Same.
    saliency_dict['example_id_strings']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['component_type_string']: Same.
    saliency_dict['target_class']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['ideal_activation']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['channel_index']: Same.

    ...or the following keys if composite...

    saliency_dict['mean_denorm_predictor_matrix']: See doc for
        `write_pmm_file`.
    saliency_dict['mean_saliency_matrix']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['non_pmm_file_name']: Same.
    saliency_dict['pmm_max_percentile_level']: Same.

    :return: pmm_flag: Boolean flag.  True if `saliency_dict` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    saliency_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_PREDICTOR_MATRIX_KEY in saliency_dict

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(saliency_dict.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(saliency_dict.keys())
        )

    if len(missing_keys) == 0:
        return saliency_dict, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
