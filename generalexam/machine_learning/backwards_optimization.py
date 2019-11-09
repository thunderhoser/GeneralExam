"""Helper methods for backwards optimization."""

import pickle
import numpy
from gewittergefahr.deep_learning import \
    backwards_optimization as gg_backwards_opt
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

INPUT_MATRIX_KEY = 'denorm_input_matrix'
OUTPUT_MATRIX_KEY = 'denorm_output_matrix'
INITIAL_ACTIVATIONS_KEY = gg_backwards_opt.INITIAL_ACTIVATIONS_KEY
FINAL_ACTIVATIONS_KEY = gg_backwards_opt.FINAL_ACTIVATIONS_KEY
EXAMPLE_IDS_KEY = 'example_id_strings'
MODEL_FILE_KEY = 'model_file_name'

NUM_ITERATIONS_KEY = gg_backwards_opt.NUM_ITERATIONS_KEY
LEARNING_RATE_KEY = gg_backwards_opt.LEARNING_RATE_KEY
L2_WEIGHT_KEY = gg_backwards_opt.L2_WEIGHT_KEY
COMPONENT_TYPE_KEY = gg_backwards_opt.COMPONENT_TYPE_KEY
TARGET_CLASS_KEY = gg_backwards_opt.TARGET_CLASS_KEY
LAYER_NAME_KEY = gg_backwards_opt.LAYER_NAME_KEY
IDEAL_ACTIVATION_KEY = gg_backwards_opt.IDEAL_ACTIVATION_KEY
NEURON_INDICES_KEY = gg_backwards_opt.NEURON_INDICES_KEY
CHANNEL_INDEX_KEY = gg_backwards_opt.CHANNEL_INDEX_KEY

STANDARD_FILE_KEYS = [
    INPUT_MATRIX_KEY, OUTPUT_MATRIX_KEY, INITIAL_ACTIVATIONS_KEY,
    FINAL_ACTIVATIONS_KEY, EXAMPLE_IDS_KEY, MODEL_FILE_KEY,
    NUM_ITERATIONS_KEY, LEARNING_RATE_KEY, L2_WEIGHT_KEY,
    COMPONENT_TYPE_KEY, TARGET_CLASS_KEY, LAYER_NAME_KEY,
    IDEAL_ACTIVATION_KEY, NEURON_INDICES_KEY, CHANNEL_INDEX_KEY
]

MEAN_INPUT_MATRIX_KEY = 'mean_denorm_input_matrix'
MEAN_OUTPUT_MATRIX_KEY = 'mean_denorm_output_matrix'
MEAN_INITIAL_ACTIVATION_KEY = 'mean_initial_activation'
MEAN_FINAL_ACTIVATION_KEY = 'mean_final_activation'
NON_PMM_FILE_KEY = 'non_pmm_file_name'
PMM_MAX_PERCENTILE_KEY = 'pmm_max_percentile_level'

PMM_FILE_KEYS = [
    MEAN_INPUT_MATRIX_KEY, MEAN_OUTPUT_MATRIX_KEY, MEAN_INITIAL_ACTIVATION_KEY,
    MEAN_FINAL_ACTIVATION_KEY, MODEL_FILE_KEY, NON_PMM_FILE_KEY,
    PMM_MAX_PERCENTILE_KEY
]


def _check_in_and_out_matrices(input_matrix, output_matrix, num_examples=None):
    """Error-checks input and output matrices.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param input_matrix: E-by-M-by-N-by-C numpy array of input values
        (predictors before optimization).
    :param output_matrix: Same but after optimization.
    :param num_examples: E in the above discussion.  If you don't know the
        number of examples, leave this as None.
    """

    error_checking.assert_is_numpy_array_without_nan(input_matrix)
    num_dimensions = len(input_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 3)
    error_checking.assert_is_leq(num_dimensions, 4)

    if num_examples is not None:
        these_expected_dim = numpy.array(
            (num_examples,) + input_matrix.shape[1:], dtype=int
        )
        error_checking.assert_is_numpy_array(
            input_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(output_matrix)
    error_checking.assert_is_numpy_array(
        output_matrix,
        exact_dimensions=numpy.array(input_matrix.shape, dtype=int)
    )


def write_pmm_file(
        pickle_file_name, mean_denorm_input_matrix, mean_denorm_output_matrix,
        mean_initial_activation, mean_final_activation, model_file_name,
        non_pmm_file_name, pmm_max_percentile_level):
    """Writes composite of pre- and post-optimized examples to Pickle file.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param pickle_file_name: Path to output file.
    :param mean_denorm_input_matrix: M-by-N-by-C numpy array of denormalized
        input values (predictors before optimization).
    :param mean_denorm_output_matrix: Same but after optimization.
    :param mean_initial_activation: Mean model activation before optimization.
    :param mean_final_activation: Mean model activation after optimization.
    :param model_file_name: Path to model that performed backwards optimization
        (readable by `cnn.read_model`).
    :param non_pmm_file_name: Path to standard backwards-optimization file (with
        non-composited results).
    :param pmm_max_percentile_level: Max percentile level for PMM.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string(non_pmm_file_name)
    error_checking.assert_is_geq(pmm_max_percentile_level, 90.)
    error_checking.assert_is_leq(pmm_max_percentile_level, 100.)

    _check_in_and_out_matrices(
        input_matrix=mean_denorm_input_matrix,
        output_matrix=mean_denorm_output_matrix, num_examples=None)

    error_checking.assert_is_not_nan(mean_initial_activation)
    error_checking.assert_is_not_nan(mean_final_activation)

    mean_bwo_dictionary = {
        MEAN_INPUT_MATRIX_KEY: mean_denorm_input_matrix,
        MEAN_OUTPUT_MATRIX_KEY: mean_denorm_output_matrix,
        MEAN_INITIAL_ACTIVATION_KEY: mean_initial_activation,
        MEAN_FINAL_ACTIVATION_KEY: mean_final_activation,
        MODEL_FILE_KEY: model_file_name,
        NON_PMM_FILE_KEY: non_pmm_file_name,
        PMM_MAX_PERCENTILE_KEY: pmm_max_percentile_level
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_bwo_dictionary, pickle_file_handle)
    pickle_file_handle.close()


def write_standard_file(
        pickle_file_name, denorm_input_matrix, denorm_output_matrix,
        initial_activations, final_activations, example_id_strings,
        model_file_name, metadata_dict):
    """Writes pre- and post-optimized examples to Pickle file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (predictors)

    :param pickle_file_name: Path to output file.
    :param denorm_input_matrix: E-by-M-by-N-by-C numpy array of denormalized
        input values (predictors before optimization).
    :param denorm_output_matrix: Same but after optimization.
    :param initial_activations: length-E numpy array of model activations before
        optimization.
    :param final_activations: length-E numpy array of model activations after
        optimization.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to model that created saliency maps (readable
        by `cnn.read_model`).
    :param metadata_dict: Dictionary created by
        `backwards_optimization.check_metadata` in GewitterGefahr.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    num_examples = len(example_id_strings)
    these_expected_dim = numpy.array([num_examples], dtype=int)

    _check_in_and_out_matrices(
        input_matrix=denorm_input_matrix, output_matrix=denorm_output_matrix,
        num_examples=num_examples)

    error_checking.assert_is_numpy_array_without_nan(initial_activations)
    error_checking.assert_is_numpy_array(initial_activations,
                                         exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array_without_nan(final_activations)
    error_checking.assert_is_numpy_array(final_activations,
                                         exact_dimensions=these_expected_dim)

    bwo_dictionary = {
        INPUT_MATRIX_KEY: denorm_input_matrix,
        OUTPUT_MATRIX_KEY: denorm_output_matrix,
        INITIAL_ACTIVATIONS_KEY: initial_activations,
        FINAL_ACTIVATIONS_KEY: final_activations,
        EXAMPLE_IDS_KEY: example_id_strings,
        MODEL_FILE_KEY: model_file_name,
        NUM_ITERATIONS_KEY: metadata_dict[NUM_ITERATIONS_KEY],
        LEARNING_RATE_KEY: metadata_dict[LEARNING_RATE_KEY],
        L2_WEIGHT_KEY: metadata_dict[L2_WEIGHT_KEY],
        COMPONENT_TYPE_KEY: metadata_dict[COMPONENT_TYPE_KEY],
        TARGET_CLASS_KEY: metadata_dict[TARGET_CLASS_KEY],
        LAYER_NAME_KEY: metadata_dict[LAYER_NAME_KEY],
        IDEAL_ACTIVATION_KEY: metadata_dict[IDEAL_ACTIVATION_KEY],
        NEURON_INDICES_KEY: metadata_dict[NEURON_INDICES_KEY],
        CHANNEL_INDEX_KEY: metadata_dict[CHANNEL_INDEX_KEY]
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(bwo_dictionary, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads composite or non-composite results from Pickle file.

    :param pickle_file_name: Path to input file (created by
        `write_standard_file` or `write_pmm_file`).
    :return: bwo_dictionary: Has the following keys if not a composite...
    bwo_dictionary['denorm_input_matrix']: See doc for `write_standard_file`.
    bwo_dictionary['denorm_output_matrix']: Same.
    bwo_dictionary['initial_activations']: Same.
    bwo_dictionary['final_activations']: Same.
    bwo_dictionary['example_id_strings']: Same.
    bwo_dictionary['model_file_name']: Same.
    bwo_dictionary['num_iterations']: Same.
    bwo_dictionary['learning_rate']: Same.
    bwo_dictionary['l2_weight']: Same.
    bwo_dictionary['component_type_string']: Same.
    bwo_dictionary['target_class']: Same.
    bwo_dictionary['layer_name']: Same.
    bwo_dictionary['ideal_activation']: Same.
    bwo_dictionary['neuron_indices']: Same.
    bwo_dictionary['channel_index']: Same.

    ...or the following keys if composite...

    bwo_dictionary['mean_denorm_input_matrix']: See doc for `write_pmm_file`.
    bwo_dictionary['mean_denorm_output_matrix']: Same.
    bwo_dictionary['mean_initial_activation']: Same.
    bwo_dictionary['mean_final_activation']: Same.
    bwo_dictionary['model_file_name']: Same.
    bwo_dictionary['non_pmm_file_name']: Same.
    bwo_dictionary['pmm_max_percentile_level']: Same.

    :return: pmm_flag: Boolean flag.  True if `bwo_dictionary` contains
        composite, False otherwise.

    :raises: ValueError: if dictionary does not contain expected keys.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    bwo_dictionary = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    pmm_flag = MEAN_INPUT_MATRIX_KEY in bwo_dictionary

    if pmm_flag:
        missing_keys = list(
            set(PMM_FILE_KEYS) - set(bwo_dictionary.keys())
        )
    else:
        missing_keys = list(
            set(STANDARD_FILE_KEYS) - set(bwo_dictionary.keys())
        )

    if len(missing_keys) == 0:
        return bwo_dictionary, pmm_flag

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
