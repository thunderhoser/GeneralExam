"""IO methods for saliency maps created by GeneralExam models."""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as gg_saliency

MODEL_FILE_NAME_KEY = 'model_file_name'
COMPONENT_TYPE_KEY = 'component_type_string'
TARGET_CLASS_KEY = 'target_class'
LAYER_NAME_KEY = 'layer_name'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NEURON_INDICES_KEY = 'neuron_indices'
CHANNEL_INDEX_KEY = 'channel_index'


def write_file(
        pickle_file_name, normalized_predictor_matrix, saliency_matrix,
        model_file_name, component_type_string, target_class=None,
        layer_name=None, ideal_activation=None, neuron_indices=None,
        channel_index=None):
    """Writes saliency maps to Pickle file.

    E = number of examples
    M = number of rows in each grid
    N = number of columns in each grid
    C = number of channels (predictor variables)

    :param pickle_file_name: Path to output file.
    :param normalized_predictor_matrix: E-by-M-by-N-by-C numpy array of
        normalized predictor values (input images).
    :param saliency_matrix: E-by-M-by-N-by-C numpy array of saliency values.
    :param model_file_name: Path to file containing trained CNN on which
        saliency maps are based.  Should be readable by
        `traditional_cnn.read_keras_model`.
    :param component_type_string: See doc for
        `gewittergefahr.deep_learning.saliency_maps.check_metadata`.
    :param target_class: Same.
    :param layer_name: Same.
    :param ideal_activation: Same.
    :param neuron_indices: Same.
    :param channel_index: Same.
    """

    gg_saliency.check_metadata(
        component_type_string=component_type_string, target_class=target_class,
        layer_name=layer_name, ideal_activation=ideal_activation,
        neuron_indices=neuron_indices, channel_index=channel_index)

    error_checking.assert_is_numpy_array_without_nan(
        normalized_predictor_matrix)
    error_checking.assert_is_numpy_array(
        normalized_predictor_matrix, num_dimensions=4)

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(
        saliency_matrix,
        exact_dimensions=numpy.array(normalized_predictor_matrix.shape)
    )

    error_checking.assert_is_string(model_file_name)

    metadata_dict = {
        MODEL_FILE_NAME_KEY: model_file_name,
        COMPONENT_TYPE_KEY: component_type_string,
        TARGET_CLASS_KEY: target_class,
        LAYER_NAME_KEY: layer_name,
        IDEAL_ACTIVATION_KEY: ideal_activation,
        NEURON_INDICES_KEY: neuron_indices,
        CHANNEL_INDEX_KEY: channel_index
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(normalized_predictor_matrix, pickle_file_handle)
    pickle.dump(saliency_matrix, pickle_file_handle)
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads saliency maps from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: normalized_predictor_matrix: See doc for `write_file`.
    :return: saliency_matrix: Same.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['model_file_name']: See doc for `write_file`.
    metadata_dict['component_type_string']: Same.
    metadata_dict['target_class']: Same.
    metadata_dict['layer_name']: Same.
    metadata_dict['ideal_activation']: Same.
    metadata_dict['neuron_indices']: Same.
    metadata_dict['channel_index']: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    normalized_predictor_matrix = pickle.load(pickle_file_handle)
    saliency_matrix = pickle.load(pickle_file_handle)
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return normalized_predictor_matrix, saliency_matrix, metadata_dict
