"""Methods for neighbourhood evaluation of gridded labels."""

import pickle
import numpy
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

# TODO(thunderhoser): Make sure to deal with masked grid cells.  Maybe by just
# having these labels and probs be all zero?

TOLERANCE = 1e-6
NARR_GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME
)[0]

FRONT_TYPE_ENUMS = [
    front_utils.NO_FRONT_ENUM, front_utils.WARM_FRONT_ENUM,
    front_utils.COLD_FRONT_ENUM
]

NUM_ACTUAL_ORIENTED_TP_KEY = 'num_actual_oriented_true_positives'
NUM_PREDICTION_ORIENTED_TP_KEY = 'num_prediction_oriented_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'

PREDICTED_LABELS_KEY = 'predicted_label_matrix'
ACTUAL_LABELS_KEY = 'actual_label_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
NEIGH_DISTANCE_KEY = 'neigh_distance_metres'
BINARY_CONTINGENCY_TABLE_KEY = 'binary_ct_as_dict'
PREDICTION_ORIENTED_CT_KEY = 'prediction_oriented_ct_matrix'
ACTUAL_ORIENTED_CT_KEY = 'actual_oriented_ct_matrix'
GRID_SPACING_KEY = 'grid_spacing_metres'

REQUIRED_KEYS = [
    PREDICTED_LABELS_KEY, ACTUAL_LABELS_KEY, NEIGH_DISTANCE_KEY,
    BINARY_CONTINGENCY_TABLE_KEY, PREDICTION_ORIENTED_CT_KEY,
    ACTUAL_ORIENTED_CT_KEY, GRID_SPACING_KEY
]


def _check_gridded_predictions(prediction_matrix, expect_probs):
    """Error-checks gridded predictions.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    K = number of classes

    :param prediction_matrix: [if `expect_probs == True`]
        T-by-M-by-N-by-K numpy array of class probabilities, where
        prediction_matrix[i, m, n, k] = probability that grid cell [m, n] in the
        [i]th example belongs to the [k]th class.

        [if `expect_probs == False`]
        T-by-M-by-N numpy array of integers (each must be accepted by
        `front_utils.check_front_type_enum`).

    :param expect_probs: Boolean flag.  If True, will expect `prediction_matrix`
        to contain probabilities.  If False, will expect `prediction_matrix` to
        contain deterministic labels.
    """

    if expect_probs:
        error_checking.assert_is_numpy_array(
            prediction_matrix, num_dimensions=4)
        error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)
        error_checking.assert_is_leq_numpy_array(prediction_matrix, 1.)

        num_classes = prediction_matrix.shape[-1]
        error_checking.assert_is_geq(num_classes, 3)
        error_checking.assert_is_leq(num_classes, 3)

        summed_prediction_matrix = numpy.sum(prediction_matrix, axis=-1)
        assert numpy.allclose(summed_prediction_matrix, 1., atol=TOLERANCE)

        return

    error_checking.assert_is_integer_numpy_array(prediction_matrix)
    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=3)

    error_checking.assert_is_geq_numpy_array(
        prediction_matrix, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS)
    )
    error_checking.assert_is_leq_numpy_array(
        prediction_matrix, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS)
    )


def _match_actual_wf_grid_cells(
        predicted_label_matrix_one_time, actual_label_matrix_one_time,
        neigh_distance_metres, grid_spacing_metres):
    """Matches actual warm-frontal grid cells with predictions.

    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix_one_time: M-by-N numpy array of integers (each
        must be accepted by `front_utils.check_front_type_enum`).
    :param actual_label_matrix_one_time: Same.
    :param neigh_distance_metres: Neighbourhood distance.
    :param grid_spacing_metres: Grid spacing (this method assumes that the grid
        is equidistant).
    :return: num_predicted_by_class: length-3 numpy array with number of
        matching predictions for each class.  Correspondence between array index
        and class is given by "ENUM"s listed at the top of this file.
    """

    predicted_label_matrix_one_time = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=predicted_label_matrix_one_time,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.WARM_FRONT_ENUM)

    predicted_front_type_enums = predicted_label_matrix_one_time[
        numpy.where(actual_label_matrix_one_time == front_utils.WARM_FRONT_ENUM)
    ]

    num_predicted_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_predicted_by_class[k] = numpy.sum(predicted_front_type_enums == k)

    return num_predicted_by_class


def _match_actual_cf_grid_cells(
        predicted_label_matrix_one_time, actual_label_matrix_one_time,
        neigh_distance_metres, grid_spacing_metres):
    """Matches actual cold-frontal grid cells with predictions.

    :param predicted_label_matrix_one_time: See doc for
        `_match_actual_wf_grid_cells`.
    :param actual_label_matrix_one_time: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_predicted_by_class: Same.
    """

    predicted_label_matrix_one_time = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=predicted_label_matrix_one_time,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.COLD_FRONT_ENUM)

    predicted_front_type_enums = predicted_label_matrix_one_time[
        numpy.where(actual_label_matrix_one_time == front_utils.COLD_FRONT_ENUM)
    ]

    num_predicted_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_predicted_by_class[k] = numpy.sum(predicted_front_type_enums == k)

    return num_predicted_by_class


def _match_predicted_wf_grid_cells(
        predicted_label_matrix_one_time, actual_label_matrix_one_time,
        neigh_distance_metres, grid_spacing_metres):
    """Matches predicted warm-frontal grid cells with predictions.

    :param predicted_label_matrix_one_time: See doc for
        `_match_actual_wf_grid_cells`.
    :param actual_label_matrix_one_time: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_actual_by_class: length-3 numpy array with number of matching
        actual labels for each class.  Correspondence between array index
        and class is given by "ENUM"s listed at the top of this file.
    """

    actual_label_matrix_one_time = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=actual_label_matrix_one_time,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.WARM_FRONT_ENUM)

    actual_front_type_enums = actual_label_matrix_one_time[
        numpy.where(predicted_label_matrix_one_time ==
                    front_utils.WARM_FRONT_ENUM)
    ]

    num_actual_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_actual_by_class[k] = numpy.sum(actual_front_type_enums == k)

    return num_actual_by_class


def _match_predicted_cf_grid_cells(
        predicted_label_matrix_one_time, actual_label_matrix_one_time,
        neigh_distance_metres, grid_spacing_metres):
    """Matches predicted cold-frontal grid cells with predictions.

    :param predicted_label_matrix_one_time: See doc for
        `_match_actual_wf_grid_cells`.
    :param actual_label_matrix_one_time: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_actual_by_class: See doc for `_match_predicted_wf_grid_cells`.
    """

    actual_label_matrix_one_time = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=actual_label_matrix_one_time,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.COLD_FRONT_ENUM)

    actual_front_type_enums = actual_label_matrix_one_time[
        numpy.where(predicted_label_matrix_one_time ==
                    front_utils.COLD_FRONT_ENUM)
    ]

    num_actual_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_actual_by_class[k] = numpy.sum(actual_front_type_enums == k)

    return num_actual_by_class


def determinize_predictions(class_probability_matrix, binarization_threshold):
    """Determinizes predictions (converts from probabilistic to deterministic).

    :param class_probability_matrix: See doc for `_check_gridded_predictions`
        with `expect_probs == True`.
    :param binarization_threshold: Binarization threshold.  For each case (i.e.,
        each grid cell at each time step), if NF probability >=
        `binarization_threshold`, the deterministic label will be NF.
        Otherwise, the deterministic label will be the max of WF and CF
        probabilities.
    :return: predicted_label_matrix: See doc for `_check_gridded_predictions`
        with `expect_probs == False`.
    """

    _check_gridded_predictions(prediction_matrix=class_probability_matrix,
                               expect_probs=True)

    error_checking.assert_is_geq(binarization_threshold, 0.)
    error_checking.assert_is_leq(binarization_threshold, 1.)

    predicted_label_matrix = 1 + numpy.argmax(
        class_probability_matrix[..., 1:], axis=-1
    )

    nf_flag_matrix = (
        class_probability_matrix[..., front_utils.NO_FRONT_ENUM] >=
        binarization_threshold
    )

    predicted_label_matrix[
        numpy.where(nf_flag_matrix)
    ] = front_utils.NO_FRONT_ENUM

    return predicted_label_matrix


def make_contingency_tables(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Creates contingency tables.

    :param predicted_label_matrix: See doc for `_check_gridded_predictions`
        with `expect_probs == False`.
    :param actual_label_matrix: Same.
    :param neigh_distance_metres: Neighbourhood distance.
    :param grid_spacing_metres: Grid spacing (this method assumes that the grid
        is equidistant).

    :return: binary_ct_as_dict: Dictionary with the following keys.
    binary_ct_as_dict['num_actual_oriented_true_positives']: Number of actual
        frontal grid cells with a matching prediction within
        `neigh_distance_metres`.
    binary_ct_as_dict['num_prediction_oriented_true_positives']: Number of
        predicted frontal grid cells with a matching actual within
        `neigh_distance_metres`.
    binary_ct_as_dict['num_false_positives']: Number of predicted frontal grid
        cells with *no* matching actual within `neigh_distance_metres`.
    binary_ct_as_dict['num_false_negatives']: Number of actual frontal grid
        cells with *no* matching prediction within `neigh_distance_metres`.

    :return: prediction_oriented_ct_matrix: 3-by-3 numpy array.
        prediction_oriented_ct_matrix[i, j] is the probability, given that the
        [i]th class is predicted, that the [j]th class will be observed.  Array
        indices follow the "ENUM"s listed at the top of this file, and the first
        row (for NF predictions) is all NaN, because this file does not handle
        negative predictions.
    :return: actual_oriented_ct_matrix: 3-by-3 numpy array.
        actual_oriented_ct_matrix[i, j] is the probability, given that the [j]th
        class is observed, that the [i]th class will be predicted.  Array
        indices follow the "ENUM"s listed at the top of this file, and the first
        column (for NF predictions) is all NaN, because this file does not
        handle negative observations.
    """

    # TODO(thunderhoser): Incorporate time lag here?

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    _check_gridded_predictions(
        prediction_matrix=predicted_label_matrix, expect_probs=False)
    _check_gridded_predictions(
        prediction_matrix=actual_label_matrix, expect_probs=False)

    error_checking.assert_is_numpy_array(
        actual_label_matrix,
        exact_dimensions=numpy.array(predicted_label_matrix.shape, dtype=int)
    )

    binary_ct_as_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: 0,
        NUM_PREDICTION_ORIENTED_TP_KEY: 0,
        NUM_FALSE_POSITIVES_KEY: 0,
        NUM_FALSE_NEGATIVES_KEY: 0
    }

    num_classes = len(FRONT_TYPE_ENUMS)
    prediction_oriented_ct_matrix = numpy.full(
        (num_classes, num_classes), 0, dtype=int
    )
    actual_oriented_ct_matrix = numpy.full(
        (num_classes, num_classes), 0, dtype=int
    )

    num_times = predicted_label_matrix.shape[0]

    for i in range(num_times):
        print (
            'Matching actual WF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times)

        this_num_predicted_by_class = _match_actual_wf_grid_cells(
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix_one_time=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

        actual_oriented_ct_matrix[:, front_utils.WARM_FRONT_ENUM] = (
            actual_oriented_ct_matrix[:, front_utils.WARM_FRONT_ENUM] +
            this_num_predicted_by_class
        )

        binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY] += (
            this_num_predicted_by_class[front_utils.WARM_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY] += (
            numpy.sum(this_num_predicted_by_class) -
            this_num_predicted_by_class[front_utils.WARM_FRONT_ENUM]
        )

        print (
            'Matching actual CF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times)

        this_num_predicted_by_class = _match_actual_cf_grid_cells(
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix_one_time=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

        actual_oriented_ct_matrix[:, front_utils.COLD_FRONT_ENUM] = (
            actual_oriented_ct_matrix[:, front_utils.COLD_FRONT_ENUM] +
            this_num_predicted_by_class
        )

        binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY] += (
            this_num_predicted_by_class[front_utils.COLD_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY] += (
            numpy.sum(this_num_predicted_by_class) -
            this_num_predicted_by_class[front_utils.COLD_FRONT_ENUM]
        )

        print (
            'Matching predicted WF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times)

        this_num_actual_by_class = _match_predicted_wf_grid_cells(
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix_one_time=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

        prediction_oriented_ct_matrix[front_utils.WARM_FRONT_ENUM, :] = (
            prediction_oriented_ct_matrix[front_utils.WARM_FRONT_ENUM, :] +
            this_num_actual_by_class
        )

        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY] += (
            this_num_actual_by_class[front_utils.WARM_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] += (
            numpy.sum(this_num_actual_by_class) -
            this_num_actual_by_class[front_utils.WARM_FRONT_ENUM]
        )

        print (
            'Matching predicted CF grid cells at {0:d}th of {1:d} times...\n'
        ).format(i + 1, num_times)

        this_num_actual_by_class = _match_predicted_cf_grid_cells(
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix_one_time=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

        prediction_oriented_ct_matrix[front_utils.COLD_FRONT_ENUM, :] = (
            prediction_oriented_ct_matrix[front_utils.COLD_FRONT_ENUM, :] +
            this_num_actual_by_class
        )

        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY] += (
            this_num_actual_by_class[front_utils.COLD_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] += (
            numpy.sum(this_num_actual_by_class) -
            this_num_actual_by_class[front_utils.COLD_FRONT_ENUM]
        )

    prediction_oriented_ct_matrix = prediction_oriented_ct_matrix.astype(float)
    prediction_oriented_ct_matrix[0, :] = numpy.nan

    for k in range(1, num_classes):
        if numpy.sum(prediction_oriented_ct_matrix[k, :]) == 0:
            prediction_oriented_ct_matrix[k, :] = numpy.nan
        else:
            prediction_oriented_ct_matrix[k, :] = (
                prediction_oriented_ct_matrix[k, :] /
                numpy.sum(prediction_oriented_ct_matrix[k, :])
            )

    actual_oriented_ct_matrix = actual_oriented_ct_matrix.astype(float)
    actual_oriented_ct_matrix[:, 0] = numpy.nan

    for k in range(1, num_classes):
        if numpy.sum(actual_oriented_ct_matrix[:, k]) == 0:
            actual_oriented_ct_matrix[:, k] = numpy.nan
        else:
            actual_oriented_ct_matrix[:, k] = (
                actual_oriented_ct_matrix[:, k] /
                numpy.sum(actual_oriented_ct_matrix[:, k])
            )

    return (binary_ct_as_dict, prediction_oriented_ct_matrix,
            actual_oriented_ct_matrix)


def get_binary_pod(binary_ct_as_dict):
    """Returns probability of detection.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_pod: Binary POD.
    """

    numerator = binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY]
    denominator = numerator + binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_fom(binary_ct_as_dict):
    """Returns frequency of misses.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_fom: Binary FOM.
    """

    return 1. - get_binary_pod(binary_ct_as_dict)


def get_binary_success_ratio(binary_ct_as_dict):
    """Returns success ratio.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_success_ratio: Binary success ratio.
    """

    numerator = binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY]
    denominator = numerator + binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_far(binary_ct_as_dict):
    """Returns false-alarm rate.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_far: Binary FAR.
    """

    return 1. - get_binary_success_ratio(binary_ct_as_dict)


def get_binary_csi(binary_ct_as_dict):
    """Returns critical success index.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_csi: Binary CSI.
    """

    binary_pod = get_binary_pod(binary_ct_as_dict)
    binary_success_ratio = get_binary_success_ratio(binary_ct_as_dict)

    try:
        return (binary_pod ** -1 + binary_success_ratio ** -1 - 1) ** -1
    except ZeroDivisionError:
        return numpy.nan


def get_binary_frequency_bias(binary_ct_as_dict):
    """Returns frequency bias.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: binary_frequency_bias: Binary frequency bias.
    """

    binary_pod = get_binary_pod(binary_ct_as_dict)
    binary_success_ratio = get_binary_success_ratio(binary_ct_as_dict)

    try:
        return binary_pod / binary_success_ratio
    except ZeroDivisionError:
        return numpy.nan


def write_results(
        pickle_file_name, predicted_label_matrix, actual_label_matrix,
        valid_times_unix_sec, neigh_distance_metres, binary_ct_as_dict,
        prediction_oriented_ct_matrix, actual_oriented_ct_matrix,
        grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Writes results of neighbourhood evaluation to file.

    :param pickle_file_name: Path to output file.
    :param predicted_label_matrix: See doc for `make_contingency_tables`.
    :param actual_label_matrix: Same.
    :param valid_times_unix_sec: 1-D numpy array of valid times.
    :param neigh_distance_metres: Same.
    :param binary_ct_as_dict: Same.
    :param prediction_oriented_ct_matrix: Same.
    :param actual_oriented_ct_matrix: Same.
    :param grid_spacing_metres: Same.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    _check_gridded_predictions(
        prediction_matrix=predicted_label_matrix, expect_probs=False)
    _check_gridded_predictions(
        prediction_matrix=actual_label_matrix, expect_probs=False)

    error_checking.assert_is_numpy_array(
        actual_label_matrix,
        exact_dimensions=numpy.array(predicted_label_matrix.shape, dtype=int)
    )

    num_times = predicted_label_matrix.shape[0]

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec,
        exact_dimensions=numpy.array([num_times], dtype=int)
    )

    num_classes = len(FRONT_TYPE_ENUMS)

    error_checking.assert_is_numpy_array(
        prediction_oriented_ct_matrix,
        exact_dimensions=numpy.array([num_classes, num_classes], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        actual_oriented_ct_matrix,
        exact_dimensions=numpy.array([num_classes, num_classes], dtype=int)
    )

    evaluation_dict = {
        PREDICTED_LABELS_KEY: predicted_label_matrix,
        ACTUAL_LABELS_KEY: actual_label_matrix,
        VALID_TIMES_KEY: valid_times_unix_sec,
        NEIGH_DISTANCE_KEY: neigh_distance_metres,
        BINARY_CONTINGENCY_TABLE_KEY: binary_ct_as_dict,
        PREDICTION_ORIENTED_CT_KEY: prediction_oriented_ct_matrix,
        ACTUAL_ORIENTED_CT_KEY: actual_oriented_ct_matrix,
        GRID_SPACING_KEY: grid_spacing_metres
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(evaluation_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_results(pickle_file_name):
    """Reads results of neighbourhood evaluation from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: evaluation_dict: Dictionary with keys listed in `write_results`.
    :raises: ValueError: if any of the expected keys are not found.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    evaluation_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(REQUIRED_KEYS) - set(evaluation_dict.keys()))
    if len(missing_keys) == 0:
        return evaluation_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
