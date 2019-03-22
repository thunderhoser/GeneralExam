"""Methods for neighbourhood evaluation of gridded labels."""

import numpy
from gewittergefahr.gg_utils import nwp_model_utils
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


def _check_gridded_predictions(prediction_matrix, expect_probs):
    """Error-checks gridded predictions.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    K = number of classes

    :param prediction_matrix: [if `expect_probs == True`]
        E-by-M-by-N-by-K numpy array of class probabilities, where
        prediction_matrix[i, m, n, k] = probability that grid cell [m, n] in the
        [i]th example belongs to the [k]th class.

        [if `expect_probs == False`]
        E-by-M-by-N numpy array of integers (each must be accepted by
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


def get_binary_contingency_table(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Creates binary ("front vs. no front") contingency table.

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

    num_times = predicted_label_matrix.shape[0]

    for i in range(num_times):
        print (
            'Matching actual WF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times)

        this_num_predicted_by_class = _match_actual_wf_grid_cells(
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...],
            actual_label_matrix_one_time=actual_label_matrix[i, ...],
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

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
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...],
            actual_label_matrix_one_time=actual_label_matrix[i, ...],
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

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
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...],
            actual_label_matrix_one_time=actual_label_matrix[i, ...],
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

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
            predicted_label_matrix_one_time=predicted_label_matrix[i, ...],
            actual_label_matrix_one_time=actual_label_matrix[i, ...],
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres)

        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY] += (
            this_num_actual_by_class[front_utils.COLD_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] += (
            numpy.sum(this_num_actual_by_class) -
            this_num_actual_by_class[front_utils.COLD_FRONT_ENUM]
        )
