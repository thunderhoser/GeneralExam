"""Methods for neighbourhood evaluation of gridded labels."""

import pickle
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6
NARR_GRID_SPACING_METRES = nwp_model_utils.get_xy_grid_spacing(
    model_name=nwp_model_utils.NARR_MODEL_NAME,
    grid_name=nwp_model_utils.NAME_OF_221GRID
)[0]

FRONT_TYPE_ENUMS = [
    front_utils.NO_FRONT_ENUM, front_utils.WARM_FRONT_ENUM,
    front_utils.COLD_FRONT_ENUM
]

NUM_ACTUAL_ORIENTED_TP_KEY = 'num_actual_oriented_true_positives'
NUM_PREDICTION_ORIENTED_TP_KEY = 'num_prediction_oriented_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'

PREDICTION_FILES_KEY = 'prediction_file_names'
NEIGH_DISTANCE_KEY = 'neigh_distance_metres'
BINARY_CONTINGENCY_TABLE_KEY = 'binary_ct_as_dict'
BINARY_CONTINGENCY_TABLES_KEY = 'list_of_binary_ct_dicts'
PREDICTION_ORIENTED_CT_KEY = 'prediction_oriented_ct_matrix'
ACTUAL_ORIENTED_CT_KEY = 'actual_oriented_ct_matrix'

REQUIRED_KEYS = [
    PREDICTION_FILES_KEY, NEIGH_DISTANCE_KEY, BINARY_CONTINGENCY_TABLES_KEY,
    PREDICTION_ORIENTED_CT_KEY, ACTUAL_ORIENTED_CT_KEY
]


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


def check_gridded_predictions(prediction_matrix, expect_probs):
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


def determinize_predictions_1threshold(
        class_probability_matrix, binarization_threshold):
    """Determinizes predictions (converts from probabilistic to deterministic).

    In this case there is only one threshold, for the NF probability.

    :param class_probability_matrix: See doc for `check_gridded_predictions`
        with `expect_probs == True`.
    :param binarization_threshold: Binarization threshold.  For each case (i.e.,
        each grid cell at each time step), if NF probability >=
        `binarization_threshold`, the deterministic label will be NF.
        Otherwise, the deterministic label will be the max of WF and CF
        probabilities.
    :return: predicted_label_matrix: See doc for `check_gridded_predictions`
        with `expect_probs == False`.
    """

    check_gridded_predictions(prediction_matrix=class_probability_matrix,
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


def determinize_predictions_2thresholds(
        class_probability_matrix, wf_threshold, cf_threshold):
    """Determinizes predictions (converts from probabilistic to deterministic).

    In this case there are two thresholds, one for WF probability and one for CF
    probability.  If both probabilities exceed their respective thresholds, the
    highest one is used to determine the label.

    :param class_probability_matrix: See doc for `check_gridded_predictions`
        with `expect_probs == True`.
    :param wf_threshold: WF-probability threshold.
    :param cf_threshold: CF-probability threshold.
    :return: predicted_label_matrix: See doc for `check_gridded_predictions`
        with `expect_probs == False`.
    """

    check_gridded_predictions(prediction_matrix=class_probability_matrix,
                              expect_probs=True)

    error_checking.assert_is_geq(wf_threshold, 0.)
    error_checking.assert_is_leq(wf_threshold, 1.)
    error_checking.assert_is_geq(cf_threshold, 0.)
    error_checking.assert_is_leq(cf_threshold, 1.)

    predicted_label_matrix = numpy.full(
        class_probability_matrix.shape[:-1], front_utils.NO_FRONT_ENUM,
        dtype=int)

    wf_flag_matrix = numpy.logical_and(
        class_probability_matrix[..., front_utils.WARM_FRONT_ENUM] >=
        wf_threshold,
        class_probability_matrix[..., front_utils.WARM_FRONT_ENUM] >=
        class_probability_matrix[..., front_utils.COLD_FRONT_ENUM]
    )

    cf_flag_matrix = numpy.logical_and(
        class_probability_matrix[..., front_utils.COLD_FRONT_ENUM] >=
        cf_threshold,
        class_probability_matrix[..., front_utils.COLD_FRONT_ENUM] >=
        class_probability_matrix[..., front_utils.WARM_FRONT_ENUM]
    )

    predicted_label_matrix[wf_flag_matrix] = front_utils.WARM_FRONT_ENUM
    predicted_label_matrix[cf_flag_matrix] = front_utils.COLD_FRONT_ENUM

    return predicted_label_matrix


def remove_small_regions_one_time(
        predicted_label_matrix, min_region_length_metres,
        buffer_distance_metres, grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Removes small regions of frontal (WF or CF) labels.

    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix: M-by-N numpy array of integers (each must be
        accepted by `front_utils.check_front_type_enum`).
    :param min_region_length_metres: Minimum region length (applied to major
        axis).
    :param buffer_distance_metres: Buffer distance.  Small region R will be
        removed if it is > `buffer_distance_metres` away from the nearest large
        region.
    :param grid_spacing_metres: Grid spacing (this method assumes that the grid
        is equidistant).
    :return: predicted_label_matrix: Same as input but maybe with fewer frontal
        labels.
    """

    error_checking.assert_is_numpy_array(predicted_label_matrix)
    check_gridded_predictions(
        prediction_matrix=numpy.expand_dims(predicted_label_matrix, axis=0),
        expect_probs=False
    )

    error_checking.assert_is_greater(min_region_length_metres, 0.)
    error_checking.assert_is_greater(buffer_distance_metres, 0.)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    region_dict = front_utils.gridded_labels_to_regions(
        ternary_label_matrix=predicted_label_matrix, compute_lengths=True)

    region_lengths_metres = grid_spacing_metres * region_dict[
        front_utils.MAJOR_AXIS_LENGTHS_KEY]

    # for i in range(len(region_lengths_metres)):
    #     print (
    #         'Front type = {0:s} ... rows = {1:d}-{2:d} ... columns = '
    #         '{3:d}-{4:d} ... length = {5:f} km'
    #     ).format(
    #         region_dict[front_utils.FRONT_TYPES_KEY][i],
    #         numpy.min(region_dict[front_utils.ROWS_BY_REGION_KEY][i]),
    #         numpy.max(region_dict[front_utils.ROWS_BY_REGION_KEY][i]),
    #         numpy.min(region_dict[front_utils.COLUMNS_BY_REGION_KEY][i]),
    #         numpy.max(region_dict[front_utils.COLUMNS_BY_REGION_KEY][i]),
    #         0.001 * region_lengths_metres[i]
    #     )

    small_region_flags = region_lengths_metres < min_region_length_metres
    if not numpy.any(small_region_flags):
        return predicted_label_matrix

    small_region_indices = numpy.where(small_region_flags)[0]
    large_region_indices = numpy.where(numpy.invert(small_region_flags))[0]

    for i in small_region_indices:
        these_rows = region_dict[front_utils.ROWS_BY_REGION_KEY][i]
        these_columns = region_dict[front_utils.COLUMNS_BY_REGION_KEY][i]

        this_small_region_coord_matrix = numpy.hstack((
            numpy.reshape(these_columns, (these_columns.size, 1)),
            numpy.reshape(these_rows, (these_rows.size, 1))
        ))

        this_small_region_closed = False

        for j in large_region_indices:
            if (region_dict[front_utils.FRONT_TYPES_KEY][i] !=
                    region_dict[front_utils.FRONT_TYPES_KEY][j]):
                continue

            these_rows = region_dict[front_utils.ROWS_BY_REGION_KEY][j]
            these_columns = region_dict[front_utils.COLUMNS_BY_REGION_KEY][j]

            this_large_region_coord_matrix = numpy.hstack((
                numpy.reshape(these_columns, (these_columns.size, 1)),
                numpy.reshape(these_rows, (these_rows.size, 1))
            ))

            this_distance_matrix_metres = grid_spacing_metres * (
                euclidean_distances(X=this_small_region_coord_matrix,
                                    Y=this_large_region_coord_matrix)
            )

            this_small_region_closed = (
                numpy.min(this_distance_matrix_metres) <= buffer_distance_metres
            )

            if this_small_region_closed:
                break

        if this_small_region_closed:
            continue

        predicted_label_matrix[
            region_dict[front_utils.ROWS_BY_REGION_KEY][i],
            region_dict[front_utils.COLUMNS_BY_REGION_KEY][i]
        ] = front_utils.NO_FRONT_ENUM

    return predicted_label_matrix


def make_contingency_tables(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        normalize, grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Creates contingency tables.

    :param predicted_label_matrix: See doc for `check_gridded_predictions`
        with `expect_probs == False`.
    :param actual_label_matrix: Same.
    :param neigh_distance_metres: Neighbourhood distance.
    :param normalize: Boolean flag.  If True, will normalize contingency tables
        so that each row in `prediction_oriented_ct_matrix` and each column in
        `actual_oriented_ct_matrix` sums to 1.0.  If False, will return raw
        counts.
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
        prediction_oriented_ct_matrix[i, j] is the total number or fraction of
        times, when the [i]th class is predicted, that the [j]th class is
        observed.  Array indices follow the "ENUM"s listed at the top of this
        file, and the first row (for NF predictions) is all NaN, because this
        file does not handle negative predictions.
    :return: actual_oriented_ct_matrix: 3-by-3 numpy array.
        actual_oriented_ct_matrix[i, j] is the total number or fraction of
        times, when the [j]th class is observed, that the [i]th class is
        predicted.  Array indices follow the "ENUM"s listed at the top of this
        file, and the first column (for NF predictions) is all NaN, because this
        file does not handle negative observations.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_boolean(normalize)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    check_gridded_predictions(
        prediction_matrix=predicted_label_matrix, expect_probs=False)
    check_gridded_predictions(
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
    dimensions = (num_classes, num_classes)

    prediction_oriented_ct_matrix = numpy.full(dimensions, 0.)
    prediction_oriented_ct_matrix[0, :] = numpy.nan
    actual_oriented_ct_matrix = numpy.full(dimensions, 0.)
    actual_oriented_ct_matrix[:, 0] = numpy.nan

    num_times = predicted_label_matrix.shape[0]

    for i in range(num_times):
        print((
            'Matching actual WF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times))

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

        print((
            'Matching actual CF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times))

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

        print((
            'Matching predicted WF grid cells at {0:d}th of {1:d} times...'
        ).format(i + 1, num_times))

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

        print((
            'Matching predicted CF grid cells at {0:d}th of {1:d} times...\n'
        ).format(i + 1, num_times))

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

    if normalize:
        prediction_oriented_ct_matrix, actual_oriented_ct_matrix = (
            normalize_contingency_tables(
                prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
                actual_oriented_ct_matrix=actual_oriented_ct_matrix)
        )

    return (binary_ct_as_dict, prediction_oriented_ct_matrix,
            actual_oriented_ct_matrix)


def normalize_contingency_tables(prediction_oriented_ct_matrix,
                                 actual_oriented_ct_matrix):
    """Normalizes 3-class contingency tables.

    :param prediction_oriented_ct_matrix: See output doc for
        `make_contingency_tables` with `normalize == False`.
    :param actual_oriented_ct_matrix: Same.
    :return: prediction_oriented_ct_matrix: See output doc for
        `make_contingency_tables` with `normalize == True`.
    :return: actual_oriented_ct_matrix: Same.
    """

    num_classes = len(FRONT_TYPE_ENUMS)
    expected_dim = numpy.array([num_classes, num_classes], dtype=int)

    error_checking.assert_is_numpy_array(
        prediction_oriented_ct_matrix, exact_dimensions=expected_dim)
    assert numpy.all(numpy.isnan(prediction_oriented_ct_matrix[0, ...]))
    error_checking.assert_is_numpy_array_without_nan(
        prediction_oriented_ct_matrix[1:, ...]
    )
    error_checking.assert_is_geq_numpy_array(
        prediction_oriented_ct_matrix[1:, ...], 0
    )

    error_checking.assert_is_numpy_array(
        actual_oriented_ct_matrix, exact_dimensions=expected_dim)
    assert numpy.all(numpy.isnan(actual_oriented_ct_matrix[..., 0]))
    error_checking.assert_is_numpy_array_without_nan(
        actual_oriented_ct_matrix[..., 1:]
    )
    error_checking.assert_is_geq_numpy_array(
        actual_oriented_ct_matrix[..., 1:], 0
    )

    for k in range(1, num_classes):
        if numpy.sum(prediction_oriented_ct_matrix[k, :]) == 0:
            prediction_oriented_ct_matrix[k, :] = numpy.nan
        else:
            prediction_oriented_ct_matrix[k, :] = (
                prediction_oriented_ct_matrix[k, :] /
                numpy.sum(prediction_oriented_ct_matrix[k, :])
            )

    for k in range(1, num_classes):
        if numpy.sum(actual_oriented_ct_matrix[:, k]) == 0:
            actual_oriented_ct_matrix[:, k] = numpy.nan
        else:
            actual_oriented_ct_matrix[:, k] = (
                actual_oriented_ct_matrix[:, k] /
                numpy.sum(actual_oriented_ct_matrix[:, k])
            )

    return prediction_oriented_ct_matrix, actual_oriented_ct_matrix


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
        pickle_file_name, prediction_file_names, neigh_distance_metres,
        list_of_binary_ct_dicts, prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix):
    """Writes results of neighbourhood evaluation to Pickle file.

    B = number of bootstrap replicates

    :param pickle_file_name: Path to output file.
    :param prediction_file_names: 1-D list of paths to input files (readable by
        `prediction_io.read_file`).
    :param neigh_distance_metres: Neighbourhood distance.
    :param list_of_binary_ct_dicts: length-B list of binary contingency tables
        created by `make_contingency_tables`.
    :param prediction_oriented_ct_matrix: B-by-3-by-3 numpy array, where
        prediction_oriented_ct_matrix[i, ...] is the prediction-oriented
        contingency table, created by `make_contingency_tables`, for the [i]th
        bootstrap replicate.
    :param actual_oriented_ct_matrix: Same but with actual-oriented contingency
        tables.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(prediction_file_names), num_dimensions=1
    )

    error_checking.assert_is_list(list_of_binary_ct_dicts)

    num_classes = len(FRONT_TYPE_ENUMS)
    num_bootstrap_reps = len(list_of_binary_ct_dicts)
    these_expected_dim = numpy.array(
        [num_bootstrap_reps, num_classes, num_classes], dtype=int
    )

    error_checking.assert_is_numpy_array(
        prediction_oriented_ct_matrix, exact_dimensions=these_expected_dim
    )
    error_checking.assert_is_numpy_array(
        actual_oriented_ct_matrix, exact_dimensions=these_expected_dim
    )

    evaluation_dict = {
        PREDICTION_FILES_KEY: prediction_file_names,
        NEIGH_DISTANCE_KEY: neigh_distance_metres,
        BINARY_CONTINGENCY_TABLES_KEY: list_of_binary_ct_dicts,
        PREDICTION_ORIENTED_CT_KEY: prediction_oriented_ct_matrix,
        ACTUAL_ORIENTED_CT_KEY: actual_oriented_ct_matrix
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
