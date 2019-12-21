"""Methods for neighbourhood evaluation of gridded labels."""

import pickle
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

TOLERANCE = 1e-6
NUM_CLASSES = 3
METRES_TO_KM = 0.001

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
BINARY_CONTINGENCY_TABLES_KEY = 'list_of_binary_ct_dicts'
PREDICTION_ORIENTED_CT_KEY = 'prediction_oriented_ct_matrix'
ACTUAL_ORIENTED_CT_KEY = 'actual_oriented_ct_matrix'

REQUIRED_KEYS = [
    PREDICTION_FILES_KEY, NEIGH_DISTANCE_KEY, BINARY_CONTINGENCY_TABLES_KEY,
    PREDICTION_ORIENTED_CT_KEY, ACTUAL_ORIENTED_CT_KEY
]


def _match_actual_wf_one_time(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres):
    """Matches actual warm-frontal grid points at one time.

    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix: M-by-N numpy array of predictions (integers
        accepted by `front_utils.check_front_type_enum`).
    :param actual_label_matrix: Same but with actual labels.
    :param neigh_distance_metres: Neighbourhood distance for matching.
    :param grid_spacing_metres: Grid spacing.  This method assumes that the grid
        is equidistant.
    :return: num_predicted_by_class: length-3 numpy array.
        num_predicted_by_class[i] is the number of matched grid points where the
        predicted class is i.  Correspondence between i and the class (no front,
        warm front, or cold front) is given by `FRONT_TYPE_ENUMS`.
    """

    predicted_label_matrix = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=predicted_label_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.WARM_FRONT_ENUM)

    predicted_types_enum = predicted_label_matrix[
        numpy.where(actual_label_matrix == front_utils.WARM_FRONT_ENUM)
    ]

    num_predicted_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_predicted_by_class[k] = numpy.sum(predicted_types_enum == k)

    return num_predicted_by_class


def _match_actual_cf_one_time(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres):
    """Matches actual cold-frontal grid points at one time.

    :param predicted_label_matrix: See doc for `_match_actual_wf_one_time`.
    :param actual_label_matrix: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_predicted_by_class: Same.
    """

    predicted_label_matrix = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=predicted_label_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.COLD_FRONT_ENUM)

    predicted_types_enum = predicted_label_matrix[
        numpy.where(actual_label_matrix == front_utils.COLD_FRONT_ENUM)
    ]

    num_predicted_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_predicted_by_class[k] = numpy.sum(predicted_types_enum == k)

    return num_predicted_by_class


def _match_predicted_wf_one_time(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres):
    """Matches predicted warm-frontal grid points at one time.

    :param predicted_label_matrix: See doc for `_match_actual_wf_one_time`.
    :param actual_label_matrix: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_actual_by_class: length-3 numpy array.  num_actual_by_class[i]
        is the number of matched grid points where the actual class is j.
        Correspondence between j and the class (no front, warm front, or cold
        front) is given by `FRONT_TYPE_ENUMS`.
    """

    actual_label_matrix = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=actual_label_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.WARM_FRONT_ENUM)

    actual_types_enum = actual_label_matrix[
        numpy.where(predicted_label_matrix == front_utils.WARM_FRONT_ENUM)
    ]

    num_actual_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_actual_by_class[k] = numpy.sum(actual_types_enum == k)

    return num_actual_by_class


def _match_predicted_cf_one_time(
        predicted_label_matrix, actual_label_matrix, neigh_distance_metres,
        grid_spacing_metres):
    """Matches predicted cold-frontal grid points at one time.

    :param predicted_label_matrix: See doc for `_match_actual_wf_one_time`.
    :param actual_label_matrix: Same.
    :param neigh_distance_metres: Same.
    :param grid_spacing_metres: Same.
    :return: num_actual_by_class: See doc for `_match_predicted_wf_one_time`.
    """

    actual_label_matrix = front_utils.dilate_ternary_label_matrix(
        ternary_label_matrix=actual_label_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=grid_spacing_metres,
        tiebreaker_enum=front_utils.COLD_FRONT_ENUM)

    actual_types_enum = actual_label_matrix[
        numpy.where(predicted_label_matrix == front_utils.COLD_FRONT_ENUM)
    ]

    num_actual_by_class = numpy.full(len(FRONT_TYPE_ENUMS), 0, dtype=int)
    for k in range(len(FRONT_TYPE_ENUMS)):
        num_actual_by_class[k] = numpy.sum(actual_types_enum == k)

    return num_actual_by_class


def _check_3class_contingency_tables(
        prediction_oriented_ct_matrix, actual_oriented_ct_matrix,
        expect_normalized):
    """Error-checks 3-class contingency tables.

    :param prediction_oriented_ct_matrix: 3-by-3 numpy array.
        prediction_oriented_ct_matrix[i, j] is the total number or fraction of
        times, when the [i]th class is predicted, that it is matched with an
        observation of the [j]th class.  Array indices follow
        `FRONT_TYPE_ENUMS`.  The first row is all NaN, because there is no such
        thing as a non-frontal region.
    :param actual_oriented_ct_matrix: 3-by-3 numpy array.
        actual_oriented_ct_matrix[i, j] is the total number or fraction of
        times, when the [j]th class is predicted, that it is matched with a
        prediction of the [i]th class.  Array indices follow `FRONT_TYPE_ENUMS`.
        The first column is all NaN, because there is no such thing as a
        non-frontal region.
    :param expect_normalized: Boolean flag.  If True, will expect normalized
        contingency tables, containing fractions.  If False, will expect
        unnormalized tables, containing raw counts.
    """

    expected_dim = numpy.array([NUM_CLASSES, NUM_CLASSES], dtype=int)

    error_checking.assert_is_numpy_array(
        prediction_oriented_ct_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array_without_nan(
        prediction_oriented_ct_matrix[1:, ...]
    )
    error_checking.assert_is_geq_numpy_array(
        prediction_oriented_ct_matrix[1:, ...], 0
    )
    assert numpy.all(numpy.isnan(
        prediction_oriented_ct_matrix[0, ...]
    ))

    error_checking.assert_is_numpy_array(
        actual_oriented_ct_matrix, exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array_without_nan(
        actual_oriented_ct_matrix[..., 1:]
    )
    error_checking.assert_is_geq_numpy_array(
        actual_oriented_ct_matrix[..., 1:], 0
    )
    assert numpy.all(numpy.isnan(
        actual_oriented_ct_matrix[..., 0]
    ))

    if expect_normalized:
        assert numpy.allclose(
            numpy.sum(prediction_oriented_ct_matrix[1:, ...], axis=1),
            1., atol=TOLERANCE
        )

        assert numpy.allclose(
            numpy.sum(actual_oriented_ct_matrix[..., 1:], axis=0),
            1., atol=TOLERANCE
        )
    else:
        assert numpy.allclose(
            prediction_oriented_ct_matrix,
            numpy.round(prediction_oriented_ct_matrix),
            atol=TOLERANCE, equal_nan=True
        )

        assert numpy.allclose(
            actual_oriented_ct_matrix,
            numpy.round(actual_oriented_ct_matrix),
            atol=TOLERANCE, equal_nan=True
        )


def check_gridded_predictions(prediction_matrix, expect_probs):
    """Checks gridded predictions for errors.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    K = number of classes

    :param prediction_matrix: [if `expect_probs == True`]
        T-by-M-by-N-by-K numpy array of class probabilities.

        [if `expect_probs == False`]
        T-by-M-by-N numpy array of predicted labels (integers accepted by
        `front_utils.check_front_type_enum`).

    :param expect_probs: Boolean flag.  If True, will expect probabilities.  If
        False, will expect deterministic labels.
    """

    if expect_probs:
        error_checking.assert_is_numpy_array(
            prediction_matrix, num_dimensions=4
        )
        error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)
        error_checking.assert_is_leq_numpy_array(prediction_matrix, 1.)

        expected_dim = numpy.array(
            prediction_matrix.shape[:-1] + (NUM_CLASSES,), dtype=int
        )
        error_checking.assert_is_numpy_array(
            prediction_matrix, exact_dimensions=expected_dim
        )

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


def dilate_narr_mask(narr_mask_matrix, neigh_distance_metres):
    """Dilates NARR mask.

    All grid points within `neigh_distance_metres` of an unmasked grid point in
    the original mask, will also be unmasked after dilation.

    The resulting mask determines grid points at which a prediction must be
    made, *not* grid points used for evaluation.

    M = number of rows in grid
    N = number of columns in grid

    :param narr_mask_matrix: M-by-N numpy array of integers.  If
        narr_mask_matrix[i, j] = 1, grid point [i, j] is unmasked.
    :param neigh_distance_metres: Neighbourhood distance for evaluation.
    :return: narr_mask_matrix: Same as input but with more ones.
    """

    return front_utils.dilate_binary_label_matrix(
        binary_label_matrix=narr_mask_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES)


def erode_narr_mask(narr_mask_matrix, neigh_distance_metres):
    """Erodes NARR mask.

    All grid points within `neigh_distance_metres` of a masked grid point in the
    original mask, will also be masked after erosion.

    The resulting mask determines grid points to be used for evaluation.

    :param narr_mask_matrix: See doc for `dilate_narr_mask`.
    :param neigh_distance_metres: Same.
    :return: narr_mask_matrix: Same as input but with more zeros.
    """

    this_matrix = front_utils.dilate_binary_label_matrix(
        binary_label_matrix=1 - narr_mask_matrix,
        dilation_distance_metres=neigh_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES)

    return 1 - this_matrix


def determinize_predictions_1threshold(class_probability_matrix, nf_threshold):
    """Converts probabilities to deterministic labels.

    This method uses only one threshold, on the NF probability.

    :param class_probability_matrix: See doc for `check_gridded_predictions`.
    :param nf_threshold: NF-probability threshold.
    :return: predicted_label_matrix: See doc for `check_gridded_predictions`.
    """

    check_gridded_predictions(prediction_matrix=class_probability_matrix,
                              expect_probs=True)

    error_checking.assert_is_geq(nf_threshold, 0.)
    error_checking.assert_is_leq(nf_threshold, 1.)

    predicted_label_matrix = 1 + numpy.argmax(
        class_probability_matrix[..., 1:], axis=-1
    )

    predicted_label_matrix[
        class_probability_matrix[..., front_utils.NO_FRONT_ENUM] >= nf_threshold
    ] = front_utils.NO_FRONT_ENUM

    return predicted_label_matrix


def determinize_predictions_2thresholds(
        class_probability_matrix, wf_threshold, cf_threshold):
    """Converts probabilities to deterministic labels.

    This method uses two thresholds, on both the WF and CF probability.

    :param class_probability_matrix: See doc for `check_gridded_predictions`.
    :param wf_threshold: WF-probability threshold.
    :param cf_threshold: CF-probability threshold.
    :return: predicted_label_matrix: See doc for `check_gridded_predictions`.
    """

    check_gridded_predictions(prediction_matrix=class_probability_matrix,
                              expect_probs=True)

    error_checking.assert_is_geq(wf_threshold, 0.)
    error_checking.assert_is_leq(wf_threshold, 1.)
    error_checking.assert_is_geq(cf_threshold, 0.)
    error_checking.assert_is_leq(cf_threshold, 1.)

    predicted_label_matrix = numpy.full(
        class_probability_matrix.shape[:-1], front_utils.NO_FRONT_ENUM,
        dtype=int
    )

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
        predicted_label_matrix, min_length_metres, buffer_distance_metres,
        grid_spacing_metres=NARR_GRID_SPACING_METRES):
    """Removes small frontal (either WF or CF) regions.

    :param predicted_label_matrix: See doc for `check_gridded_predictions`.
    :param min_length_metres: Minimum length of region's major axis.
    :param buffer_distance_metres: Buffer distance.  A small region will be
        removed only if it is more than `buffer_distance_metres` away from the
        nearest large region of the same type.
    :param grid_spacing_metres: Grid spacing.  This method assumes that the grid
        is equidistant.
    :return: predicted_label_matrix: Same as input but maybe with different
        values.
    """

    error_checking.assert_is_numpy_array(predicted_label_matrix)
    check_gridded_predictions(
        prediction_matrix=numpy.expand_dims(predicted_label_matrix, axis=0),
        expect_probs=False
    )

    error_checking.assert_is_greater(min_length_metres, 0.)
    error_checking.assert_is_greater(buffer_distance_metres, 0.)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    region_dict = front_utils.gridded_labels_to_regions(
        ternary_label_matrix=predicted_label_matrix, compute_lengths=True
    )
    region_lengths_metres = (
        grid_spacing_metres * region_dict[front_utils.MAJOR_AXIS_LENGTHS_KEY]
    )

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

    small_region_flags = region_lengths_metres < min_length_metres
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
            are_regions_same_type = (
                region_dict[front_utils.FRONT_TYPES_KEY][i] ==
                region_dict[front_utils.FRONT_TYPES_KEY][j]
            )

            if not are_regions_same_type:
                continue

            these_rows = region_dict[front_utils.ROWS_BY_REGION_KEY][j]
            these_columns = region_dict[front_utils.COLUMNS_BY_REGION_KEY][j]

            this_large_region_coord_matrix = numpy.hstack((
                numpy.reshape(these_columns, (these_columns.size, 1)),
                numpy.reshape(these_rows, (these_rows.size, 1))
            ))

            this_distance_matrix_metres = grid_spacing_metres * (
                euclidean_distances(
                    X=this_small_region_coord_matrix,
                    Y=this_large_region_coord_matrix
                )
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
        normalize, grid_spacing_metres=NARR_GRID_SPACING_METRES,
        training_mask_matrix=None):
    """Creates contingency tables.

    M = number of rows in grid
    N = number of columns in grid

    :param predicted_label_matrix: See doc for `are_regions_same_type`.
    :param actual_label_matrix: Same as `predicted_label_matrix` but with
        actual, not predicted, fronts.
    :param neigh_distance_metres: Neighbourhood distance for matching frontal
        grid points.
    :param normalize: Boolean flag.  If True, will normalize 3-class contingency
        tables so that each row in `prediction_oriented_ct_matrix` and each
        column in `actual_oriented_ct_matrix` adds up to 1.  If False, will
        return raw counts.
    :param grid_spacing_metres: Grid spacing.  This method assumes that the grid
        is equidistant.
    :param training_mask_matrix: M-by-N numpy array of integers in 0...1,
        indicating which grid cells were masked during training.  0 means that
        the grid cell was masked.

    :return: binary_ct_dict: Dictionary with the following keys.
    binary_ct_dict["num_actual_oriented_true_positives"]: Number of actual
        frontal grid points matched with a predicted grid point of the same
        type.
    binary_ct_dict["num_predicted_oriented_true_positives"]: Number of predicted
        frontal grid points matched with an actual grid point of the same type.
    binary_ct_dict["num_false_positives"]: Number of predicted frontal grid
        points *not* matched with an actual grid point of the same type.
    binary_ct_dict["num_false_negatives"]: Number of actual frontal grid points
        *not* matched with a predicted grid point of the same type.

    :return: prediction_oriented_ct_matrix: See doc for
        `_check_3class_contingency_tables`.
    :return: actual_oriented_ct_matrix: Same.
    """

    check_gridded_predictions(
        prediction_matrix=predicted_label_matrix, expect_probs=False
    )
    check_gridded_predictions(
        prediction_matrix=actual_label_matrix, expect_probs=False
    )
    error_checking.assert_is_numpy_array(
        actual_label_matrix,
        exact_dimensions=numpy.array(predicted_label_matrix.shape, dtype=int)
    )

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_boolean(normalize)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    if training_mask_matrix is None:
        mask_matrix = None
    else:
        mask_matrix = erode_narr_mask(
            narr_mask_matrix=training_mask_matrix + 0,
            neigh_distance_metres=neigh_distance_metres
        )

        orig_num_unmasked_pts = numpy.sum(training_mask_matrix == 1)
        num_unmasked_grid_pts = numpy.sum(mask_matrix == 1)

        print((
            'Number of unmasked grid points for training = {0:d} ... for '
            'neighbourhood evaluation = {1:d}'
        ).format(
            orig_num_unmasked_pts, num_unmasked_grid_pts
        ))

    binary_ct_as_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: 0,
        NUM_PREDICTION_ORIENTED_TP_KEY: 0,
        NUM_FALSE_POSITIVES_KEY: 0,
        NUM_FALSE_NEGATIVES_KEY: 0
    }

    prediction_oriented_ct_matrix = numpy.full((NUM_CLASSES, NUM_CLASSES), 0.)
    prediction_oriented_ct_matrix[0, :] = numpy.nan
    actual_oriented_ct_matrix = numpy.full((NUM_CLASSES, NUM_CLASSES), 0.)
    actual_oriented_ct_matrix[:, 0] = numpy.nan

    num_times = predicted_label_matrix.shape[0]

    for i in range(num_times):
        print((
            'Matching actual WF grid points at {0:d}th of {1:d} times, with '
            '{2:.1f}-km neigh distance...'
        ).format(
            i + 1, num_times, neigh_distance_metres * METRES_TO_KM
        ))

        this_actual_label_matrix = actual_label_matrix[i, ...] + 0
        if mask_matrix is not None:
            this_actual_label_matrix[mask_matrix == 0] = (
                front_utils.NO_FRONT_ENUM
            )

        these_num_predicted = _match_actual_wf_one_time(
            predicted_label_matrix=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        )

        actual_oriented_ct_matrix[:, front_utils.WARM_FRONT_ENUM] = (
            actual_oriented_ct_matrix[:, front_utils.WARM_FRONT_ENUM] +
            these_num_predicted
        )
        binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY] += (
            these_num_predicted[front_utils.WARM_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY] += (
            numpy.sum(these_num_predicted) -
            these_num_predicted[front_utils.WARM_FRONT_ENUM]
        )

        print((
            'Matching actual CF grid points at {0:d}th of {1:d} times, with '
            '{2:.1f}-km neigh distance...'
        ).format(
            i + 1, num_times, neigh_distance_metres * METRES_TO_KM
        ))

        these_num_predicted = _match_actual_cf_one_time(
            predicted_label_matrix=predicted_label_matrix[i, ...] + 0,
            actual_label_matrix=this_actual_label_matrix,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        )

        actual_oriented_ct_matrix[:, front_utils.COLD_FRONT_ENUM] = (
            actual_oriented_ct_matrix[:, front_utils.COLD_FRONT_ENUM] +
            these_num_predicted
        )
        binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY] += (
            these_num_predicted[front_utils.COLD_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY] += (
            numpy.sum(these_num_predicted) -
            these_num_predicted[front_utils.COLD_FRONT_ENUM]
        )

        print((
            'Matching predicted WF grid points at {0:d}th of {1:d} times, with '
            '{2:.1f}-km neigh distance...'
        ).format(
            i + 1, num_times, neigh_distance_metres * METRES_TO_KM
        ))

        this_predicted_label_matrix = predicted_label_matrix[i, ...] + 0
        if mask_matrix is not None:
            this_predicted_label_matrix[mask_matrix == 0] = (
                front_utils.NO_FRONT_ENUM
            )

        these_num_actual = _match_predicted_wf_one_time(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        )

        prediction_oriented_ct_matrix[front_utils.WARM_FRONT_ENUM, :] = (
            prediction_oriented_ct_matrix[front_utils.WARM_FRONT_ENUM, :] +
            these_num_actual
        )
        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY] += (
            these_num_actual[front_utils.WARM_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] += (
            numpy.sum(these_num_actual) -
            these_num_actual[front_utils.WARM_FRONT_ENUM]
        )

        print((
            'Matching predicted CF grid points at {0:d}th of {1:d} times, with '
            '{2:.1f}-km neigh distance...'
        ).format(
            i + 1, num_times, neigh_distance_metres * METRES_TO_KM
        ))

        these_num_actual = _match_predicted_cf_one_time(
            predicted_label_matrix=this_predicted_label_matrix,
            actual_label_matrix=actual_label_matrix[i, ...] + 0,
            neigh_distance_metres=neigh_distance_metres,
            grid_spacing_metres=grid_spacing_metres
        )

        prediction_oriented_ct_matrix[front_utils.COLD_FRONT_ENUM, :] = (
            prediction_oriented_ct_matrix[front_utils.COLD_FRONT_ENUM, :] +
            these_num_actual
        )
        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY] += (
            these_num_actual[front_utils.COLD_FRONT_ENUM]
        )
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] += (
            numpy.sum(these_num_actual) -
            these_num_actual[front_utils.COLD_FRONT_ENUM]
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

    :param prediction_oriented_ct_matrix: See doc for
        `_check_3class_contingency_tables`.
    :param actual_oriented_ct_matrix: Same.
    :return: prediction_oriented_ct_matrix: Normalized version of input.
    :return: actual_oriented_ct_matrix: Normalized version of input.
    """

    _check_3class_contingency_tables(
        prediction_oriented_ct_matrix=prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix=actual_oriented_ct_matrix,
        expect_normalized=False)

    for k in range(1, NUM_CLASSES):
        if numpy.sum(prediction_oriented_ct_matrix[k, :]) == 0:
            prediction_oriented_ct_matrix[k, :] = numpy.nan
        else:
            prediction_oriented_ct_matrix[k, :] = (
                prediction_oriented_ct_matrix[k, :] /
                numpy.sum(prediction_oriented_ct_matrix[k, :])
            )

    for k in range(1, NUM_CLASSES):
        if numpy.sum(actual_oriented_ct_matrix[:, k]) == 0:
            actual_oriented_ct_matrix[:, k] = numpy.nan
        else:
            actual_oriented_ct_matrix[:, k] = (
                actual_oriented_ct_matrix[:, k] /
                numpy.sum(actual_oriented_ct_matrix[:, k])
            )

    return prediction_oriented_ct_matrix, actual_oriented_ct_matrix


def get_pod(binary_ct_as_dict):
    """Computes POD (probability of detection).

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: pod: Probability of detection.
    """

    numerator = binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY]
    denominator = (
        binary_ct_as_dict[NUM_ACTUAL_ORIENTED_TP_KEY] +
        binary_ct_as_dict[NUM_FALSE_NEGATIVES_KEY]
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_far(binary_ct_as_dict):
    """Computes FAR (false-alarm ratio).

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: far: False-alarm ratio.
    """

    numerator = binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY]
    denominator = (
        binary_ct_as_dict[NUM_FALSE_POSITIVES_KEY] +
        binary_ct_as_dict[NUM_PREDICTION_ORIENTED_TP_KEY]
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_csi(binary_ct_as_dict, far_weight=1.):
    """Computes CSI (critical success index).

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :param far_weight: Weight for FAR.  Make this < 1 to penalize false alarms
        less.
    :return: CSI: Critical success index.
    """

    error_checking.assert_is_leq(far_weight, 1.)

    pod = get_pod(binary_ct_as_dict)
    success_ratio = 1. - far_weight * get_far(binary_ct_as_dict)

    try:
        return (pod ** -1 + success_ratio ** -1 - 1) ** -1
    except ZeroDivisionError:
        return numpy.nan


def get_frequency_bias(binary_ct_as_dict):
    """Computes frequency bias.

    :param binary_ct_as_dict: See doc for `make_contingency_tables`.
    :return: frequency_bias: Frequency bias.
    """

    pod = get_pod(binary_ct_as_dict)
    success_ratio = 1. - get_far(binary_ct_as_dict)

    try:
        return pod / success_ratio
    except ZeroDivisionError:
        return numpy.nan


def write_results(
        pickle_file_name, prediction_file_names, neigh_distance_metres,
        list_of_binary_ct_dicts, prediction_oriented_ct_matrix,
        actual_oriented_ct_matrix):
    """Writes results of neighbourhood evaluation to Pickle file.

    B = number of bootstrap replicates
    K = number of classes = 3

    :param pickle_file_name: Path to output file.
    :param prediction_file_names: 1-D list of paths to input files (readable by
        `prediction_io.read_file`).
    :param neigh_distance_metres: Neighbourhood distance for matching actual
        with predicted frontal grid points.
    :param list_of_binary_ct_dicts: length-B list of binary contingency tables,
        each created by `make_contingency_tables`.
    :param prediction_oriented_ct_matrix: B-by-3-by-3 numpy array, where
        prediction_oriented_ct_matrix[k, ...] is the unnormalized prediction-
        oriented contingency table for the [k]th bootstrap replicate, created by
        `make_contingency_tables`.
    :param actual_oriented_ct_matrix: B-by-3-by-3 numpy array, where
        actual_oriented_ct_matrix[k, ...] is the unnormalized actual-oriented
        contingency table for the [k]th bootstrap replicate, created by
        `make_contingency_tables`.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(prediction_file_names), num_dimensions=1
    )

    error_checking.assert_is_list(list_of_binary_ct_dicts)
    num_bootstrap_reps = len(list_of_binary_ct_dicts)

    these_expected_dim = numpy.array(
        [num_bootstrap_reps, NUM_CLASSES, NUM_CLASSES], dtype=int
    )
    error_checking.assert_is_numpy_array(
        prediction_oriented_ct_matrix, exact_dimensions=these_expected_dim
    )
    error_checking.assert_is_numpy_array(
        actual_oriented_ct_matrix, exact_dimensions=these_expected_dim
    )

    for k in range(num_bootstrap_reps):
        _check_3class_contingency_tables(
            prediction_oriented_ct_matrix=prediction_oriented_ct_matrix[k, ...],
            actual_oriented_ct_matrix=actual_oriented_ct_matrix[k, ...],
            expect_normalized=False
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
    :return: evaluation_dict: Dictionary with the following keys.
    evaluation_dict["prediction_file_names"]: See doc for `write_results`.
    evaluation_dict["neigh_distance_metres"]: Same.
    evaluation_dict["list_of_binary_ct_dicts"]: Same.
    evaluation_dict["prediction_oriented_ct_matrix"]: Same.
    evaluation_dict["actual_oriented_ct_matrix"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
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
