"""Methods for object-based evaluation of a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of images
M = number of rows in grid (unique y-coordinates at grid points)
N = number of columns in grid (unique x-coordinates at grid points)
K = number of classes (possible target values)
"""

import numpy
import pandas
import skimage.measure
import skimage.morphology
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

NARR_X_GRID_SPACING_METRES, NARR_Y_GRID_SPACING_METRES = (
    nwp_model_utils.get_xy_grid_spacing(
        model_name=nwp_model_utils.NARR_MODEL_NAME))

ROW_INDICES_COLUMN = 'row_indices'
COLUMN_INDICES_COLUMN = 'column_indices'

MIN_REGION_LENGTH_METRES = 5e5


def _check_prediction_images(prediction_matrix, probabilistic):
    """Checks prediction images for errors.

    :param prediction_matrix:
        [if probabilistic = True] E-by-M-by-N-by-K numpy array of floats.
        class_probability_matrix[i, j, k, m] is the predicted probability that
        pixel [j, k] in the [i]th image belongs to the [m]th class.

        [if probabilistic = False] E-by-M-by-N numpy array of integers.
        predicted_label_matrix[i, j, k] is the predicted class at pixel [j, k]
        in the [i]th image.

    :param probabilistic: Boolean flag.  If True, this method will expect
        probabilistic predictions.  If False, this method will expect
        deterministic predictions.
    """

    if probabilistic:
        error_checking.assert_is_numpy_array(
            prediction_matrix, num_dimensions=4)
        error_checking.assert_is_geq_numpy_array(prediction_matrix, 0.)
        error_checking.assert_is_leq_numpy_array(prediction_matrix, 1.)

        num_classes = prediction_matrix.shape[-1]
        error_checking.assert_is_geq(num_classes, 2)
        error_checking.assert_is_leq(num_classes, 3)

    else:
        error_checking.assert_is_integer_numpy_array(prediction_matrix)
        error_checking.assert_is_numpy_array(
            prediction_matrix, num_dimensions=3)
        error_checking.assert_is_geq_numpy_array(
            prediction_matrix, numpy.min(front_utils.VALID_INTEGER_IDS))
        error_checking.assert_is_leq_numpy_array(
            prediction_matrix, numpy.max(front_utils.VALID_INTEGER_IDS))


def _one_region_to_binary_image(
        row_indices_in_region, column_indices_in_region, num_grid_rows,
        num_grid_columns):
    """Converts one region to a binary image.

    P = number of points in region

    :param row_indices_in_region: length-P numpy array with row indices
        (integers) of grid cells in region.
    :param column_indices_in_region: Same as above, except for columns.
    :param num_grid_rows: M in discussion at top of file.
    :param num_grid_columns: N in discussion at top of file.
    :return: binary_image_matrix: M-by-N numpy array of integers (0 or 1).  If
        binary_image_matrix[i, j] = 1, pixel [i, j] is part of the region.
    """

    binary_image_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int)
    binary_image_matrix[row_indices_in_region, column_indices_in_region] = 1
    return binary_image_matrix


def _one_binary_image_to_region(binary_image_matrix):
    """Converts one binary image to a region.

    P = number of points in region

    :param binary_image_matrix: M-by-N numpy array of integers (0 or 1).  If
        binary_image_matrix[i, j] = 1, pixel [i, j] is part of the region.
    :return: row_indices_in_region: length-P numpy array with row indices
        (integers) of grid cells in region.
    :return: column_indices_in_region: Same as above, except for columns.
    """

    return numpy.where(binary_image_matrix)


def _get_length_of_bounding_box_diagonal(
        row_indices_in_region, column_indices_in_region, x_grid_spacing_metres,
        y_grid_spacing_metres):
    """Returns length of diagonal across bounding box of region.

    P = number of points in region

    :param row_indices_in_region: length-P numpy array with row indices
        (integers) of grid cells in region.
    :param column_indices_in_region: Same as above, except for columns.
    :param x_grid_spacing_metres: Spacing between adjacent grid points in the
        same row (same y-coord, different x-coords).
    :param y_grid_spacing_metres: Spacing between adjacent grid points in the
        same column (same x-coord, different y-coords).
    :return: diagonal_length_metres: As advertised.
    """

    x_distance_metres = x_grid_spacing_metres * (
        numpy.max(column_indices_in_region) -
        numpy.min(column_indices_in_region))
    y_distance_metres = y_grid_spacing_metres * (
        numpy.max(row_indices_in_region) - numpy.min(row_indices_in_region))

    return numpy.sqrt(x_distance_metres ** 2 + y_distance_metres ** 2)


def determinize_probabilities(class_probability_matrix, binarization_threshold):
    """Determinizes probabilistic predictions.

    :param class_probability_matrix: E-by-M-by-N-by-K numpy array of floats.
        class_probability_matrix[i, j, k, m] is the predicted probability that
        pixel [j, k] in the [i]th image belongs to the [m]th class.
    :param binarization_threshold: Threshold for discriminating between front
        and no front.  For details, see
        `evaluation_utils.find_best_binarization_threshold`.
    :return: predicted_label_matrix: E-by-M-by-N numpy array of integers.
        predicted_label_matrix[i, j, k] is the predicted class at pixel [j, k]
        in the [i]th image.
    """

    _check_prediction_images(
        prediction_matrix=class_probability_matrix, probabilistic=True)
    error_checking.assert_is_geq(binarization_threshold, 0.)
    error_checking.assert_is_leq(binarization_threshold, 1.)

    predicted_label_matrix = 1 + numpy.argmax(
        class_probability_matrix[..., 1:], axis=-1)

    no_front_matrix = class_probability_matrix[..., 0] >= binarization_threshold
    no_front_indices_as_tuple = numpy.where(no_front_matrix)
    predicted_label_matrix[no_front_indices_as_tuple] = 0

    return predicted_label_matrix


def images_to_regions(predicted_label_matrix, image_times_unix_sec):
    """Converts each image of predicted labels to a list of regions.

    P = number of grid points in a given region

    :param predicted_label_matrix: E-by-M-by-N numpy array of integers.
        predicted_label_matrix[i, j, k] is the predicted class at pixel [j, k]
        in the [i]th image.
    :param image_times_unix_sec: length-E numpy array of valid times.
    :return: predicted_region_table: pandas DataFrame with the following
        columns.  Each row is one predicted region (either cold-frontal or warm-
        frontal region).
    predicted_region_table.unix_time_sec: Valid time.
    predicted_region_table.front_type: Front type (either "warm" or "cold").
    predicted_region_table.row_indices: length-P numpy array with row indices
        (integers) of grid cells in frontal region.
    predicted_region_table.column_indices: Same as above, except for columns.
    """

    _check_prediction_images(
        prediction_matrix=predicted_label_matrix, probabilistic=False)

    region_times_unix_sec = []
    front_types = []
    row_indices_by_region = []
    column_indices_by_region = []

    num_images = predicted_label_matrix.shape[0]
    for i in range(num_images):
        this_region_matrix = skimage.measure.label(
            predicted_label_matrix[i, ...], connectivity=2)

        this_num_regions = numpy.max(this_region_matrix)
        for j in range(this_num_regions):
            these_row_indices, these_column_indices = numpy.where(
                this_region_matrix == j + 1)
            row_indices_by_region.append(numpy.array(these_row_indices))
            column_indices_by_region.append(numpy.array(these_column_indices))

            region_times_unix_sec.append(image_times_unix_sec[i])
            this_integer_front_id = predicted_label_matrix[
                i, these_row_indices[0], these_column_indices[0]]

            if this_integer_front_id == front_utils.WARM_FRONT_INTEGER_ID:
                front_types.append(front_utils.WARM_FRONT_STRING_ID)
            else:
                front_types.append(front_utils.COLD_FRONT_STRING_ID)

    predicted_region_dict = {
        front_utils.TIME_COLUMN: region_times_unix_sec,
        front_utils.FRONT_TYPE_COLUMN: front_types,
        ROW_INDICES_COLUMN: row_indices_by_region,
        COLUMN_INDICES_COLUMN: column_indices_by_region
    }
    return pandas.DataFrame.from_dict(predicted_region_dict)


def thin_frontal_regions(
        predicted_region_table, num_grid_rows, num_grid_columns):
    """Thins out frontal regions.

    This makes frontal regions look more like polylines (with infinitesimal
    width), which is the way that humans usually think of fronts.  This makes
    frontal regions more easily comparable to the human labels (polylines),
    which we consider as "ground truth".

    :param predicted_region_table: pandas DataFrame created by
        `images_to_regions`.
    :param num_grid_rows: M in discussion at top of file.
    :param num_grid_columns: N in discussion at top of file.
    :return: predicted_region_table: Same as input, but with thinner regions.
    """

    num_regions = len(predicted_region_table.index)
    for i in range(num_regions):
        this_binary_image_matrix = _one_region_to_binary_image(
            row_indices_in_region=
            predicted_region_table[ROW_INDICES_COLUMN].values[i],
            column_indices_in_region=
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i],
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

        this_binary_image_matrix = skimage.morphology.medial_axis(
            this_binary_image_matrix)

        (predicted_region_table[ROW_INDICES_COLUMN].values[i],
         predicted_region_table[COLUMN_INDICES_COLUMN].values[i]) = (
             _one_binary_image_to_region(this_binary_image_matrix))

    return predicted_region_table


def discard_small_regions(
        predicted_region_table,
        min_bounding_box_diag_length_metres=MIN_REGION_LENGTH_METRES):
    """Throws out small frontal regions.

    :param predicted_region_table: pandas DataFrame created by
        `thin_frontal_regions`.
    :param min_bounding_box_diag_length_metres: Minimum length of diagonal
        through bounding box.  Any frontal region with a smaller length will be
        thrown out.
    :return: predicted_region_table: Same as input, but maybe with fewer rows.
    """

    error_checking.assert_is_greater(min_bounding_box_diag_length_metres, 0.)

    num_regions = len(predicted_region_table.index)
    rows_to_drop = []

    for i in range(num_regions):
        this_length_metres = _get_length_of_bounding_box_diagonal(
            row_indices_in_region=
            predicted_region_table[ROW_INDICES_COLUMN].values[i],
            column_indices_in_region=
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i],
            x_grid_spacing_metres=NARR_X_GRID_SPACING_METRES,
            y_grid_spacing_metres=NARR_Y_GRID_SPACING_METRES)

        if this_length_metres < min_bounding_box_diag_length_metres:
            rows_to_drop.append(i)

    return predicted_region_table.drop(
        predicted_region_table.index[rows_to_drop], axis=0, inplace=False)
