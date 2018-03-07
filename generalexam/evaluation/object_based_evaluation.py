"""Methods for object-based evaluation of a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of images
M = number of rows in grid (unique y-coordinates at grid points)
N = number of columns in grid (unique x-coordinates at grid points)
K = number of classes (possible target values)
"""

import copy
import pickle
import cv2
import numpy
import pandas
import skimage.measure
import skimage.morphology
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import a_star_search
from generalexam.machine_learning import machine_learning_utils as ml_utils

ROW_INDICES_COLUMN = 'row_indices'
COLUMN_INDICES_COLUMN = 'column_indices'
X_COORDS_COLUMN = 'x_coords_metres'
Y_COORDS_COLUMN = 'y_coords_metres'

DEFAULT_MIN_REGION_LENGTH_METRES = 5e5  # 500 km
DEFAULT_MIN_REGION_AREA_METRES2 = 2e11  # 0.2 million km^2

NUM_ACTUAL_FRONTS_PREDICTED_KEY = 'num_actual_fronts_predicted'
NUM_PREDICTED_FRONTS_VERIFIED_KEY = 'num_predicted_fronts_verified'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'

PREDICTED_REGION_TABLE_KEY = 'predicted_region_table'
ACTUAL_POLYLINE_TABLE_KEY = 'actual_polyline_table'
NEIGH_DISTANCE_METRES_KEY = 'neigh_distance_metres'
BINARY_CONTINGENCY_TABLE_KEY = 'binary_contingency_table_as_dict'
BINARY_POD_KEY = 'binary_pod'
BINARY_SUCCESS_RATIO_KEY = 'binary_success_ratio'
BINARY_CSI_KEY = 'binary_csi'
BINARY_FREQUENCY_BIAS_KEY = 'binary_frequency_bias'
ROW_NORMALIZED_CONTINGENCY_TABLE_KEY = 'row_normalized_ct_as_matrix'
COLUMN_NORMALIZED_CONTINGENCY_TABLE_KEY = 'column_normalized_ct_as_matrix'

KERNEL_MATRIX_FOR_ENDPOINT_FILTER = numpy.array([[1, 1, 1],
                                                 [1, 10, 1],
                                                 [1, 1, 1]], dtype=numpy.uint8)
FILTERED_VALUE_AT_ENDPOINT = 11


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


def _get_distance_between_fronts(
        first_x_coords_metres, first_y_coords_metres, second_x_coords_metres,
        second_y_coords_metres):
    """Returns distance between two fronts.

    Specifically, this method returns the median distance from a point in the
    first front to the nearest point in the second front.

    P1 = number of points in first front
    P2 = number of points in second front

    :param first_x_coords_metres: numpy array (length P1) of x-coordinates in
        first front.
    :param first_y_coords_metres: numpy array (length P1) of y-coordinates in
        first front.
    :param second_x_coords_metres: numpy array (length P2) of x-coordinates in
        second front.
    :param second_y_coords_metres: numpy array (length P2) of y-coordinates in
        second front.
    :return: median_shortest_distance_metres: Inter-front distance.
    """

    num_points_in_first_front = len(first_x_coords_metres)
    shortest_distances_metres = numpy.full(num_points_in_first_front, numpy.nan)

    for i in range(num_points_in_first_front):
        shortest_distances_metres[i] = numpy.min(numpy.sqrt(
            (first_x_coords_metres[i] - second_x_coords_metres) ** 2 +
            (first_y_coords_metres[i] - second_y_coords_metres) ** 2))

    return numpy.median(shortest_distances_metres)


def _find_endpoints_of_skeleton(binary_image_matrix):
    """Finds endpoints of skeleton.

    :param binary_image_matrix: M-by-N numpy array of integers in 0...1.  If
        binary_image_matrix[i, j] = 1, grid cell [i, j] is part of the skeleton.
    :return: binary_endpoint_matrix: M-by-N numpy array of integers in 0...1.
        If binary_endpoint_matrix[i, j] = 1, grid cell [i, j] is an endpoint of
        the skeleton.
    """

    if numpy.sum(binary_image_matrix) == 1:
        return copy.deepcopy(binary_image_matrix)

    filtered_image_matrix = numpy.pad(
        binary_image_matrix, pad_width=2, mode='constant', constant_values=0)

    filtered_image_matrix = cv2.filter2D(
        filtered_image_matrix.astype(numpy.uint8), -1,
        KERNEL_MATRIX_FOR_ENDPOINT_FILTER)
    filtered_image_matrix = filtered_image_matrix[2:-2, 2:-2]

    endpoint_flag_matrix = numpy.full(binary_image_matrix.shape, 0, dtype=int)
    endpoint_flag_matrix[
        filtered_image_matrix == FILTERED_VALUE_AT_ENDPOINT] = 1
    return endpoint_flag_matrix


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

    num_images = predicted_label_matrix.shape[0]
    error_checking.assert_is_integer_numpy_array(image_times_unix_sec)
    error_checking.assert_is_numpy_array(
        image_times_unix_sec, exact_dimensions=numpy.array([num_images]))

    region_times_unix_sec = []
    front_types = []
    row_indices_by_region = []
    column_indices_by_region = []

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


def regions_to_images(predicted_region_table, num_grid_rows, num_grid_columns):
    """Converts frontal regions to images (one image per time step).

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param num_grid_rows: M in discussion at top of file.
    :param num_grid_columns: N in discussion at top of file.
    :return: predicted_label_matrix: E-by-M-by-N numpy array of integers.
        predicted_label_matrix[i, j, k] is the predicted class at pixel [j, k]
        in the [i]th image.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    image_times_unix_sec = numpy.unique(
        predicted_region_table[front_utils.TIME_COLUMN].values)
    num_image_times = len(image_times_unix_sec)

    predicted_label_matrix = numpy.full(
        (num_image_times, num_grid_rows, num_grid_columns), -1, dtype=int)

    for i in range(num_image_times):
        this_time_indices = numpy.where(
            predicted_region_table[front_utils.TIME_COLUMN].values ==
            image_times_unix_sec[i])[0]

        these_warm_front_row_indices = numpy.array([], dtype=int)
        these_warm_front_column_indices = numpy.array([], dtype=int)
        these_cold_front_row_indices = numpy.array([], dtype=int)
        these_cold_front_column_indices = numpy.array([], dtype=int)

        for j in this_time_indices:
            this_front_type = predicted_region_table[
                front_utils.FRONT_TYPE_COLUMN].values[j]

            if this_front_type == front_utils.WARM_FRONT_STRING_ID:
                these_warm_front_row_indices = numpy.concatenate((
                    these_warm_front_row_indices,
                    predicted_region_table[ROW_INDICES_COLUMN].values[j]))
                these_warm_front_column_indices = numpy.concatenate((
                    these_warm_front_column_indices,
                    predicted_region_table[COLUMN_INDICES_COLUMN].values[j]))

            else:
                these_cold_front_row_indices = numpy.concatenate((
                    these_cold_front_row_indices,
                    predicted_region_table[ROW_INDICES_COLUMN].values[j]))
                these_cold_front_column_indices = numpy.concatenate((
                    these_cold_front_column_indices,
                    predicted_region_table[COLUMN_INDICES_COLUMN].values[j]))

        this_frontal_grid_point_dict = {
            front_utils.WARM_FRONT_ROW_INDICES_COLUMN:
                these_warm_front_row_indices,
            front_utils.WARM_FRONT_COLUMN_INDICES_COLUMN:
                these_warm_front_column_indices,
            front_utils.COLD_FRONT_ROW_INDICES_COLUMN:
                these_cold_front_row_indices,
            front_utils.COLD_FRONT_COLUMN_INDICES_COLUMN:
                these_cold_front_column_indices
        }

        predicted_label_matrix[i, ...] = (
            front_utils.frontal_grid_points_to_image(
                frontal_grid_point_dict=this_frontal_grid_point_dict,
                num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns))

    return predicted_label_matrix


def skeletonize_frontal_regions(
        predicted_region_table, num_grid_rows, num_grid_columns):
    """Skeletonizes ("thins out") frontal regions.

    This makes frontal regions look more like polylines (with infinitesimal
    width), which is how humans usually think of fronts (and also how the
    verification data, or "ground truth," are formatted).

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param num_grid_rows: M in discussion at top of file.
    :param num_grid_columns: N in discussion at top of file.
    :return: predicted_region_table: Same as input, but with thinner regions.
    """

    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_greater(num_grid_columns, 0)

    num_regions = len(predicted_region_table.index)
    for i in range(num_regions):
        this_binary_image_matrix = _one_region_to_binary_image(
            row_indices_in_region=
            predicted_region_table[ROW_INDICES_COLUMN].values[i],
            column_indices_in_region=
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i],
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

        this_binary_image_matrix = skimage.morphology.thin(
            this_binary_image_matrix).astype(int)

        (predicted_region_table[ROW_INDICES_COLUMN].values[i],
         predicted_region_table[COLUMN_INDICES_COLUMN].values[i]) = (
             _one_binary_image_to_region(this_binary_image_matrix))

    return predicted_region_table


def find_main_skeletons(
        predicted_region_table, class_probability_matrix, image_times_unix_sec):
    """Converts each (already skeletonized) frontal region to its main skeleton.

    The "main skeleton" is a simple polyline***, whereas the original skeleton
    is usually a complex polyline (with > 2 endpoints).  In other words, this
    method removes "branches" from the original skeleton line, leaving only the
    main skeleton line.

    *** This method represents each skeleton line as a polygon with
        one-grid-cell width, rather than an explicit polyline.

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param class_probability_matrix: E-by-M-by-N-by-K numpy array of floats.
        class_probability_matrix[i, j, k, m] is the predicted probability that
        pixel [j, k] in the [i]th image belongs to the [m]th class.
    :param image_times_unix_sec: length-E numpy array of valid times.
    :return: predicted_region_table: Same as input, except that each region has
        been reduced to its main skeleton line.
    """

    _check_prediction_images(class_probability_matrix, probabilistic=True)
    num_images = class_probability_matrix.shape[0]
    error_checking.assert_is_integer_numpy_array(image_times_unix_sec)
    error_checking.assert_is_numpy_array(
        image_times_unix_sec, exact_dimensions=numpy.array([num_images]))

    num_grid_rows = class_probability_matrix.shape[1]
    num_grid_columns = class_probability_matrix.shape[2]
    num_regions = len(predicted_region_table.index)

    for i in range(num_regions):
        print 'Finding main skeleton for {0:d}th of {1:d} regions...'.format(
            i + 1, num_regions)

        this_image_index = numpy.where(
            image_times_unix_sec ==
            predicted_region_table[front_utils.TIME_COLUMN].values[i])[0]
        this_front_type_integer = front_utils.string_id_to_integer(
            predicted_region_table[front_utils.FRONT_TYPE_COLUMN].values[i])

        this_binary_region_matrix = _one_region_to_binary_image(
            row_indices_in_region=
            predicted_region_table[ROW_INDICES_COLUMN].values[i],
            column_indices_in_region=
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i],
            num_grid_rows=num_grid_rows, num_grid_columns=num_grid_columns)

        this_binary_endpoint_matrix = _find_endpoints_of_skeleton(
            this_binary_region_matrix)
        these_endpoint_rows, these_endpoint_columns = numpy.where(
            this_binary_endpoint_matrix == 1)
        this_num_endpoints = len(these_endpoint_rows)

        if this_num_endpoints == 1:
            these_main_skeleton_rows = numpy.array([these_endpoint_rows[0]])
            these_main_skeleton_columns = numpy.array(
                [these_endpoint_columns[0]])

            this_max_mean_probability = class_probability_matrix[
                this_image_index, these_main_skeleton_rows[0],
                these_main_skeleton_columns[0], this_front_type_integer]

        else:
            this_max_mean_probability = 0.
            these_main_skeleton_rows = numpy.array([], dtype=int)
            these_main_skeleton_columns = numpy.array([], dtype=int)

        for j in range(this_num_endpoints):
            for k in range(j + 1, this_num_endpoints):
                this_grid_search_object = a_star_search.GridSearch(
                    binary_region_matrix=this_binary_region_matrix)

                these_skeleton_rows, these_skeleton_columns = (
                    a_star_search.run_a_star(
                        grid_search_object=this_grid_search_object,
                        start_row=these_endpoint_rows[j],
                        start_column=these_endpoint_columns[j],
                        end_row=these_endpoint_rows[k],
                        end_column=these_endpoint_columns[k]))

                if these_skeleton_rows is None:
                    continue

                this_mean_probability = numpy.mean(
                    class_probability_matrix[
                        this_image_index, these_skeleton_rows,
                        these_skeleton_columns, this_front_type_integer])
                if this_mean_probability <= this_max_mean_probability:
                    continue

                this_max_mean_probability = this_mean_probability + 0.
                these_main_skeleton_rows = these_skeleton_rows + 0
                these_main_skeleton_columns = these_skeleton_columns + 0

        predicted_region_table[ROW_INDICES_COLUMN].values[
            i] = these_main_skeleton_rows
        predicted_region_table[COLUMN_INDICES_COLUMN].values[
            i] = these_main_skeleton_columns

    return predicted_region_table


def discard_regions_with_small_length(
        predicted_region_table, x_grid_spacing_metres, y_grid_spacing_metres,
        min_bounding_box_diag_length_metres=DEFAULT_MIN_REGION_LENGTH_METRES):
    """Throws out frontal regions with small length.

    The "length" of a region is defined as the length of the diagonal through
    its bounding box.

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param x_grid_spacing_metres: Spacing between adjacent grid points in the
        same row (same y-coord, different x-coords).
    :param y_grid_spacing_metres: Spacing between adjacent grid points in the
        same column (same x-coord, different y-coords).
    :param min_bounding_box_diag_length_metres: Minimum length.  Any region with
        a smaller length will be thrown out.
    :return: predicted_region_table: Same as input, but maybe with fewer rows.
    """

    error_checking.assert_is_greater(x_grid_spacing_metres, 0.)
    error_checking.assert_is_greater(y_grid_spacing_metres, 0.)
    error_checking.assert_is_greater(min_bounding_box_diag_length_metres, 0.)

    num_regions = len(predicted_region_table.index)
    rows_to_drop = []

    for i in range(num_regions):
        this_length_metres = _get_length_of_bounding_box_diagonal(
            row_indices_in_region=
            predicted_region_table[ROW_INDICES_COLUMN].values[i],
            column_indices_in_region=
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i],
            x_grid_spacing_metres=x_grid_spacing_metres,
            y_grid_spacing_metres=y_grid_spacing_metres)

        if this_length_metres < min_bounding_box_diag_length_metres:
            rows_to_drop.append(i)

    return predicted_region_table.drop(
        predicted_region_table.index[rows_to_drop], axis=0, inplace=False)


def discard_regions_with_small_area(
        predicted_region_table, x_grid_spacing_metres, y_grid_spacing_metres,
        min_area_metres2=DEFAULT_MIN_REGION_AREA_METRES2):
    """Throws out frontal regions with small area.

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param x_grid_spacing_metres: Spacing between adjacent grid points in the
        same row (same y-coord, different x-coords).
    :param y_grid_spacing_metres: Spacing between adjacent grid points in the
        same column (same x-coord, different y-coords).
    :param min_area_metres2: Minimum area.  Any region with a smaller area will
        be thrown out.
    :return: predicted_region_table: Same as input, but maybe with fewer rows.
    """

    error_checking.assert_is_greater(min_area_metres2, 0.)
    error_checking.assert_is_greater(x_grid_spacing_metres, 0.)
    error_checking.assert_is_greater(y_grid_spacing_metres, 0.)

    grid_cell_area_metres2 = x_grid_spacing_metres * y_grid_spacing_metres
    min_grid_cells_in_region = int(numpy.round(
        min_area_metres2 / grid_cell_area_metres2))

    num_regions = len(predicted_region_table.index)
    rows_to_drop = []

    for i in range(num_regions):
        this_num_grid_cells = len(
            predicted_region_table[ROW_INDICES_COLUMN].values[i])
        if this_num_grid_cells < min_grid_cells_in_region:
            rows_to_drop.append(i)

    return predicted_region_table.drop(
        predicted_region_table.index[rows_to_drop], axis=0, inplace=False)


def project_polylines_latlng_to_narr(polyline_table):
    """Projects frontal polylines from lat-long to NARR coordinates.

    V = number of vertices in a given polyline

    :param polyline_table: pandas DataFrame with the following columns.  Each
        row is a single front.
    polyline_table.front_type: Either "warm" or "cold".
    polyline_table.unix_time_sec: Valid time.
    polyline_table.latitudes_deg: length-V numpy array of latitudes (deg N).
    polyline_table.longitudes_deg: length-V numpy array of longitudes (deg E).

    :return: polyline_table: Same as input, but with extra columns listed below.
    polyline_table.x_coords_metres: length-V numpy array of x-coordinates.
    polyline_table.y_coords_metres: length-V numpy array of y-coordinates.
    """

    num_fronts = len(polyline_table.index)
    x_coords_by_front_metres = [numpy.array([])] * num_fronts
    y_coords_by_front_metres = [numpy.array([])] * num_fronts

    for i in range(num_fronts):
        x_coords_by_front_metres[i], y_coords_by_front_metres[i] = (
            nwp_model_utils.project_latlng_to_xy(
                latitudes_deg=polyline_table[
                    front_utils.LATITUDES_COLUMN].values[i],
                longitudes_deg=polyline_table[
                    front_utils.LONGITUDES_COLUMN].values[i],
                model_name=nwp_model_utils.NARR_MODEL_NAME))

    argument_dict = {
        X_COORDS_COLUMN: x_coords_by_front_metres,
        Y_COORDS_COLUMN: y_coords_by_front_metres
    }
    return polyline_table.assign(**argument_dict)


def convert_regions_rowcol_to_narr_xy(
        predicted_region_table, are_predictions_from_fcn):
    """Converts frontal regions from row-column to x-y coordinates.

    This method assumes that rows and columns are on the NARR grid.

    P = number of grid points in a given region.

    :param predicted_region_table: See documentation for `images_to_regions`.
    :param are_predictions_from_fcn: Boolean flag.  If True, predictions in
        `predicted_region_table` are from an FCN (fully convolutional network),
        in which case an offset must be applied to line up the FCN grid with the
        NARR grid.  (The FCN is trained with, and thus predicts, only a subset
        of the NARR grid.)
    :return: predicted_region_table: Same as input, but with extra columns
        listed below.
    predicted_region_table.x_coords_metres: length-P numpy array of
        x-coordinates.
    predicted_region_table.y_coords_metres: length-P numpy array of
        y-coordinates.
    """

    error_checking.assert_is_boolean(are_predictions_from_fcn)
    if are_predictions_from_fcn:
        row_offset = ml_utils.FIRST_NARR_ROW_FOR_FCN_INPUT
        column_offset = ml_utils.FIRST_NARR_COLUMN_FOR_FCN_INPUT
    else:
        row_offset = 0
        column_offset = 0

    grid_point_x_matrix_metres, grid_point_y_matrix_metres = (
        nwp_model_utils.get_xy_grid_point_matrices(
            model_name=nwp_model_utils.NARR_MODEL_NAME))

    num_regions = len(predicted_region_table.index)
    x_coords_by_region_metres = [numpy.array([], dtype=int)] * num_regions
    y_coords_by_region_metres = [numpy.array([], dtype=int)] * num_regions

    for i in range(num_regions):
        these_row_indices = row_offset + numpy.array(
            predicted_region_table[ROW_INDICES_COLUMN].values[i]).astype(int)
        these_column_indices = column_offset + numpy.array(
            predicted_region_table[COLUMN_INDICES_COLUMN].values[i]).astype(int)

        x_coords_by_region_metres[i] = grid_point_x_matrix_metres[
            these_row_indices, these_column_indices]
        y_coords_by_region_metres[i] = grid_point_y_matrix_metres[
            these_row_indices, these_column_indices]

    argument_dict = {
        X_COORDS_COLUMN: x_coords_by_region_metres,
        Y_COORDS_COLUMN: y_coords_by_region_metres
    }
    return predicted_region_table.assign(**argument_dict)


def get_binary_contingency_table(
        predicted_region_table, actual_polyline_table, neigh_distance_metres):
    """Creates binary (front vs. no front) contingency table.

    V = number of vertices in a given polyline

    :param predicted_region_table: pandas DataFrame created by
        `convert_regions_rowcol_to_narr_xy`.  Each row is one predicted front.
    :param actual_polyline_table: pandas DataFrame created by
        `project_polylines_latlng_to_narr`.  Each row is one actual front
        ("ground truth").
    :param neigh_distance_metres: Neighbourhood distance.  Will be used to
        define when actual front f_A and predicted front f_P are "matching".
    :return: binary_contingency_table_as_dict: Dictionary with the following
        keys.
    binary_contingency_table_as_dict['num_actual_fronts_predicted']: Number of
        actual fronts that were predicted (this is one kind of "true positive"
        in the table).
    binary_contingency_table_as_dict['num_predicted_fronts_verified']: Number of
        predicted fronts that actually occurred (this is another kind of "true
        positive" in the table).
    binary_contingency_table_as_dict['num_false_positives']: Number of predicted
        fronts that did not occur.
    binary_contingency_table_as_dict['num_false_negatives']: Number of actual
        fronts that were not predicted.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)

    actual_front_times_unix_sec = actual_polyline_table[
        front_utils.TIME_COLUMN].values
    predicted_front_times_unix_sec = predicted_region_table[
        front_utils.TIME_COLUMN].values

    num_actual_fronts = len(actual_polyline_table.index)
    num_actual_fronts_predicted = 0
    num_false_negatives = 0

    for i in range(num_actual_fronts):
        this_actual_front_type = actual_polyline_table[
            front_utils.FRONT_TYPE_COLUMN].values[i]
        these_front_type_flags = numpy.array(
            [s == this_actual_front_type for s in
             predicted_region_table[front_utils.FRONT_TYPE_COLUMN].values])

        these_predicted_front_indices = numpy.where(numpy.logical_and(
            predicted_front_times_unix_sec == actual_front_times_unix_sec[i],
            these_front_type_flags))[0]

        found_match = False

        for j in these_predicted_front_indices:
            this_distance_metres = _get_distance_between_fronts(
                first_x_coords_metres=
                actual_polyline_table[X_COORDS_COLUMN].values[i],
                first_y_coords_metres=
                actual_polyline_table[Y_COORDS_COLUMN].values[i],
                second_x_coords_metres=
                predicted_region_table[X_COORDS_COLUMN].values[j],
                second_y_coords_metres=
                predicted_region_table[Y_COORDS_COLUMN].values[j])

            if this_distance_metres < neigh_distance_metres:
                found_match = True
                break

        if found_match:
            num_actual_fronts_predicted += 1
        else:
            num_false_negatives += 1

    num_predicted_fronts = len(predicted_region_table.index)
    num_predicted_fronts_verified = 0
    num_false_positives = 0

    for i in range(num_predicted_fronts):
        this_predicted_front_type = predicted_region_table[
            front_utils.FRONT_TYPE_COLUMN].values[i]
        these_front_type_flags = numpy.array(
            [s == this_predicted_front_type for s in
             actual_polyline_table[front_utils.FRONT_TYPE_COLUMN].values])

        these_actual_front_indices = numpy.where(numpy.logical_and(
            actual_front_times_unix_sec == predicted_front_times_unix_sec[i],
            these_front_type_flags))[0]

        found_match = False

        for j in these_actual_front_indices:
            this_distance_metres = _get_distance_between_fronts(
                first_x_coords_metres=
                predicted_region_table[X_COORDS_COLUMN].values[i],
                first_y_coords_metres=
                predicted_region_table[Y_COORDS_COLUMN].values[i],
                second_x_coords_metres=
                actual_polyline_table[X_COORDS_COLUMN].values[j],
                second_y_coords_metres=
                actual_polyline_table[Y_COORDS_COLUMN].values[j])

            if this_distance_metres < neigh_distance_metres:
                found_match = True
                break

        if found_match:
            num_predicted_fronts_verified += 1
        else:
            num_false_positives += 1

    return {
        NUM_ACTUAL_FRONTS_PREDICTED_KEY: num_actual_fronts_predicted,
        NUM_PREDICTED_FRONTS_VERIFIED_KEY: num_predicted_fronts_verified,
        NUM_FALSE_POSITIVES_KEY: num_false_positives,
        NUM_FALSE_NEGATIVES_KEY: num_false_negatives
    }


def get_row_normalized_contingency_table(
        predicted_region_table, actual_polyline_table, neigh_distance_metres):
    """Creates "row-normalized" contingency table.

    "Row-normalized" means that each row sums to 1.

    K = number of classes

    :param predicted_region_table: pandas DataFrame created by
        `convert_regions_rowcol_to_narr_xy`.  Each row is one predicted front.
    :param actual_polyline_table: pandas DataFrame created by
        `project_polylines_latlng_to_narr`.  Each row is one actual front
        ("ground truth").
    :param neigh_distance_metres: Neighbourhood distance.  Will be used to
        define when actual front f_A and predicted front f_P are "matching".
    :return: row_normalized_ct_as_matrix: (K - 1)-by-K numpy array.
        row_normalized_ct_as_matrix[i, j] is the conditional probability, when
        the [i + 1]th class is predicted, that the [j]th class will be observed.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)

    num_predicted_fronts = len(predicted_region_table.index)
    actual_front_times_unix_sec = actual_polyline_table[
        front_utils.TIME_COLUMN].values
    predicted_front_times_unix_sec = predicted_region_table[
        front_utils.TIME_COLUMN].values

    num_classes = 1 + len(front_utils.VALID_STRING_IDS)
    row_normalized_ct_as_matrix = numpy.full(
        (num_classes, num_classes), 0, dtype=int)

    for i in range(num_predicted_fronts):
        these_actual_front_indices = numpy.where(
            actual_front_times_unix_sec == predicted_front_times_unix_sec[i])[0]
        these_distances_to_actual_metres = []

        for j in these_actual_front_indices:
            this_distance_metres = _get_distance_between_fronts(
                first_x_coords_metres=
                predicted_region_table[X_COORDS_COLUMN].values[i],
                first_y_coords_metres=
                predicted_region_table[Y_COORDS_COLUMN].values[i],
                second_x_coords_metres=
                actual_polyline_table[X_COORDS_COLUMN].values[j],
                second_y_coords_metres=
                actual_polyline_table[Y_COORDS_COLUMN].values[j])

            these_distances_to_actual_metres.append(this_distance_metres)

        this_min_distance = numpy.min(these_distances_to_actual_metres)
        this_actual_front_index = these_actual_front_indices[
            numpy.argmin(these_distances_to_actual_metres)]

        this_predicted_front_type_int = front_utils.string_id_to_integer(
            predicted_region_table[front_utils.FRONT_TYPE_COLUMN].values[i])

        if this_min_distance > neigh_distance_metres:
            row_normalized_ct_as_matrix[
                this_predicted_front_type_int,
                front_utils.NO_FRONT_INTEGER_ID] += 1
        else:
            this_actual_front_type_int = front_utils.string_id_to_integer(
                actual_polyline_table[front_utils.FRONT_TYPE_COLUMN].values[
                    this_actual_front_index])

            row_normalized_ct_as_matrix[
                this_predicted_front_type_int, this_actual_front_type_int] += 1

    row_normalized_ct_as_matrix = row_normalized_ct_as_matrix.astype(float)

    for k in range(1, num_classes):
        if numpy.sum(row_normalized_ct_as_matrix[k, :]) == 0:
            row_normalized_ct_as_matrix[k, :] = numpy.nan
        else:
            row_normalized_ct_as_matrix[k, :] = (
                row_normalized_ct_as_matrix[k, :] /
                numpy.sum(row_normalized_ct_as_matrix[k, :]))

    return row_normalized_ct_as_matrix[1:, :]


def get_column_normalized_contingency_table(
        predicted_region_table, actual_polyline_table, neigh_distance_metres):
    """Creates "column-normalized" contingency table.

    "Column-normalized" means that each column sums to 1.

    K = number of classes

    :param predicted_region_table: See documentation for
        `get_row_normalized_contingency_table`.
    :param actual_polyline_table: Same.
    :param neigh_distance_metres: Same.
    :return: column_normalized_ct_as_matrix: K-by-(K - 1) numpy array.
        column_normalized_ct_as_matrix[i, j] is the conditional probability,
        when the [j]th class is observed, that the [i]th class will be
        predicted.
    """

    error_checking.assert_is_greater(neigh_distance_metres, 0.)

    num_actual_fronts = len(actual_polyline_table.index)
    actual_front_times_unix_sec = actual_polyline_table[
        front_utils.TIME_COLUMN].values
    predicted_front_times_unix_sec = predicted_region_table[
        front_utils.TIME_COLUMN].values

    num_classes = 1 + len(front_utils.VALID_STRING_IDS)
    column_normalized_ct_as_matrix = numpy.full(
        (num_classes, num_classes), 0, dtype=int)

    for i in range(num_actual_fronts):
        these_predicted_front_indices = numpy.where(
            predicted_front_times_unix_sec == actual_front_times_unix_sec[i])[0]
        these_distances_to_predicted_metres = []

        for j in these_predicted_front_indices:
            this_distance_metres = _get_distance_between_fronts(
                first_x_coords_metres=
                actual_polyline_table[X_COORDS_COLUMN].values[i],
                first_y_coords_metres=
                actual_polyline_table[Y_COORDS_COLUMN].values[i],
                second_x_coords_metres=
                predicted_region_table[X_COORDS_COLUMN].values[j],
                second_y_coords_metres=
                predicted_region_table[Y_COORDS_COLUMN].values[j])

            these_distances_to_predicted_metres.append(this_distance_metres)

        this_min_distance = numpy.min(these_distances_to_predicted_metres)
        this_predicted_front_index = these_predicted_front_indices[
            numpy.argmin(these_distances_to_predicted_metres)]

        this_actual_front_type_int = front_utils.string_id_to_integer(
            actual_polyline_table[front_utils.FRONT_TYPE_COLUMN].values[i])

        if this_min_distance > neigh_distance_metres:
            column_normalized_ct_as_matrix[
                front_utils.NO_FRONT_INTEGER_ID,
                this_actual_front_type_int] += 1
        else:
            this_predicted_front_type_int = front_utils.string_id_to_integer(
                predicted_region_table[front_utils.FRONT_TYPE_COLUMN].values[
                    this_predicted_front_index])

            column_normalized_ct_as_matrix[
                this_predicted_front_type_int, this_actual_front_type_int] += 1

    column_normalized_ct_as_matrix = column_normalized_ct_as_matrix.astype(float)

    for k in range(1, num_classes):
        if numpy.sum(column_normalized_ct_as_matrix[:, k]) == 0:
            column_normalized_ct_as_matrix[:, k] = numpy.nan
        else:
            column_normalized_ct_as_matrix[:, k] = (
                column_normalized_ct_as_matrix[:, k] /
                numpy.sum(column_normalized_ct_as_matrix[:, k]))

    return column_normalized_ct_as_matrix[:, 1:]


def get_binary_pod(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) probability of detection.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_pod: Binary POD.
    """

    numerator = binary_contingency_table_as_dict[
        NUM_ACTUAL_FRONTS_PREDICTED_KEY]
    denominator = numerator + binary_contingency_table_as_dict[
        NUM_FALSE_NEGATIVES_KEY]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_fom(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) frequency of misses.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_fom: Binary FOM.
    """

    return 1. - get_binary_pod(binary_contingency_table_as_dict)


def get_binary_success_ratio(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) success ratio.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_success_ratio: As advertised.
    """

    numerator = binary_contingency_table_as_dict[
        NUM_PREDICTED_FRONTS_VERIFIED_KEY]
    denominator = numerator + binary_contingency_table_as_dict[
        NUM_FALSE_POSITIVES_KEY]

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def get_binary_far(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) false-alarm rate.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_far: Binary FAR.
    """

    return 1. - get_binary_success_ratio(binary_contingency_table_as_dict)


def get_binary_csi(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) critical success index.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_csi: Binary CSI.
    """

    binary_pod = get_binary_pod(binary_contingency_table_as_dict)
    binary_success_ratio = get_binary_success_ratio(
        binary_contingency_table_as_dict)

    try:
        return (binary_pod ** -1 + binary_success_ratio ** -1 - 1) ** -1
    except ZeroDivisionError:
        return numpy.nan


def get_binary_frequency_bias(binary_contingency_table_as_dict):
    """Returns binary (front vs. no front) frequency bias.

    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :return: binary_frequency_bias: As advertised.
    """

    binary_pod = get_binary_pod(binary_contingency_table_as_dict)
    binary_success_ratio = get_binary_success_ratio(
        binary_contingency_table_as_dict)

    try:
        return binary_pod / binary_success_ratio
    except ZeroDivisionError:
        return numpy.nan


def write_evaluation_results(
        predicted_region_table, actual_polyline_table, neigh_distance_metres,
        binary_contingency_table_as_dict, binary_pod, binary_success_ratio,
        binary_csi, binary_frequency_bias, row_normalized_ct_as_matrix,
        column_normalized_ct_as_matrix, pickle_file_name):
    """Writes evaluation results to Pickle file.

    K = number of classes

    :param predicted_region_table: pandas DataFrame created by
        `convert_regions_rowcol_to_narr_xy`.  Each row is one predicted front.
    :param actual_polyline_table: pandas DataFrame created by
        `project_polylines_latlng_to_narr`.  Each row is one actual front
        ("ground truth").
    :param neigh_distance_metres: Neighbourhood distance.  Used to define when
        actual front f_A and predicted front f_P are "matching".
    :param binary_contingency_table_as_dict: Dictionary created by
        `get_binary_contingency_table`.
    :param binary_pod: Binary probability of detection.
    :param binary_success_ratio: Binary success ratio.
    :param binary_csi: Binary critical success index.
    :param binary_frequency_bias: Binary frequency bias.
    :param row_normalized_ct_as_matrix: (K - 1)-by-K numpy array.
        row_normalized_ct_as_matrix[i, j] is the conditional probability, when
        the [i + 1]th class is predicted, that the [j]th class will be observed.
    :param column_normalized_ct_as_matrix: K-by-(K - 1) numpy array.
        column_normalized_ct_as_matrix[i, j] is the conditional probability,
        when the [j]th class is observed, that the [i]th class will be
        predicted.
    :param pickle_file_name: Path to output file.
    """

    evaluation_dict = {
        PREDICTED_REGION_TABLE_KEY: predicted_region_table,
        ACTUAL_POLYLINE_TABLE_KEY: actual_polyline_table,
        NEIGH_DISTANCE_METRES_KEY: neigh_distance_metres,
        BINARY_CONTINGENCY_TABLE_KEY: binary_contingency_table_as_dict,
        BINARY_POD_KEY: binary_pod,
        BINARY_SUCCESS_RATIO_KEY: binary_success_ratio,
        BINARY_CSI_KEY: binary_csi,
        BINARY_FREQUENCY_BIAS_KEY: binary_frequency_bias,
        ROW_NORMALIZED_CONTINGENCY_TABLE_KEY: row_normalized_ct_as_matrix,
        COLUMN_NORMALIZED_CONTINGENCY_TABLE_KEY: column_normalized_ct_as_matrix
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(evaluation_dict, pickle_file_handle)
    pickle_file_handle.close()
