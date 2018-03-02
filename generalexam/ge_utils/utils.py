"""General helper methods."""

import numpy
from scipy.ndimage import distance_transform_edt
from gewittergefahr.gg_utils import error_checking


def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True)
    return data_matrix[tuple(indices)]
