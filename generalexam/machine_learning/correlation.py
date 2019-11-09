"""Methods to compute correlation among predictors."""

import numpy
from scipy.stats import pearsonr
from gewittergefahr.gg_utils import error_checking


def get_pearson_correlations(predictor_matrix):
    """Computes Pearson correlation between each pair of predictors.

    E = number of examples
    M = number of rows in example grid
    N = number of columns in example grid
    C = number of predictors (channels)

    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: correlation_matrix: C-by-C numpy array of Pearson correlations.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=4)

    num_predictors = predictor_matrix.shape[-1]
    correlation_matrix = numpy.full((num_predictors, num_predictors), numpy.nan)

    for i in range(num_predictors):
        for j in range(i, num_predictors):
            if i == j:
                correlation_matrix[i, j] = 1.
                continue

            correlation_matrix[i, j] = pearsonr(
                numpy.ravel(predictor_matrix[..., i]),
                numpy.ravel(predictor_matrix[..., j])
            )[0]

            correlation_matrix[j, i] = correlation_matrix[i, j]

    return correlation_matrix
