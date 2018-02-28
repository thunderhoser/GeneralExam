"""Methods for evaluating a machine-learning model.

--- NOTATION ---

Throughout this module, the following letters will be used to denote matrix
dimensions.

E = number of examples.  Each example is one image or a time sequence of images.
M = number of pixel rows in each image
N = number of pixel columns in each image
T = number of predictor times per example (images per sequence)
C = number of channels (predictor variables) in each image

--- DEFINITIONS ---

"Evaluation pair" = forecast-prediction pair

A "downsized" example covers only a portion of the NARR grid (as opposed to
a full-size example, which covers the entire NARR grid).

For a 3-D example, the dimensions are M x N x C (M rows, N columns, C predictor
variables).

For a 4-D example, the dimensions are M x N x T x C (M rows, N columns, T time
steps, C predictor variables).
"""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import model_evaluation as model_eval
from gewittergefahr.gg_utils import error_checking
from generalexam.machine_learning import testing_io
from generalexam.machine_learning import machine_learning_utils as ml_utils

NARR_TIME_INTERVAL_SECONDS = 10800
DEFAULT_FORECAST_PRECISION_FOR_THRESHOLDS = 1e-3

MIN_OPTIMIZATION_DIRECTION = 'min'
MAX_OPTIMIZATION_DIRECTION = 'max'
VALID_OPTIMIZATION_DIRECTIONS = [
    MIN_OPTIMIZATION_DIRECTION, MAX_OPTIMIZATION_DIRECTION]


def downsized_3d_examples_to_eval_pairs(
        model_object, first_target_time_unix_sec, last_target_time_unix_sec,
        num_target_times_to_sample, num_examples_per_time,
        top_narr_directory_name, top_frontal_grid_dir_name,
        narr_predictor_names, pressure_level_mb, dilation_half_width_for_target,
        num_rows_in_half_grid, num_columns_in_half_grid):
    """Creates evaluation pairs from downsized 3-D examples.

    M = number of pixel rows in full NARR grid
    N = number of pixel columns in full NARR grid

    m = number of pixel rows in each downsized grid
      = 2 * num_rows_in_half_grid + 1
    n = number of pixel columns in each downsized grid
      = 2 * num_columns_in_half_grid + 1

    Q = number of evaluation pairs created by this method

    :param model_object: Instance of `keras.models.Model`.  This will be applied
        to each downsized example, creating the prediction for said example.
    :param first_target_time_unix_sec: Target time.  Downsized examples will be
        randomly chosen from the period `first_target_time_unix_sec`...
        `last_target_time_unix_sec`.
    :param last_target_time_unix_sec: See above.
    :param num_target_times_to_sample: Number of target times to sample (from
        the period `first_target_time_unix_sec`...`last_target_time_unix_sec`).
    :param num_examples_per_time: Number of downsized examples per target time.
        Downsized examples will be randomly drawn from each target time.
    :param top_narr_directory_name: Name of top-level directory with NARR data
        (one file for each variable, pressure level, and time step).
    :param top_frontal_grid_dir_name: Name of top-level directory with frontal
        grids (one file per time step).
    :param narr_predictor_names: 1-D list of NARR fields to use as predictors.
    :param pressure_level_mb: Pressure level (millibars).
    :param dilation_half_width_for_target: Half-width of dilation window for
        target variable.  For each time step t and grid cell [j, k], if a front
        occurs within `dilation_half_width_for_target` of [j, k] at time t, the
        label at [t, j, k] will be positive.
    :param num_rows_in_half_grid: See general discussion above.
    :param num_columns_in_half_grid: See general discussion above.
    :return: predicted_probabilities: length-Q numpy array, where the [i]th
        element is the predicted probability of the positive class for the [i]th
        evaluation pair.
    :return: observed_labels: length-Q numpy array, where the [i]th element is
        the observed class label (0 or 1, integer) for the [i]th evaluation
        pair.
    """

    error_checking.assert_is_integer(num_target_times_to_sample)
    error_checking.assert_is_greater(num_target_times_to_sample, 0)
    error_checking.assert_is_integer(num_examples_per_time)
    error_checking.assert_is_greater(num_examples_per_time, 0)

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(target_times_unix_sec)
    target_times_unix_sec = target_times_unix_sec[:num_target_times_to_sample]

    all_row_indices = numpy.linspace(
        0, len(ml_utils.NARR_ROWS_WITHOUT_NAN) - 1,
        num=len(ml_utils.NARR_ROWS_WITHOUT_NAN), dtype=int)
    all_column_indices = numpy.linspace(
        0, len(ml_utils.NARR_COLUMNS_WITHOUT_NAN) - 1,
        num=len(ml_utils.NARR_COLUMNS_WITHOUT_NAN), dtype=int)

    column_index_matrix, row_index_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=all_column_indices, y_unique_metres=all_row_indices)

    full_predictor_matrix = None
    full_target_matrix = None
    predicted_probabilities = numpy.full(
        (num_target_times_to_sample, num_examples_per_time), numpy.nan)
    observed_labels = numpy.full(
        (num_target_times_to_sample, num_examples_per_time), -1, dtype=int)

    for i in range(num_target_times_to_sample):
        these_center_row_indices = numpy.random.choice(
            row_index_matrix, size=num_examples_per_time, replace=False
        ).astype(int)
        these_center_column_indices = numpy.random.choice(
            column_index_matrix, size=num_examples_per_time, replace=False
        ).astype(int)

        if i == 0:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             full_predictor_matrix,
             full_target_matrix) = testing_io.create_downsized_3d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 target_time_unix_sec=target_times_unix_sec[i],
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_half_width_for_target=dilation_half_width_for_target)

        else:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             _, _) = testing_io.create_downsized_3d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 full_predictor_matrix=full_predictor_matrix,
                 full_target_matrix=full_target_matrix)

        this_prediction_matrix = model_object.predict(
            this_downsized_predictor_matrix, batch_size=num_examples_per_time)
        predicted_probabilities[i, :] = this_prediction_matrix[:, 1]

    predicted_probabilities = numpy.reshape(
        predicted_probabilities, predicted_probabilities.size)
    observed_labels = numpy.reshape(observed_labels, observed_labels.size)

    return predicted_probabilities, observed_labels


def downsized_4d_examples_to_eval_pairs(
        model_object, first_target_time_unix_sec, last_target_time_unix_sec,
        num_target_times_to_sample, num_examples_per_time,
        num_predictor_time_steps, num_lead_time_steps, top_narr_directory_name,
        top_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_half_width_for_target, num_rows_in_half_grid,
        num_columns_in_half_grid):
    """Creates evaluation pairs from downsized 4-D examples.

    :param model_object: See documentation for
        `downsized_3d_examples_to_eval_pairs`.
    :param first_target_time_unix_sec: Same.
    :param last_target_time_unix_sec: Same.
    :param num_target_times_to_sample: Same.
    :param num_examples_per_time: Same.
    :param num_predictor_time_steps: Number of predictor times per example.
    :param num_lead_time_steps: Number of time steps separating latest predictor
        time from target time.
    :param top_narr_directory_name: See documentation for
        `downsized_3d_examples_to_eval_pairs`.
    :param top_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_half_width_for_target: Same.
    :param num_rows_in_half_grid: Same.
    :param num_columns_in_half_grid: Same.
    :return: predicted_probabilities: Same.
    :return: observed_labels: Same.
    """

    error_checking.assert_is_integer(num_target_times_to_sample)
    error_checking.assert_is_greater(num_target_times_to_sample, 0)
    error_checking.assert_is_integer(num_examples_per_time)
    error_checking.assert_is_greater(num_examples_per_time, 0)

    target_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_target_time_unix_sec,
        end_time_unix_sec=last_target_time_unix_sec,
        time_interval_sec=NARR_TIME_INTERVAL_SECONDS, include_endpoint=True)

    numpy.random.shuffle(target_times_unix_sec)
    target_times_unix_sec = target_times_unix_sec[:num_target_times_to_sample]

    all_row_indices = numpy.linspace(
        0, len(ml_utils.NARR_ROWS_WITHOUT_NAN) - 1,
        num=len(ml_utils.NARR_ROWS_WITHOUT_NAN), dtype=int)
    all_column_indices = numpy.linspace(
        0, len(ml_utils.NARR_COLUMNS_WITHOUT_NAN) - 1,
        num=len(ml_utils.NARR_COLUMNS_WITHOUT_NAN), dtype=int)

    column_index_matrix, row_index_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=all_column_indices, y_unique_metres=all_row_indices)

    full_predictor_matrix = None
    full_target_matrix = None
    predicted_probabilities = numpy.full(
        (num_target_times_to_sample, num_examples_per_time), numpy.nan)
    observed_labels = numpy.full(
        (num_target_times_to_sample, num_examples_per_time), -1, dtype=int)

    for i in range(num_target_times_to_sample):
        these_center_row_indices = numpy.random.choice(
            row_index_matrix, size=num_examples_per_time, replace=False
        ).astype(int)
        these_center_column_indices = numpy.random.choice(
            column_index_matrix, size=num_examples_per_time, replace=False
        ).astype(int)

        if i == 0:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             full_predictor_matrix,
             full_target_matrix) = testing_io.create_downsized_4d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 num_predictor_time_steps=num_predictor_time_steps,
                 num_lead_time_steps=num_lead_time_steps,
                 target_time_unix_sec=target_times_unix_sec[i],
                 top_narr_directory_name=top_narr_directory_name,
                 top_frontal_grid_dir_name=top_frontal_grid_dir_name,
                 narr_predictor_names=narr_predictor_names,
                 pressure_level_mb=pressure_level_mb,
                 dilation_half_width_for_target=dilation_half_width_for_target)

        else:
            (this_downsized_predictor_matrix,
             observed_labels[i, :],
             _, _) = testing_io.create_downsized_4d_examples(
                 center_row_indices=these_center_row_indices,
                 center_column_indices=these_center_column_indices,
                 num_rows_in_half_grid=num_rows_in_half_grid,
                 num_columns_in_half_grid=num_columns_in_half_grid,
                 full_predictor_matrix=full_predictor_matrix,
                 full_target_matrix=full_target_matrix)

        this_prediction_matrix = model_object.predict(
            this_downsized_predictor_matrix, batch_size=num_examples_per_time)
        predicted_probabilities[i, :] = this_prediction_matrix[:, 1]

    predicted_probabilities = numpy.reshape(
        predicted_probabilities, predicted_probabilities.size)
    observed_labels = numpy.reshape(observed_labels, observed_labels.size)

    return predicted_probabilities, observed_labels


def find_best_decision_threshold(
        predicted_probabilities, observed_labels, threshold_arg,
        criterion_function=model_eval.get_peirce_score,
        optimization_direction=MAX_OPTIMIZATION_DIRECTION,
        forecast_precision_for_thresholds=
        DEFAULT_FORECAST_PRECISION_FOR_THRESHOLDS):
    """Finds the best decision threshold.

    The "best" decision threshold is the probability p* that, when used to
    binarize probabilistic forecasts, yields the best value of the given
    criterion (examples: maximum CSI, maximum Peirce score, minimum POFD, etc.).

    Q = number of evaluation pairs

    :param predicted_probabilities: length-Q numpy array, where the [i]th
        element is the predicted probability of the positive class for the [i]th
        evaluation pair.
    :param observed_labels: length-Q numpy array, where the [i]th element is
        the observed class label (0 or 1, integer) for the [i]th evaluation
        pair.
    :param threshold_arg: See documentation for
        `model_evaluation._get_binarization_thresholds`.  Determines thresholds
        that will be tried.
    :param criterion_function: Criterion to be either minimized or maximized.
        This must be a function that takes input `contingency_table_as_dict` and
        returns a single float.  See `model_evaluation.get_csi` for an example.
    :param optimization_direction: Direction in which criterion function is
        optimized.  Options are "min" and "max".
    :param forecast_precision_for_thresholds: See documentation for
        `model_evaluation._get_binarization_thresholds`.  Determines thresholds
        that will be tried.
    :return: best_probability_threshold: Best decision threshold.
    :return: best_criterion_value: Value of criterion function at said
        threshold.
    """

    error_checking.assert_is_string(optimization_direction)
    if optimization_direction not in VALID_OPTIMIZATION_DIRECTIONS:
        error_string = (
            '\n\n{0:s}\nValid optimization directions (listed above) do not '
            'include "{1:s}".').format(VALID_OPTIMIZATION_DIRECTIONS,
                                       optimization_direction)
        raise ValueError(error_string)

    probability_thresholds = model_eval.get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=predicted_probabilities,
        unique_forecast_precision=forecast_precision_for_thresholds)

    num_thresholds = len(probability_thresholds)
    criterion_values = numpy.full(num_thresholds, numpy.nan)

    for i in range(num_thresholds):
        these_predicted_labels = model_eval.binarize_forecast_probs(
            forecast_probabilities=predicted_probabilities,
            binarization_threshold=probability_thresholds[i])

        this_contingency_table_as_dict = model_eval.get_contingency_table(
            forecast_labels=these_predicted_labels,
            observed_labels=observed_labels)

        criterion_values[i] = criterion_function(this_contingency_table_as_dict)

    if optimization_direction == MAX_OPTIMIZATION_DIRECTION:
        best_criterion_value = numpy.nanmax(criterion_values)
        best_probability_threshold = probability_thresholds[
            numpy.nanargmax(criterion_values)]
    else:
        best_criterion_value = numpy.nanmin(criterion_values)
        best_probability_threshold = probability_thresholds[
            numpy.nanargmin(criterion_values)]

    return best_probability_threshold, best_criterion_value
