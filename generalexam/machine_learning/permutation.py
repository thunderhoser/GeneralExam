"""Methods to run permutation test for front-detection models."""

import numpy
from gewittergefahr.gg_utils import model_evaluation as gg_evaluation
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import permutation_utils
from generalexam.ge_utils import predictor_utils
from generalexam.ge_utils import pixelwise_evaluation as pixelwise_eval
from generalexam.machine_learning import cnn

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1
NUM_THRESHOLDS_FOR_AUC = 1001

PREDICTOR_NAME_TO_FANCY = {
    predictor_utils.TEMPERATURE_NAME: 'Temperature',
    predictor_utils.HEIGHT_NAME: 'Geopotential height',
    predictor_utils.PRESSURE_NAME: 'Pressure',
    predictor_utils.DEWPOINT_NAME: 'Dewpoint',
    predictor_utils.SPECIFIC_HUMIDITY_NAME: 'Specific humidity',
    predictor_utils.U_WIND_GRID_RELATIVE_NAME: r'$u$-wind',
    predictor_utils.V_WIND_GRID_RELATIVE_NAME: r'$v$-wind',
    predictor_utils.WET_BULB_THETA_NAME: 'Wet-bulb potential temperature'
}

PREDICTOR_MATRICES_KEY = permutation_utils.PREDICTOR_MATRICES_KEY
PERMUTED_FLAGS_KEY = permutation_utils.PERMUTED_FLAGS_KEY
PERMUTED_PREDICTORS_KEY = permutation_utils.PERMUTED_PREDICTORS_KEY
PERMUTED_COST_MATRIX_KEY = permutation_utils.PERMUTED_COST_MATRIX_KEY
UNPERMUTED_PREDICTORS_KEY = permutation_utils.UNPERMUTED_PREDICTORS_KEY
UNPERMUTED_COST_MATRIX_KEY = permutation_utils.UNPERMUTED_COST_MATRIX_KEY
BEST_PREDICTOR_KEY = permutation_utils.BEST_PREDICTOR_KEY
BEST_COST_ARRAY_KEY = permutation_utils.BEST_COST_ARRAY_KEY


def _prediction_function(model_object, predictor_matrix_as_list):
    """Prediction function for CNN that does front detection.

    E = number of examples
    M = number of rows in example grid
    N = number of columns in example grid
    C = number of predictors (channels)
    K = number of classes

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix_as_list: length-1 list, where the only item is the
        predictor matrix (E-by-M-by-N-by-C numpy array).
    :return: class_probability_matrix: E-by-K numpy array of class
        probabilities.
    """

    return cnn.apply_model(
        model_object=model_object, predictor_matrix=predictor_matrix_as_list[0],
        verbose=True
    )


def negative_auc_function(observed_labels, class_probability_matrix):
    """Computes negative AUC (area under the ROC curve).

    For multi-class problems, the "AUC" computed by this function is the mean
    AUC over all classes except 0 (the non-event class).

    E = number of examples
    K = number of classes

    :param observed_labels: length-E numpy array of observed classes (integers
        in range 0...[K - 1]).
    :param class_probability_matrix: E-by-K numpy array of predicted
        probabilities.
    :return: negative_auc: Negative AUC.
    """

    pixelwise_eval.check_predictions_and_obs(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels)

    prob_thresholds = gg_evaluation.get_binarization_thresholds(
        threshold_arg=NUM_THRESHOLDS_FOR_AUC
    )

    num_thresholds = len(prob_thresholds)
    pod_values = numpy.full(num_thresholds, numpy.nan)
    pofd_values = numpy.full(num_thresholds, numpy.nan)

    for k in range(num_thresholds):
        these_predicted_labels = pixelwise_eval.determinize_predictions(
            class_probability_matrix=class_probability_matrix,
            threshold=prob_thresholds[k]
        )

        this_contingency_matrix = pixelwise_eval.get_contingency_table(
            predicted_labels=these_predicted_labels,
            observed_labels=observed_labels)

        pod_values[k] = pixelwise_eval.get_binary_pod(this_contingency_matrix)
        pofd_values[k] = pixelwise_eval.get_binary_pofd(this_contingency_matrix)

    return -1 * gg_evaluation.get_area_under_roc_curve(
        pod_by_threshold=pod_values, pofd_by_threshold=pofd_values
    )


def get_nice_predictor_names(predictor_names, pressure_levels_mb):
    """Creates nice (human-readable) predictor names.

    C = number of predictors (channels)

    :param predictor_names: length-C list of default predictor names (accepted
        by `predictor_utils.check_field_name`).
    :param pressure_levels_mb: length-C numpy array of pressure levels
        (millibars).
    :return: nice_predictor_names: length-C list of nice predictor names that
        include pressure level.
    """

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names), num_dimensions=1
    )
    expected_dim = numpy.array([len(predictor_names)], dtype=int)

    error_checking.assert_is_numpy_array(pressure_levels_mb)
    pressure_levels_mb = numpy.round(pressure_levels_mb).astype(int)
    error_checking.assert_is_numpy_array(
        pressure_levels_mb, exact_dimensions=expected_dim)

    return [
        '{0:s} {1:s}{2:s}'.format(
            'Surface' if l == predictor_utils.DUMMY_SURFACE_PRESSURE_MB
            else '{0:d}-mb'.format(l),
            PREDICTOR_NAME_TO_FANCY[n][0].lower(),
            PREDICTOR_NAME_TO_FANCY[n][1:]
        )
        for n, l in zip(predictor_names, pressure_levels_mb)
    ]


def run_forward_test(
        model_object, predictor_matrix, observed_labels, model_metadata_dict,
        cost_function, num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs forward version of permutation test.

    E = number of examples
    M = number of rows in example grid
    N = number of columns in example grid
    C = number of predictors (channels)
    K = number of classes

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param observed_labels: length-E numpy array of observed classes (integers
        in range 0...[K - 1]).
    :param model_metadata_dict: Dictionary returned by
        `cnn.read_model_metadata`, corresponding to `model_object`.
    :param cost_function: Function used to evaluate model predictions.  Must be
        negatively oriented (lower is better), with the following inputs and
        outputs.
    Input: observed_labels: Same as input to this method.
    Input: class_probability_matrix: E-by-K numpy array of class probabilities.
    Output: cost: Scalar value

    :param num_bootstrap_reps: Number of bootstrap replicates.  If you do not
        want to use bootstrapping, make this <= 1.

    :return: result_dict: Dictionary with the following keys, where B = number
        of bootstrap replicates.

    result_dict["best_predictor_names"]: length-C list of best predictors.
        The [j]th element is the name of the [j]th predictor to be permanently
        permuted.
    result_dict["best_cost_matrix"]: C-by-B numpy array of costs after
        permutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before permutation.
    result_dict["step1_predictor_names"]: length-C list of predictors in
        the order that they were permuted in step 1.
    result_dict["step1_cost_matrix"]: C-by-B numpy array of costs after
        permutation in step 1.
    result_dict["backwards_test"]: Boolean flag (always False).
    """

    # Check and process input args.
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    nice_predictor_names = get_nice_predictor_names(
        predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    )

    print('Nice predictor names:\n{0:s}'.format(str(nice_predictor_names)))
    print(SEPARATOR_STRING)

    # Find original cost (before permutation).
    print('Finding original cost (before permutation)...')

    class_probability_matrix = _prediction_function(
        model_object, [predictor_matrix]
    )
    print(MINOR_SEPARATOR_STRING)

    original_cost_array = permutation_utils.bootstrap_cost(
        target_values=observed_labels,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Do the dirty work.
    num_predictors = len(nice_predictor_names)
    permuted_flags = numpy.full(num_predictors, 0, dtype=bool)

    step_num = 0

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_dict = permutation_utils.run_forward_test_one_step(
            model_object=model_object, predictor_matrices=[predictor_matrix],
            predictor_names_by_matrix=[nice_predictor_names],
            target_values=observed_labels, separate_heights=False,
            prediction_function=_prediction_function,
            cost_function=cost_function,
            num_bootstrap_reps=num_bootstrap_reps, step_num=step_num,
            permuted_flags_by_matrix=[permuted_flags]
        )

        if this_dict is None:
            break

        predictor_matrix = this_dict[PREDICTOR_MATRICES_KEY][0]
        permuted_flags = this_dict[PERMUTED_FLAGS_KEY][0]
        best_predictor_names.append(this_dict[BEST_PREDICTOR_KEY])

        this_best_cost_array = this_dict[BEST_COST_ARRAY_KEY]
        this_best_cost_matrix = numpy.reshape(
            this_best_cost_array, (1, len(this_best_cost_array))
        )
        best_cost_matrix = numpy.concatenate(
            (best_cost_matrix, this_best_cost_matrix), axis=0
        )

        if step_num == 1:
            step1_predictor_names = this_dict[PERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[PERMUTED_COST_MATRIX_KEY]

    return {
        permutation_utils.BEST_PREDICTORS_KEY: best_predictor_names,
        permutation_utils.BEST_COST_MATRIX_KEY: best_cost_matrix,
        permutation_utils.ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        permutation_utils.STEP1_PREDICTORS_KEY: step1_predictor_names,
        permutation_utils.STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        permutation_utils.BACKWARDS_FLAG: False
    }


def run_backwards_test(
        model_object, predictor_matrix, observed_labels, model_metadata_dict,
        cost_function, num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs backwards version of permutation test.

    C = number of predictors (channels)
    B = number of bootstrap replicates

    :param model_object: See doc for `run_forward_test`.
    :param predictor_matrix: Same.
    :param observed_labels: Same.
    :param model_metadata_dict: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.

    :return: result_dict: Dictionary with the following keys.
    result_dict["best_predictor_names"]: length-C list of best
        predictors.  The [j]th element is the name of the [j]th predictor to be
        permanently unpermuted.
    result_dict["best_cost_matrix"]: C-by-B numpy array of costs after
        unpermutation.
    result_dict["original_cost_array"]: length-B numpy array of costs
        before unpermutation.
    result_dict["step1_predictor_names"]: length-C list of predictors in
        the order that they were unpermuted in step 1.
    result_dict["step1_cost_matrix"]: C-by-B numpy array of costs after
        unpermutation in step 1.
    result_dict["backwards_test"]: Boolean flag (always True).
    """

    # Check and process input args.
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = max([num_bootstrap_reps, 1])

    nice_predictor_names = get_nice_predictor_names(
        predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
        pressure_levels_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY]
    )

    print('Nice predictor names:\n{0:s}'.format(str(nice_predictor_names)))
    print(SEPARATOR_STRING)

    # Permute all predictors.
    num_predictors = len(nice_predictor_names)
    clean_predictor_matrix = predictor_matrix + 0.

    for k in range(num_predictors):
        predictor_matrix = permutation_utils.permute_one_predictor(
            predictor_matrices=[predictor_matrix],
            separate_heights=False, matrix_index=0, predictor_index=k
        )[0][0]

    # Find original cost (before unpermutation).
    print('Finding original cost (before unpermutation)...')
    class_probability_matrix = _prediction_function(
        model_object, [predictor_matrix]
    )
    print(MINOR_SEPARATOR_STRING)

    original_cost_array = permutation_utils.bootstrap_cost(
        target_values=observed_labels,
        class_probability_matrix=class_probability_matrix,
        cost_function=cost_function, num_replicates=num_bootstrap_reps)

    # Do the dirty work.
    step_num = 0
    permuted_flags = numpy.full(num_predictors, 1, dtype=bool)

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_dict = permutation_utils.run_backwards_test_one_step(
            model_object=model_object, predictor_matrices=[predictor_matrix],
            clean_predictor_matrices=[clean_predictor_matrix],
            predictor_names_by_matrix=[nice_predictor_names],
            target_values=observed_labels, separate_heights=False,
            prediction_function=_prediction_function,
            cost_function=cost_function,
            num_bootstrap_reps=num_bootstrap_reps, step_num=step_num,
            permuted_flags_by_matrix=[permuted_flags]
        )

        if this_dict is None:
            break

        predictor_matrix = this_dict[PREDICTOR_MATRICES_KEY][0]
        permuted_flags = this_dict[PERMUTED_FLAGS_KEY][0]
        best_predictor_names.append(this_dict[BEST_PREDICTOR_KEY])

        this_best_cost_array = this_dict[BEST_COST_ARRAY_KEY]
        this_best_cost_matrix = numpy.reshape(
            this_best_cost_array, (1, len(this_best_cost_array))
        )
        best_cost_matrix = numpy.concatenate(
            (best_cost_matrix, this_best_cost_matrix), axis=0
        )

        if step_num == 1:
            step1_predictor_names = this_dict[UNPERMUTED_PREDICTORS_KEY]
            step1_cost_matrix = this_dict[UNPERMUTED_COST_MATRIX_KEY]

    return {
        permutation_utils.BEST_PREDICTORS_KEY: best_predictor_names,
        permutation_utils.BEST_COST_MATRIX_KEY: best_cost_matrix,
        permutation_utils.ORIGINAL_COST_ARRAY_KEY: original_cost_array,
        permutation_utils.STEP1_PREDICTORS_KEY: step1_predictor_names,
        permutation_utils.STEP1_COST_MATRIX_KEY: step1_cost_matrix,
        permutation_utils.BACKWARDS_FLAG: True
    }
