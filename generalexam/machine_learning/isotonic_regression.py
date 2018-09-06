"""Training and testing methods for isotonic regression."""

import pickle
import os.path
import numpy
from sklearn.isotonic import IsotonicRegression
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking


def _check_evaluation_pairs(class_probability_matrix, observed_labels):
    """Checks evaluation pairs for errors.

    P = number of evaluation pairs
    K = number of classes

    :param class_probability_matrix: P-by-K numpy array of floats.
        class_probability_matrix[i, k] is the predicted probability that the
        [i]th example belongs to the [k]th class.
    :param observed_labels: length-P numpy array of integers.  If
        observed_labels[i] = k, the [i]th example truly belongs to the [k]th
        class.
    """

    # TODO(thunderhoser): This method is duplicated from evaluation_utils.py.  I
    # can't just import evaluation_utils.py, because this leads to a circular
    # import chain.  The answer is to put this method somewhere more general.

    error_checking.assert_is_numpy_array(
        class_probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(class_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(class_probability_matrix, 1.)

    num_evaluation_pairs = class_probability_matrix.shape[0]
    num_classes = class_probability_matrix.shape[1]

    error_checking.assert_is_numpy_array(
        observed_labels, exact_dimensions=numpy.array([num_evaluation_pairs]))
    error_checking.assert_is_integer_numpy_array(observed_labels)
    error_checking.assert_is_geq_numpy_array(observed_labels, 0)
    error_checking.assert_is_less_than_numpy_array(observed_labels, num_classes)


def train_model_for_each_class(orig_class_probability_matrix, observed_labels):
    """Trains isotonic-regression model for each class.

    P = number of examples
    K = number of classes

    :param orig_class_probability_matrix: P-by-K numpy array uncalibrated
        probabilities.  class_probability_matrix[i, k] is the predicted
        probability that the [i]th example belongs to the [k]th class.
    :param observed_labels: length-P numpy array of integers.  If
        observed_labels[i] = k, the [i]th example truly belongs to the [k]th
        class.
    :return: model_object_by_class: length-K list with trained instances of
        `sklearn.isotonic.IsotonicRegression`.
    """

    _check_evaluation_pairs(
        class_probability_matrix=orig_class_probability_matrix,
        observed_labels=observed_labels)

    num_classes = orig_class_probability_matrix.shape[1]
    model_object_by_class = [None] * num_classes

    for k in range(num_classes):
        print 'Training isotonic-regression model for class {0:d}...'.format(k)

        model_object_by_class[k] = IsotonicRegression(
            y_min=0., y_max=1., increasing=True, out_of_bounds='clip')
        model_object_by_class[k].fit(
            X=orig_class_probability_matrix[:, k],
            y=(observed_labels == k).astype(int))

    return model_object_by_class


def apply_model_for_each_class(
        orig_class_probability_matrix, observed_labels, model_object_by_class):
    """Applies isotonic-regression model for each class.

    :param orig_class_probability_matrix: See documentation for
        `train_model_for_each_class`.
    :param observed_labels: Same.
    :param model_object_by_class: Same.
    :return: new_class_probability_matrix: Calibrated version of
        `orig_class_probability_matrix`.
    :raises: ValueError: if number of models != number of columns in
        `orig_class_probability_matrix`.
    """

    _check_evaluation_pairs(
        class_probability_matrix=orig_class_probability_matrix,
        observed_labels=observed_labels)

    error_checking.assert_is_list(model_object_by_class)
    num_classes = orig_class_probability_matrix.shape[1]
    if len(model_object_by_class) != num_classes:
        error_string = (
            'Number of models ({0:d}) should = number of columns in '
            'orig_class_probability_matrix ({1:d}).').format(
                len(model_object_by_class), num_classes)
        raise ValueError(error_string)

    num_examples = orig_class_probability_matrix.shape[0]
    new_class_probability_matrix = numpy.full(
        (num_examples, num_classes), numpy.nan)

    for k in range(num_classes):
        print 'Applying isotonic-regression model for class {0:d}...'.format(k)
        new_class_probability_matrix[:, k] = model_object_by_class[k].predict(
            orig_class_probability_matrix[:, k])

    # Ensure that sum of class probabilities = 1 for each example.
    for i in range(num_examples):
        new_class_probability_matrix[i, :] = (
            new_class_probability_matrix[i, :] /
            numpy.sum(new_class_probability_matrix[i, :]))

    _check_evaluation_pairs(
        class_probability_matrix=new_class_probability_matrix,
        observed_labels=observed_labels)

    return new_class_probability_matrix


def find_model_file(base_model_file_name, raise_error_if_missing=True):
    """Finds file containing isotonic-regression model(s).

    This file should be written by `write_model_for_each_class`.

    :param base_model_file_name: Path to file containing base model (e.g., CNN).
    :param raise_error_if_missing: Boolean flag.  If isotonic-regression file is
        missing and `raise_error_if_missing = True`, this method will error out.
    :return: isotonic_file_name: Path to metafile.  If isotonic-regression file
        is missing and `raise_error_if_missing = False`, this will be the
        *expected* path.
    :raises: ValueError: if isotonic-regression file is missing and
        `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(base_model_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    isotonic_file_name = '{0:s}/isotonic_regression_models.p'.format(
        os.path.split(base_model_file_name)[0])
    if not os.path.isfile(isotonic_file_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            isotonic_file_name)
        raise ValueError(error_string)

    return isotonic_file_name


def write_model_for_each_class(model_object_by_class, pickle_file_name):
    """Writes models to Pickle file.

    :param model_object_by_class: See documentation for
        `train_model_for_each_class`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(model_object_by_class, pickle_file_handle)
    pickle_file_handle.close()


def read_model_for_each_class(pickle_file_name):
    """Reads models from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: model_object_by_class: See documentation for
        `train_model_for_each_class`.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    model_object_by_class = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return model_object_by_class
