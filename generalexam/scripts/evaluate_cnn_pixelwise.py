"""Evaluates CNN-generated front probabilities pixel by pixel."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_io import ungridded_prediction_io
from generalexam.machine_learning import evaluation_utils

INPUT_TIME_FORMAT = '%Y%m%d%H'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCORE_NAME_TO_FUNCTION = {
    'gerrity': evaluation_utils.get_gerrity_score,
    'csi': evaluation_utils.get_csi
}

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
SCORING_FUNCTION_ARG_NAME = 'scoring_function_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with ungridded predictions from the CNN.  '
    'Files therein will be found by `ungridded_prediction_io.find_file` and '
    'read by `ungridded_prediction_io.read_file`.')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  All predictions in the period '
    '`{0:s}`...`{1:s}` will be evaluated.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

SCORING_FUNCTION_HELP_STRING = (
    'Scoring function used to choose best determinization threshold.  Must be '
    'in the following list:\n{0:s}'
).format(
    str(SCORE_NAME_TO_FUNCTION.keys())
)

OUTPUT_FILE_HELP_STRING = 'Path to output file (NetCDF).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + SCORING_FUNCTION_ARG_NAME, type=str, required=False, default='csi',
    help=SCORING_FUNCTION_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(top_prediction_dir_name, first_time_string, last_time_string,
         scoring_function_name, output_file_name):
    """Evaluates CNN-generated front probabilities pixel by pixel.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param scoring_function_name: Same.
    :param output_file_name: Same.
    """

    scoring_function = SCORE_NAME_TO_FUNCTION[scoring_function_name]

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    prediction_file_names = ungridded_prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec)

    class_probability_matrix = None
    observed_labels = numpy.array([], dtype=int)

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = ungridded_prediction_io.read_file(this_file_name)

        this_prob_matrix = this_prediction_dict[
            ungridded_prediction_io.CLASS_PROBABILITIES_KEY
        ]
        these_observed_labels = this_prediction_dict[
            ungridded_prediction_io.OBSERVED_LABELS_KEY
        ]

        observed_labels = numpy.concatenate((
            observed_labels, these_observed_labels
        ))

        if class_probability_matrix is None:
            class_probability_matrix = this_prob_matrix + 0.
        else:
            class_probability_matrix = numpy.concatenate(
                (class_probability_matrix, this_prob_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    best_threshold, best_score, all_thresholds = (
        evaluation_utils.find_best_determinization_threshold(
            class_probability_matrix=class_probability_matrix,
            observed_labels=observed_labels, scoring_function=scoring_function)
    )
    print(all_thresholds)

    print((
        '\nBest determinization threshold = {0:.4f} ... corresponding "{1:s}" '
        'score = {2:.4f}'
    ).format(
        best_threshold, scoring_function_name, best_score
    ))
    print(SEPARATOR_STRING)

    climo_counts = numpy.array([
        numpy.sum(observed_labels == k)
        for k in range(evaluation_utils.NUM_CLASSES)
    ], dtype=int)

    result_table_xarray = evaluation_utils.run_evaluation(
        class_probability_matrix=class_probability_matrix,
        observed_labels=observed_labels,
        best_determinizn_threshold=best_threshold,
        all_determinizn_thresholds=all_thresholds,
        climo_counts=climo_counts, bootstrap_rep_index=0)

    print(SEPARATOR_STRING)
    print('Writing results to: "{0:s}"...'.format(output_file_name))

    evaluation_utils.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        scoring_function_name=getattr(
            INPUT_ARG_OBJECT, SCORING_FUNCTION_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
