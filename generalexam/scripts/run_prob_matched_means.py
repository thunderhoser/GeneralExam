"""Runs probability-matched means (PMM).

Specifically, this script applies PMM to inputs (predictors) and outputs from
one of the following interpretation methods:

- saliency maps
- class-activation maps
- backwards optimization
- novelty detection
"""

import argparse
from gewittergefahr.gg_utils import prob_matched_means as pmm
from generalexam.machine_learning import saliency_maps

NONE_STRINGS = ['', 'None']

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency_maps.read_file`).  If you'
    ' are compositing something other than saliency maps, leave this argument '
    'alone.')

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile used in PMM procedure.  See '
    '`prob_matched_means.run_pmm_one_variable` for details.')

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `saliency_maps.write_pmm_file`, '
    '`gradcam.write_pmm_file`, or `backwards_optimization.write_pmm_file`.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=False, default='',
    help=SALIENCY_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False,
    default=pmm.DEFAULT_MAX_PERCENTILE_LEVEL, help=MAX_PERCENTILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_FILE_HELP_STRING)


def _composite_saliency_maps(
        input_file_name, max_percentile_level, output_file_name):
    """Composites predictors and resulting saliency maps.

    :param input_file_name: Path to input file.  Will be read by
        `saliency_maps.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `saliency_maps.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    saliency_dict = saliency_maps.read_file(input_file_name)[0]
    denorm_predictor_matrix = saliency_dict[saliency_maps.PREDICTOR_MATRIX_KEY]
    saliency_matrix = saliency_dict[saliency_maps.SALIENCY_MATRIX_KEY]

    print('Compositing predictors...')
    mean_denorm_predictor_matrix = pmm.run_pmm_many_variables(
        input_matrix=denorm_predictor_matrix,
        max_percentile_level=max_percentile_level)

    print('Compositing saliency maps...')
    mean_saliency_matrix = pmm.run_pmm_many_variables(
        input_matrix=saliency_matrix,
        max_percentile_level=max_percentile_level)

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    saliency_maps.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_predictor_matrix=mean_denorm_predictor_matrix,
        mean_saliency_matrix=mean_saliency_matrix,
        model_file_name=saliency_dict[saliency_maps.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level)


def _run(input_saliency_file_name, max_percentile_level, output_file_name):
    """Runs probability-matched means (PMM).

    This is effectively the main method.

    :param input_saliency_file_name: See documentation at top of file.
    :param max_percentile_level: Same.
    :param output_file_name: Same.
    """

    if input_saliency_file_name not in NONE_STRINGS:
        _composite_saliency_maps(
            input_file_name=input_saliency_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_saliency_file_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
