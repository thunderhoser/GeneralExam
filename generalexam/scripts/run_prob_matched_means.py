"""Runs probability-matched means (PMM).

Specifically, this script applies PMM to inputs (predictors) and outputs from
one of the following interpretation methods:

- saliency maps
- class-activation maps
- backwards optimization
- novelty detection
"""

import argparse
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from generalexam.machine_learning import saliency_maps
from generalexam.machine_learning import gradcam
from generalexam.machine_learning import backwards_optimization as backwards_opt

NONE_STRINGS = ['', 'None']

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
BWO_FILE_ARG_NAME = 'input_bwo_file_name'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file (will be read by `saliency_maps.read_file`).  If you'
    ' are compositing something other than saliency maps, leave this argument '
    'alone.')

GRADCAM_FILE_HELP_STRING = (
    'Path to Grad-CAM file (will be read by `gradcam.read_file`).  If you are '
    'compositing something other than class-activation maps, leave this '
    'argument alone.')

BWO_FILE_HELP_STRING = (
    'Path to backwards-optimization file (will be read by '
    '`backwards_optimization.read_file`).  If you are compositing something '
    'other than BWO results, leave this argument alone.')

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
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=False, default='',
    help=GRADCAM_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + BWO_FILE_ARG_NAME, type=str, required=False, default='',
    help=BWO_FILE_HELP_STRING)

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


def _composite_gradcam(input_file_name, max_percentile_level, output_file_name):
    """Composites predictors and resulting class-activation maps.

    :param input_file_name: Path to input file.  Will be read by
        `gradcam.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `gradcam.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gradcam_dict = gradcam.read_file(input_file_name)[0]
    denorm_predictor_matrix = gradcam_dict[gradcam.PREDICTOR_MATRIX_KEY]
    class_activn_matrix = gradcam_dict[gradcam.ACTIVN_MATRIX_KEY]
    guided_class_activn_matrix = gradcam_dict[gradcam.GUIDED_ACTIVN_MATRIX_KEY]

    print('Compositing predictors...')
    mean_denorm_predictor_matrix = pmm.run_pmm_many_variables(
        input_matrix=denorm_predictor_matrix,
        max_percentile_level=max_percentile_level)

    print('Compositing unguided CAMs...')
    mean_activn_matrix = pmm.run_pmm_many_variables(
        input_matrix=numpy.expand_dims(class_activn_matrix, axis=-1),
        max_percentile_level=max_percentile_level
    )[..., 0]

    print('Compositing guided CAMs...')
    mean_guided_activn_matrix = pmm.run_pmm_many_variables(
        input_matrix=guided_class_activn_matrix,
        max_percentile_level=max_percentile_level)

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    gradcam.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_predictor_matrix=mean_denorm_predictor_matrix,
        mean_activn_matrix=mean_activn_matrix,
        mean_guided_activn_matrix=mean_guided_activn_matrix,
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level)


def _composite_backwards_opt(
        input_file_name, max_percentile_level, output_file_name):
    """Composites pre- and post-optimized examples.

    :param input_file_name: Path to input file.  Will be read by
        `backwards_optimization.read_file`.
    :param max_percentile_level: See documentation at top of file.
    :param output_file_name: Path to output file.  Will be written by
        `backwards_optimization.write_pmm_file`.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    bwo_dictionary = backwards_opt.read_file(input_file_name)[0]
    denorm_input_matrix = bwo_dictionary[backwards_opt.INPUT_MATRIX_KEY]
    denorm_output_matrix = bwo_dictionary[backwards_opt.OUTPUT_MATRIX_KEY]

    print('Compositing pre-optimized examples...')
    mean_denorm_input_matrix = pmm.run_pmm_many_variables(
        input_matrix=denorm_input_matrix,
        max_percentile_level=max_percentile_level)

    print('Compositing optimized examples...')
    mean_denorm_output_matrix = pmm.run_pmm_many_variables(
        input_matrix=denorm_output_matrix,
        max_percentile_level=max_percentile_level)

    print('Writing output to: "{0:s}"...'.format(output_file_name))
    backwards_opt.write_pmm_file(
        pickle_file_name=output_file_name,
        mean_denorm_input_matrix=mean_denorm_input_matrix,
        mean_denorm_output_matrix=mean_denorm_output_matrix,
        mean_initial_activation=numpy.mean(
            bwo_dictionary[backwards_opt.INITIAL_ACTIVATIONS_KEY]
        ),
        mean_final_activation=numpy.mean(
            bwo_dictionary[backwards_opt.FINAL_ACTIVATIONS_KEY]
        ),
        model_file_name=bwo_dictionary[backwards_opt.MODEL_FILE_KEY],
        non_pmm_file_name=input_file_name,
        pmm_max_percentile_level=max_percentile_level
    )


def _run(input_saliency_file_name, input_gradcam_file_name, input_bwo_file_name,
         max_percentile_level, output_file_name):
    """Runs probability-matched means (PMM).

    This is effectively the main method.

    :param input_saliency_file_name: See documentation at top of file.
    :param input_gradcam_file_name: Same.
    :param input_bwo_file_name: Same.
    :param max_percentile_level: Same.
    :param output_file_name: Same.
    """

    if input_saliency_file_name not in NONE_STRINGS:
        _composite_saliency_maps(
            input_file_name=input_saliency_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return

    if input_gradcam_file_name not in NONE_STRINGS:
        _composite_gradcam(
            input_file_name=input_gradcam_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return

    if input_bwo_file_name not in NONE_STRINGS:
        _composite_backwards_opt(
            input_file_name=input_bwo_file_name,
            max_percentile_level=max_percentile_level,
            output_file_name=output_file_name)

        return


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_saliency_file_name=getattr(
            INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        input_gradcam_file_name=getattr(
            INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
        input_bwo_file_name=getattr(INPUT_ARG_OBJECT, BWO_FILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
