"""Applies trained CNN to full grids."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import predictor_io
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression

RANDOM_SEED = 6695
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SECONDS = 10800

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
EXTEND_ARG_NAME = 'extend_main_grid'
USE_MASK_ARG_NAME = 'use_mask'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
USE_ISOTONIC_ARG_NAME = 'use_isotonic_regression'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained CNN.  Will be read by `cnn.read_model`.')

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Input files therein '
    'will be found by `predictor_io.find_file` and read by '
    '`predictor_io.read_file`.')

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded front labels.  Files therein will'
    ' be found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.  If you do not want to read true labels, '
    'make this empty ("").')

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script will apply the CNN to `{0:s}` '
    'random time steps in the period `{1:s}`...`{2:s}`.'
).format(NUM_TIMES_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_TIMES_HELP_STRING = (
    'Number of times to draw randomly from `{0:s}`...`{1:s}`.  To use all times'
    ' in the period, leave this argument alone.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

EXTEND_HELP_STRING = (
    'Boolean flag.  If 1, the CNN will be applied only to grid cells on the '
    'outside of the extended NARR grid (those not in the main NARR grid).')

USE_MASK_HELP_STRING = (
    'Boolean flag.  If 1, the CNN will not be applied to masked grid cells '
    '(using the same mask used for training).  If 0, the CNN will be applied to'
    ' all grid cells.')

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for target variable.  To use the same dilation distance '
    'used for training the model, leave this argument alone.')

USE_ISOTONIC_HELP_STRING = (
    'Boolean flag.  If 1, will use isotonic regression to calibrate CNN '
    'probabilities.  If 0, will use raw CNN probabilities with no calibration.')

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`prediction_io.write_probabilities`, to exact locations determined by '
    '`prediction_io.find_file`.')

TOP_PREDICTOR_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/era5_data/processed/with_theta_w')
TOP_FRONT_DIR_NAME_DEFAULT = (
    '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=False,
    default=TOP_PREDICTOR_DIR_NAME_DEFAULT, help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=False,
    default=TOP_FRONT_DIR_NAME_DEFAULT, help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + EXTEND_ARG_NAME, type=int, required=False, default=0,
    help=EXTEND_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_MASK_ARG_NAME, type=int, required=False, default=0,
    help=USE_MASK_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False, default=-1,
    help=DILATION_DISTANCE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_ISOTONIC_ARG_NAME, type=int, required=False, default=0,
    help=USE_ISOTONIC_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING)


def _run(model_file_name, top_predictor_dir_name, top_gridded_front_dir_name,
         first_time_string, last_time_string, num_times, extend_main_grid,
         use_mask, dilation_distance_metres, use_isotonic_regression,
         output_dir_name):
    """Applies trained CNN to full grids.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param extend_main_grid: Same.
    :param use_mask: Same.
    :param dilation_distance_metres: Same.
    :param use_isotonic_regression: Same.
    :param output_dir_name: Same.
    """

    if top_gridded_front_dir_name in ['', None]:
        top_gridded_front_dir_name = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)

    if num_times > 0:
        num_times = min([num_times, len(valid_times_unix_sec)])

        numpy.random.seed(RANDOM_SEED)
        numpy.random.shuffle(valid_times_unix_sec)
        valid_times_unix_sec = valid_times_unix_sec[:num_times]

    print 'Reading CNN from: "{0:s}"...'.format(model_file_name)
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print 'Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name)
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    if dilation_distance_metres < 0:
        dilation_distance_metres = model_metadata_dict[
            cnn.DILATION_DISTANCE_KEY]

    if use_isotonic_regression:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name)

        print 'Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name)
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_model_object_by_class = None

    if use_mask:
        extend_main_grid = False
        mask_matrix = model_metadata_dict[cnn.MASK_MATRIX_KEY]
    else:
        first_predictor_file_name = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[0]
        )

        print 'Reading first predictor file: "{0:s}"...'.format(
            first_predictor_file_name)
        first_predictor_dict = predictor_io.read_file(
            netcdf_file_name=first_predictor_file_name)

        this_matrix = first_predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][0, ..., 0]

        mask_matrix = numpy.invert(numpy.isnan(this_matrix)).astype(int)

        # TODO(thunderhoser): This is a dirty old HACK.
        if extend_main_grid:
            grid_name = nwp_model_utils.dimensions_to_grid(
                num_rows=mask_matrix.shape[0], num_columns=mask_matrix.shape[1]
            )
            assert grid_name == nwp_model_utils.NAME_OF_EXTENDED_221GRID

            mask_matrix[100:-100, 100:-100] = 0
            print numpy.sum(mask_matrix)
            print mask_matrix.size

    num_times = len(valid_times_unix_sec)
    print SEPARATOR_STRING

    for i in range(num_times):
        this_class_probability_matrix, this_target_matrix = (
            cnn.apply_model_to_full_grid(
                model_object=model_object,
                top_predictor_dir_name=top_predictor_dir_name,
                top_gridded_front_dir_name=top_gridded_front_dir_name,
                valid_time_unix_sec=valid_times_unix_sec[i],
                pressure_levels_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY],
                predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
                normalization_type_string=model_metadata_dict[
                    cnn.NORMALIZATION_TYPE_KEY],
                dilation_distance_metres=dilation_distance_metres,
                isotonic_model_object_by_class=isotonic_model_object_by_class,
                mask_matrix=mask_matrix)
        )

        if top_gridded_front_dir_name is not None:
            this_target_matrix[this_target_matrix == -1] = 0

        print MINOR_SEPARATOR_STRING

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            first_time_unix_sec=valid_times_unix_sec[i],
            last_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print 'Writing gridded probabilities and labels to: "{0:s}"...'.format(
            this_output_file_name)

        prediction_io.write_probabilities(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_class_probability_matrix,
            target_matrix=this_target_matrix,
            valid_times_unix_sec=valid_times_unix_sec[[i]],
            model_file_name=model_file_name,
            target_dilation_distance_metres=dilation_distance_metres,
            used_isotonic=use_isotonic_regression)

        if i != num_times - 1:
            print SEPARATOR_STRING


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        extend_main_grid=bool(getattr(INPUT_ARG_OBJECT, EXTEND_ARG_NAME)),
        use_mask=bool(getattr(INPUT_ARG_OBJECT, USE_MASK_ARG_NAME)),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
