"""Applies trained CNN to full grids."""

import argparse
import numpy
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.ge_io import predictor_io
from generalexam.ge_io import prediction_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import cnn
from generalexam.machine_learning import isotonic_regression
from generalexam.ge_utils import neigh_evaluation

RANDOM_SEED = 6695
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

K.set_session(K.tf.Session(config=K.tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)))

METRES_TO_KM = 0.001
TIME_INTERVAL_SECONDS = 10800
INPUT_TIME_FORMAT = '%Y%m%d%H'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
FRONT_DIR_ARG_NAME = 'input_gridded_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_TIMES_ARG_NAME = 'num_times'
USE_MASK_ARG_NAME = 'use_mask'
NEIGH_DISTANCE_ARG_NAME = 'neigh_eval_distance_metres'
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

USE_MASK_HELP_STRING = (
    'Boolean flag.  If 1, the CNN will not be applied to grid cells that were '
    'masked during training.  If 0, will be applied to all grid cells.')

NEIGH_DISTANCE_HELP_STRING = (
    'Neighbourhood distance for eventual neighbourhood evaluation.  If you plan'
    ' to use multiple neighbourhood distances, make this the largest of them '
    'all.')

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

# TOP_PREDICTOR_DIR_NAME_DEFAULT = (
#     '/condo/swatwork/ralager/era5_data/processed/with_theta_w'
# )
# TOP_FRONT_DIR_NAME_DEFAULT = (
#     '/condo/swatwork/ralager/fronts_netcdf/narr_grids_no_dilation'
# )

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=True,
    help=FRONT_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TIMES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_TIMES_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + USE_MASK_ARG_NAME, type=int, required=True,
    help=USE_MASK_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + NEIGH_DISTANCE_ARG_NAME, type=float, required=True,
    help=NEIGH_DISTANCE_HELP_STRING)

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
         first_time_string, last_time_string, num_times, use_mask,
         neigh_eval_distance_metres, dilation_distance_metres,
         use_isotonic_regression, output_dir_name):
    """Applies trained CNN to full grids.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_gridded_front_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_times: Same.
    :param use_mask: Same.
    :param neigh_eval_distance_metres: Same.
    :param dilation_distance_metres: Same.
    :param use_isotonic_regression: Same.
    :param output_dir_name: Same.
    """

    if top_gridded_front_dir_name in ['', None]:
        top_gridded_front_dir_name = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True
    )

    if num_times > 0:
        num_times = min([num_times, len(valid_times_unix_sec)])

        numpy.random.seed(RANDOM_SEED)
        numpy.random.shuffle(valid_times_unix_sec)
        valid_times_unix_sec = valid_times_unix_sec[:num_times]

    print('Reading CNN from: "{0:s}"...'.format(model_file_name))
    model_object = cnn.read_model(model_file_name)

    model_metafile_name = cnn.find_metafile(model_file_name=model_file_name)
    print('Reading CNN metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = cnn.read_metadata(model_metafile_name)

    # try:
    #     model_metadata_dict = cnn.read_metadata(model_metafile_name)
    # except UnicodeDecodeError:
    #     predictor_names = 2 * [
    #         predictor_utils.U_WIND_GRID_RELATIVE_NAME,
    #         predictor_utils.V_WIND_GRID_RELATIVE_NAME,
    #         predictor_utils.TEMPERATURE_NAME,
    #         predictor_utils.SPECIFIC_HUMIDITY_NAME
    #     ]
    #
    #     pressure_levels_mb = numpy.array(
    #         [1013, 1013, 1013, 1013, 850, 850, 850, 850], dtype=int
    #     )
    #
    #     model_metadata_dict = {
    #         cnn.DILATION_DISTANCE_KEY: 50000.,
    #         cnn.PREDICTOR_NAMES_KEY: predictor_names,
    #         cnn.PRESSURE_LEVELS_KEY: pressure_levels_mb,
    #         cnn.NORMALIZATION_TYPE_KEY: 'z_score'
    #     }

    if dilation_distance_metres < 0:
        dilation_distance_metres = model_metadata_dict[
            cnn.DILATION_DISTANCE_KEY]

    if use_isotonic_regression:
        isotonic_file_name = isotonic_regression.find_model_file(
            base_model_file_name=model_file_name)

        print('Reading isotonic-regression models from: "{0:s}"...'.format(
            isotonic_file_name
        ))
        isotonic_model_object_by_class = (
            isotonic_regression.read_model_for_each_class(isotonic_file_name)
        )
    else:
        isotonic_model_object_by_class = None

    if use_mask:
        mask_matrix = model_metadata_dict[cnn.MASK_MATRIX_KEY]
    else:
        first_predictor_file_name = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[0]
        )
        first_predictor_dict = predictor_io.read_file(
            netcdf_file_name=first_predictor_file_name
        )

        this_matrix = first_predictor_dict[
            predictor_utils.DATA_MATRIX_KEY
        ][0, ..., 0]

        mask_matrix = numpy.invert(numpy.isnan(this_matrix)).astype(int)

    print('Dilating mask with {0:.1f}-km buffer...'.format(
        neigh_eval_distance_metres * METRES_TO_KM
    ))

    orig_num_unmasked_pts = numpy.sum(mask_matrix == 1)
    mask_matrix = neigh_evaluation.dilate_narr_mask(
        narr_mask_matrix=mask_matrix,
        neigh_distance_metres=neigh_eval_distance_metres
    )
    num_unmasked_grid_pts = numpy.sum(mask_matrix == 1)

    print((
        'Number of unmasked grid points for training = {0:d} ... for gridded '
        'inference = {1:d}'
    ).format(
        orig_num_unmasked_pts, num_unmasked_grid_pts
    ))

    num_times = len(valid_times_unix_sec)
    print(SEPARATOR_STRING)

    for i in range(num_times):
        this_prob_matrix, this_target_matrix = cnn.apply_model_to_full_grid(
            model_object=model_object,
            top_predictor_dir_name=top_predictor_dir_name,
            top_gridded_front_dir_name=top_gridded_front_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            pressure_levels_mb=model_metadata_dict[cnn.PRESSURE_LEVELS_KEY],
            predictor_names=model_metadata_dict[cnn.PREDICTOR_NAMES_KEY],
            normalization_file_name=model_metadata_dict[
                cnn.NORMALIZATION_FILE_KEY
            ],
            normalization_type_string=model_metadata_dict[
                cnn.NORMALIZATION_TYPE_KEY
            ],
            dilation_distance_metres=dilation_distance_metres,
            isotonic_model_object_by_class=isotonic_model_object_by_class,
            mask_matrix=mask_matrix
        )

        if top_gridded_front_dir_name is not None:
            this_target_matrix[this_target_matrix == -1] = 0

        print(MINOR_SEPARATOR_STRING)

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            first_time_unix_sec=valid_times_unix_sec[i],
            last_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=False)

        print('Writing gridded probabilities and labels to: "{0:s}"...'.format(
            this_output_file_name
        ))

        prediction_io.write_probabilities(
            netcdf_file_name=this_output_file_name,
            class_probability_matrix=this_prob_matrix,
            target_matrix=this_target_matrix,
            valid_times_unix_sec=valid_times_unix_sec[[i]],
            model_file_name=model_file_name,
            target_dilation_distance_metres=dilation_distance_metres,
            used_isotonic=use_isotonic_regression)

        if i != num_times - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_gridded_front_dir_name=getattr(
            INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_times=getattr(INPUT_ARG_OBJECT, NUM_TIMES_ARG_NAME),
        use_mask=bool(getattr(INPUT_ARG_OBJECT, USE_MASK_ARG_NAME)),
        neigh_eval_distance_metres=getattr(
            INPUT_ARG_OBJECT, NEIGH_DISTANCE_ARG_NAME
        ),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        use_isotonic_regression=bool(getattr(
            INPUT_ARG_OBJECT, USE_ISOTONIC_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
