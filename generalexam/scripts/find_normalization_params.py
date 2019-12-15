"""Finds normalization params (mean and stdev) for each predictor."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from generalexam.ge_io import predictor_io
from generalexam.ge_utils import predictor_utils
from generalexam.machine_learning import machine_learning_utils as ml_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Files therein will be found '
    'by `predictor_io.find_file` and read by `predictor_io.read_file`.'
)

MASK_FILE_HELP_STRING = (
    'Path to mask file (will be read by `machine_learning_utils.'
    'read_narr_mask`).  Masked grid cells will not be used to compute '
    'normalization params.  If you do not want to mask grid cells, leave this '
    'argument alone.'
)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Normalization params will be based on times '
    'from `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file with normalization params.  Will be written by '
    '`predictor_io.write_normalization_params`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=False, default='',
    help=MASK_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _update_intermediate_params(intermediate_param_dict, new_data_matrix):
    """Updates intermediate normalization params.

    :param intermediate_param_dict: Dictionary with the following keys.
    intermediate_param_dict['num_values']: Number of values on which current
        estimates are based.
    intermediate_param_dict['mean_value']: Current mean.
    intermediate_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `intermediate_param_dict`.
    :return: intermediate_param_dict: Same as input, but with new estimates.
    """

    this_num_values = numpy.sum(numpy.invert(numpy.isnan(new_data_matrix)))

    these_means = numpy.array([
        intermediate_param_dict[MEAN_VALUE_KEY], numpy.nanmean(new_data_matrix)
    ])
    these_weights = numpy.array([
        intermediate_param_dict[NUM_VALUES_KEY], this_num_values
    ])
    intermediate_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    these_means = numpy.array([
        intermediate_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.nanmean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        intermediate_param_dict[NUM_VALUES_KEY], this_num_values
    ])
    intermediate_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    intermediate_param_dict[NUM_VALUES_KEY] += this_num_values

    return intermediate_param_dict


def _params_intermediate_to_final(intermediate_param_dict):
    """Converts normalization params from intermediate to final.

    :param intermediate_param_dict: See doc for `_update_intermediate_params`.
    :return: mean_value: Mean.
    :return: standard_deviation: Standard deviation.
    """

    num_values = intermediate_param_dict[NUM_VALUES_KEY]
    mean_value = intermediate_param_dict[MEAN_VALUE_KEY]
    mean_of_squares = intermediate_param_dict[MEAN_OF_SQUARES_KEY]

    multiplier = float(num_values) / (num_values - 1)
    standard_deviation = numpy.sqrt(
        multiplier * (mean_of_squares - mean_value ** 2)
    )

    return mean_value, standard_deviation


def _run(top_predictor_dir_name, mask_file_name, first_time_string,
         last_time_string, output_file_name):
    """Finds normalization params (mean and stdev) for each predictor.

    This is effectively the main method.

    :param top_predictor_dir_name: See documentation at top of file.
    :param mask_file_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if mask_file_name in ['', 'None']:
        mask_matrix = None
    else:
        print('Reading mask from: "{0:s}"...'.format(mask_file_name))
        mask_matrix = ml_utils.read_narr_mask(mask_file_name)[0]

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    num_times = len(valid_times_unix_sec)
    predictor_file_names = [None] * num_times

    for i in range(num_times):
        predictor_file_names[i] = predictor_io.find_file(
            top_directory_name=top_predictor_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            raise_error_if_missing=True)

    first_predictor_dict = predictor_io.read_file(
        netcdf_file_name=predictor_file_names[0]
    )
    field_names = first_predictor_dict[predictor_utils.FIELD_NAMES_KEY]
    pressure_levels_mb = numpy.round(
        first_predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY]
    ).astype(int)

    orig_param_dict = {
        NUM_VALUES_KEY: 0, MEAN_VALUE_KEY: 0., MEAN_OF_SQUARES_KEY: 0.
    }
    intermediate_param_dict_dict = {}
    num_fields = len(field_names)

    for k in range(num_fields):
        intermediate_param_dict_dict[field_names[k], pressure_levels_mb[k]] = (
            copy.deepcopy(orig_param_dict)
        )

    print(SEPARATOR_STRING)

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(predictor_file_names[i]))
        this_predictor_dict = predictor_io.read_file(
            netcdf_file_name=predictor_file_names[i],
            field_names_to_keep=field_names,
            pressure_levels_to_keep_mb=pressure_levels_mb)

        for k in range(num_fields):
            this_param_dict = intermediate_param_dict_dict[
                field_names[k], pressure_levels_mb[k]
            ]
            this_data_matrix = this_predictor_dict[
                predictor_utils.DATA_MATRIX_KEY
            ][0, ..., k]

            if mask_matrix is not None:
                this_data_matrix[mask_matrix == 0] = numpy.nan

            if field_names[k] == predictor_utils.HEIGHT_NAME:
                print(this_data_matrix[100:110, 100:110])

            this_param_dict = _update_intermediate_params(
                intermediate_param_dict=this_param_dict,
                new_data_matrix=this_data_matrix)

            intermediate_param_dict_dict[
                field_names[k], pressure_levels_mb[k]
            ] = this_param_dict

    print(SEPARATOR_STRING)
    mean_value_dict = {}
    standard_deviation_dict = {}

    for k in range(num_fields):
        this_param_dict = intermediate_param_dict_dict[
            field_names[k], pressure_levels_mb[k]
        ]

        this_mean, this_stdev = _params_intermediate_to_final(this_param_dict)

        mean_value_dict[field_names[k], pressure_levels_mb[k]] = this_mean
        standard_deviation_dict[field_names[k], pressure_levels_mb[k]] = (
            this_stdev
        )

        print((
            'Mean and standard deviation of "{0:s}" at {1:d} mb = '
            '{2:.4e}, {3:.4e}'
        ).format(
            field_names[k], pressure_levels_mb[k], this_mean, this_stdev
        ))

    print(SEPARATOR_STRING)
    print('Writing normalization params to file: "{0:s}"...'.format(
        output_file_name
    ))
    predictor_io.write_normalization_params(
        mean_value_dict=mean_value_dict,
        standard_deviation_dict=standard_deviation_dict,
        pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
