"""Writes Slurm file to run downsize_machine_learning_examples.py."""

import argparse
from gewittergefahr.gg_io import slurm_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from generalexam.scripts import \
    downsize_machine_learning_examples as downsize_ml

INPUT_TIME_FORMAT = downsize_ml.INPUT_TIME_FORMAT
TIME_INTERVAL_SECONDS = 10800

PYTHON_EXE_NAME = '/home/ralager/anaconda2/bin/python2.7'
PYTHON_SCRIPT_NAME = (
    '/condo/swatwork/ralager/generalexam_master/generalexam/scripts/'
    'downsize_machine_learning_examples.py')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = downsize_ml.add_input_arguments(INPUT_ARG_PARSER)
INPUT_ARG_PARSER = slurm_io.add_input_arguments(
    argument_parser_object=INPUT_ARG_PARSER, use_array=False,
    use_spc_dates=False)

FIRST_TIME_INPUT_ARG = 'first_time_string'
LAST_TIME_INPUT_ARG = 'last_time_string'
MAX_SIMULTANEOUS_TASKS_INPUT_ARG = 'max_num_simultaneous_tasks'

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  This script creates downsized ML examples for'
    ' all 3-hour time steps from `{0:s}`...`{1:s}`.').format(
        FIRST_TIME_INPUT_ARG, LAST_TIME_INPUT_ARG)
MAX_SIMULTANEOUS_TASKS_HELP_STRING = (
    'Maximum number of simultaneous tasks (in array) that can be running.')

MAX_SIMULTANEOUS_TASKS_DEFAULT = 50

INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_INPUT_ARG, type=str, required=True, help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_INPUT_ARG, type=str, required=True, help=TIME_HELP_STRING)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_SIMULTANEOUS_TASKS_INPUT_ARG, type=int, required=False,
    default=MAX_SIMULTANEOUS_TASKS_DEFAULT,
    help=MAX_SIMULTANEOUS_TASKS_HELP_STRING)


def _write_slurm_file(
        first_time_string, last_time_string, max_num_simultaneous_tasks,
        email_address, partition_name, slurm_file_name, input_narr_dir_name,
        input_frontal_grid_dir_name, narr_predictor_names, pressure_level_mb,
        dilation_distance_metres, positive_fraction, num_rows_in_half_window,
        num_columns_in_half_window, output_dir_name):
    """Writes Slurm file to run downsize_machine_learning_examples.py.

    :param first_time_string: Time (format "yyyymmddHH").  This script creates
        downsized ML examples for all 3-hour time steps from `first_time_string`
        ...`last_time_string`.
    :param last_time_string: See above.
    :param max_num_simultaneous_tasks: Max number of tasks (SPC dates) running
        at once.
    :param email_address: Slurm notifications will be sent to this e-mail
        address.
    :param partition_name: Job will be run on this partition of the
        supercomputer.
    :param slurm_file_name: Path to output file.
    :param input_narr_dir_name: See documentation for
        downsize_machine_learning_examples.py.
    :param input_frontal_grid_dir_name: Same.
    :param narr_predictor_names: Same.
    :param pressure_level_mb: Same.
    :param dilation_distance_metres: Same.
    :param positive_fraction: Same.
    :param num_rows_in_half_window: Same.
    :param num_columns_in_half_window: Same.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, INPUT_TIME_FORMAT)
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, INPUT_TIME_FORMAT)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SECONDS, include_endpoint=True)
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, INPUT_TIME_FORMAT)
        for t in valid_times_unix_sec]
    num_times = len(valid_time_strings)

    slurm_file_handle = slurm_io.write_slurm_file_header(
        slurm_file_name=slurm_file_name, email_address=email_address,
        partition_name=partition_name, use_array=True,
        num_array_tasks=num_times,
        max_num_simultaneous_tasks=max_num_simultaneous_tasks)

    # Write list of valid times to file.
    slurm_file_handle.write('VALID_TIME_STRINGS=(')
    for i in range(num_times):
        if i == 0:
            slurm_file_handle.write('"{0:s}"'.format(valid_time_strings[i]))
        else:
            slurm_file_handle.write(' "{0:s}"'.format(valid_time_strings[i]))
    slurm_file_handle.write(')\n\n')

    # When each task is run, the following statement will write the task ID and
    # corresponding time step to the Slurm log file.
    slurm_file_handle.write(
        'this_valid_time_string=${VALID_TIME_STRINGS[$SLURM_ARRAY_TASK_ID]}\n')
    slurm_file_handle.write(
        'echo "Array-task ID = ${SLURM_ARRAY_TASK_ID} ... '
        'valid time = ${this_valid_time_string}"\n\n')

    # The following statement calls downsize_machine_learning_examples.py for
    # the given task (time step).
    slurm_file_handle.write(
        '"{0:s}" -u "{1:s}" --{2:s}='.format(
            PYTHON_EXE_NAME, PYTHON_SCRIPT_NAME, downsize_ml.TIME_INPUT_ARG))
    slurm_file_handle.write('"${this_valid_time_string}"')

    slurm_file_handle.write(
        (' --{0:s}="{1:s}" --{2:s}="{3:s}" --{4:s}={5:d} --{6:s}={7:.6f} '
         '--{8:s}={9:.6f} --{10:s}={11:d} --{12:s}={13:d} '
         '--{14:s}="{15:s}"').format(
             downsize_ml.NARR_DIR_INPUT_ARG, input_narr_dir_name,
             downsize_ml.FRONTAL_GRID_DIR_INPUT_ARG,
             input_frontal_grid_dir_name,
             downsize_ml.PRESSURE_LEVEL_INPUT_ARG, pressure_level_mb,
             downsize_ml.DILATION_DISTANCE_INPUT_ARG, dilation_distance_metres,
             downsize_ml.POSITIVE_FRACTION_INPUT_ARG, positive_fraction,
             downsize_ml.NUM_ROWS_INPUT_ARG, num_rows_in_half_window,
             downsize_ml.NUM_COLUMNS_INPUT_ARG, num_columns_in_half_window,
             downsize_ml.OUTPUT_DIR_INPUT_ARG, output_dir_name))

    slurm_file_handle.write(' --{0:s}'.format(
        downsize_ml.PREDICTOR_NAMES_INPUT_ARG))
    for this_predictor_name in narr_predictor_names:
        slurm_file_handle.write(' "{0:s}"'.format(this_predictor_name))

    slurm_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _write_slurm_file(
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_INPUT_ARG),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_INPUT_ARG),
        max_num_simultaneous_tasks=getattr(
            INPUT_ARG_OBJECT, MAX_SIMULTANEOUS_TASKS_INPUT_ARG),
        email_address=getattr(
            INPUT_ARG_OBJECT, slurm_io.EMAIL_ADDRESS_INPUT_ARG),
        partition_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.PARTITION_NAME_INPUT_ARG),
        slurm_file_name=getattr(
            INPUT_ARG_OBJECT, slurm_io.SLURM_FILE_INPUT_ARG),
        input_narr_dir_name=getattr(
            INPUT_ARG_OBJECT, downsize_ml.NARR_DIR_INPUT_ARG),
        input_frontal_grid_dir_name=getattr(
            INPUT_ARG_OBJECT, downsize_ml.FRONTAL_GRID_DIR_INPUT_ARG),
        narr_predictor_names=getattr(
            INPUT_ARG_OBJECT, downsize_ml.PREDICTOR_NAMES_INPUT_ARG),
        pressure_level_mb=getattr(
            INPUT_ARG_OBJECT, downsize_ml.PRESSURE_LEVEL_INPUT_ARG),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, downsize_ml.DILATION_DISTANCE_INPUT_ARG),
        positive_fraction=getattr(
            INPUT_ARG_OBJECT, downsize_ml.POSITIVE_FRACTION_INPUT_ARG),
        num_rows_in_half_window=getattr(
            INPUT_ARG_OBJECT, downsize_ml.NUM_ROWS_INPUT_ARG),
        num_columns_in_half_window=getattr(
            INPUT_ARG_OBJECT, downsize_ml.NUM_COLUMNS_INPUT_ARG),
        output_dir_name=getattr(
            INPUT_ARG_OBJECT, downsize_ml.OUTPUT_DIR_INPUT_ARG))
