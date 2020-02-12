"""Plots gridded WPC fronts."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_io import fronts_io
from generalexam.machine_learning import machine_learning_utils as ml_utils
from generalexam.plotting import front_plotting
from generalexam.scripts import plot_gridded_stats

TIME_FORMAT = '%Y%m%d%H'
TIME_INTERVAL_SEC = 10800

NUM_GRID_ROWS, NUM_GRID_COLUMNS = nwp_model_utils.get_grid_dimensions(
    model_name=nwp_model_utils.NARR_MODEL_NAME,
    grid_name=nwp_model_utils.NAME_OF_221GRID)

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.

BORDER_COLOUR = numpy.full(3, 0.)

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = FIGURE_HEIGHT_INCHES = 15

FRONT_DIR_ARG_NAME = 'input_front_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
DILATION_DISTANCE_ARG_NAME = 'dilation_distance_metres'
USE_MODEL_PROJ_ARG_NAME = 'use_model_projection'
FIRST_LETTER_ARG_NAME = 'first_letter_label'
LETTER_INTERVAL_ARG_NAME = 'letter_interval'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

FRONT_DIR_HELP_STRING = (
    'Name of top-level directory with gridded fronts.  Files therein will be '
    'found by `fronts_io.find_gridded_file` and read by '
    '`fronts_io.read_grid_from_file`.'
)

TIME_HELP_STRING = (
    'Time (format "yyyymmddHH").  Gridded fronts will be plotted all times in '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

DILATION_DISTANCE_HELP_STRING = (
    'Dilation distance for gridded fronts.  If you do not want to dilate, leave'
    ' this argument alone.'
)
USE_MODEL_PROJ_HELP_STRING = (
    'Boolean flag.  If 1, will plot in model projection.  If 0, will plot in '
    'lat-long projection.'
)
FIRST_LETTER_HELP_STRING = (
    'Letter label for first time step.  If this is "a", the label "(a)" will be'
    ' printed at the top-left of the figure.  If you do not want labels, leave '
    'this argument alone.'
)
LETTER_INTERVAL_HELP_STRING = (
    'Interval between letter labels for successive time steps.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FRONT_DIR_ARG_NAME, type=str, required=True,
    help=FRONT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DILATION_DISTANCE_ARG_NAME, type=float, required=False, default=-1,
    help=DILATION_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_MODEL_PROJ_ARG_NAME, type=int, required=False, default=1,
    help=USE_MODEL_PROJ_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_LETTER_ARG_NAME, type=str, required=False, default='',
    help=FIRST_LETTER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LETTER_INTERVAL_ARG_NAME, type=int, required=False, default=3,
    help=LETTER_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_fronts_one_time(
        input_file_name, output_file_name, use_model_projection,
        dilation_distance_metres=None, letter_label=None):
    """Plots gridded WPC fronts at one time.

    :param input_file_name: Path to input file (will be read by
        `fronts_io.read_grid_from_file`).
    :param output_file_name: Path to output file (figure will be saved here).
    :param use_model_projection: See documentation at top of file.
    :param dilation_distance_metres: Same.
    :param letter_label: Same.
    """

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gridded_front_table = fronts_io.read_grid_from_file(input_file_name)

    gridded_front_matrix = ml_utils.front_table_to_images(
        frontal_grid_table=gridded_front_table,
        num_rows_per_image=NUM_GRID_ROWS,
        num_columns_per_image=NUM_GRID_COLUMNS
    )

    if dilation_distance_metres is not None:
        gridded_front_matrix = ml_utils.dilate_ternary_target_images(
            target_matrix=gridded_front_matrix,
            dilation_distance_metres=dilation_distance_metres,
            verbose=False
        )

    gridded_front_matrix = gridded_front_matrix[0, ...]

    basemap_dict = plot_gridded_stats.plot_basemap(
        data_matrix=gridded_front_matrix, border_colour=BORDER_COLOUR,
        use_model_projection=use_model_projection
    )

    figure_object = basemap_dict[plot_gridded_stats.FIGURE_OBJECT_KEY]
    axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
    basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]
    gridded_front_matrix = basemap_dict[plot_gridded_stats.MATRIX_TO_PLOT_KEY]
    latitude_matrix_deg = basemap_dict[plot_gridded_stats.LATITUDES_KEY]
    longitude_matrix_deg = basemap_dict[plot_gridded_stats.LONGITUDES_KEY]

    front_plotting.plot_labels_on_general_grid(
        label_matrix=gridded_front_matrix,
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        axes_object=axes_object, basemap_object=basemap_object, opacity=1.
    )

    if letter_label is not None:
        plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=0. if use_model_projection else 0.075
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_front_dir_name, first_time_string, last_time_string,
         dilation_distance_metres, use_model_projection, first_letter_label,
         letter_interval, output_dir_name):
    """Plots gridded WPC fronts.

    This is effectively the main method.

    :param top_front_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param dilation_distance_metres: Same.
    :param use_model_projection: Same.
    :param first_letter_label: Same.
    :param letter_interval: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if dilation_distance_metres <= 0:
        dilation_distance_metres = None
    if first_letter_label in ['', 'None']:
        first_letter_label = None

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )
    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    letter_label = None

    for this_time_unix_sec in valid_times_unix_sec:
        this_front_file_name = fronts_io.find_gridded_file(
            top_directory_name=top_front_dir_name,
            valid_time_unix_sec=this_time_unix_sec,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_front_file_name):
            continue

        this_figure_file_name = '{0:s}/gridded_fronts_{1:s}.jpg'.format(
            output_dir_name,
            time_conversion.unix_sec_to_string(this_time_unix_sec, TIME_FORMAT)
        )

        if first_letter_label is not None:
            if letter_label is None:
                letter_label = first_letter_label
            else:
                letter_label = chr(
                    ord(letter_label) + letter_interval
                )

        _plot_fronts_one_time(
            input_file_name=this_front_file_name,
            output_file_name=this_figure_file_name,
            dilation_distance_metres=dilation_distance_metres,
            use_model_projection=use_model_projection, letter_label=letter_label
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_front_dir_name=getattr(INPUT_ARG_OBJECT, FRONT_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        dilation_distance_metres=getattr(
            INPUT_ARG_OBJECT, DILATION_DISTANCE_ARG_NAME
        ),
        use_model_projection=bool(getattr(
            INPUT_ARG_OBJECT, USE_MODEL_PROJ_ARG_NAME
        )),
        first_letter_label=getattr(INPUT_ARG_OBJECT, FIRST_LETTER_ARG_NAME),
        letter_interval=getattr(INPUT_ARG_OBJECT, LETTER_INTERVAL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
