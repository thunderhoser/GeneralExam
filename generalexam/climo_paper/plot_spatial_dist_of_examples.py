"""Plots spatial distribution of examples."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import nwp_plotting
from gewittergefahr.plotting import imagemagick_utils
from generalexam.machine_learning import learning_examples_io as examples_io

MODEL_NAME = nwp_model_utils.NARR_MODEL_NAME
GRID_NAME = nwp_model_utils.NAME_OF_221GRID

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
BORDER_COLOUR = numpy.full(3, 0.)

MAX_COLOUR_PERCENTILE = 99.
COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILES_ARG_NAME = 'input_metafile_names'
SUBSET_NAMES_ARG_NAME = 'subset_names'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing IDs for a subset of examples'
    '.  These files will be read by `learning_examples_io.read_example_ids`.'
)
SUBSET_NAMES_HELP_STRING = (
    'List of subset names (one per input file).  The list should be space-'
    'separated.  In each list item, underscores will be replaced with spaces.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SUBSET_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=SUBSET_NAMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _compute_one_distribution(example_metafile_name):
    """Computes distribution for one subset of examples.

    M = number of rows in grid
    N = number of columns in grid

    :param example_metafile_name: Path to input file.  Will be read by
        `learning_examples_io.read_example_ids`.
    :return: count_matrix: M-by-N numpy array.  count_matrix[i, j] is the number
        of examples centered on grid point [i, j].
    """

    print('Reading example IDs from: "{0:s}"...'.format(example_metafile_name))
    example_id_strings = examples_io.read_example_ids(example_metafile_name)
    _, row_indices, column_indices = examples_io.example_ids_to_metadata(
        example_id_strings
    )

    num_grid_rows, num_grid_columns = nwp_model_utils.get_grid_dimensions(
        model_name=MODEL_NAME, grid_name=GRID_NAME
    )
    count_matrix = numpy.full((num_grid_rows, num_grid_columns), -1, dtype=int)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            count_matrix[i, j] = numpy.sum(numpy.logical_and(
                row_indices == i, column_indices == j
            ))

    return count_matrix


def _plot_one_distribution(count_matrix, output_file_name, title_string=None,
                           panel_letter=None):
    """Plots distribution for one subset of examples.

    :param count_matrix: See doc for `_compute_one_distribution`.
    :param output_file_name: Path to output file (figure will be saved here).
    :param title_string: Title (will be added above figure).  If you do not want
        a title, make this None.
    :param panel_letter: Panel letter.  For example, if the letter is "a", will
        add "(a)" at top-left of figure, assuming that it will eventually be a
        panel in a larger figure.  If you do not want a panel letter, make this
        None.
    """

    full_grid_row_limits, full_grid_column_limits = (
        nwp_plotting.latlng_limits_to_rowcol_limits(
            min_latitude_deg=MIN_LATITUDE_DEG,
            max_latitude_deg=MAX_LATITUDE_DEG,
            min_longitude_deg=MIN_LONGITUDE_DEG,
            max_longitude_deg=MAX_LONGITUDE_DEG,
            model_name=MODEL_NAME, grid_id=GRID_NAME)
    )

    matrix_to_plot = count_matrix[
        full_grid_row_limits[0]:(full_grid_row_limits[1] + 1),
        full_grid_column_limits[0]:(full_grid_column_limits[1] + 1)
    ]
    matrix_to_plot = matrix_to_plot.astype(matrix_to_plot)
    matrix_to_plot[matrix_to_plot == 0] = numpy.nan

    _, axes_object, basemap_object = nwp_plotting.init_basemap(
        model_name=MODEL_NAME, grid_id=GRID_NAME,
        first_row_in_full_grid=full_grid_row_limits[0],
        last_row_in_full_grid=full_grid_row_limits[1],
        first_column_in_full_grid=full_grid_column_limits[0],
        last_column_in_full_grid=full_grid_column_limits[1]
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_states_and_provinces(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS
    )

    min_colour_value = numpy.nanpercentile(
        matrix_to_plot, 100. - MAX_COLOUR_PERCENTILE
    )
    max_colour_value = numpy.nanpercentile(
        matrix_to_plot, MAX_COLOUR_PERCENTILE
    )

    print(max_colour_value)
    print(numpy.nanmax(matrix_to_plot))

    nwp_plotting.plot_subgrid(
        field_matrix=matrix_to_plot, model_name=MODEL_NAME, grid_id=GRID_NAME,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value,
        first_row_in_full_grid=full_grid_row_limits[0],
        first_column_in_full_grid=full_grid_column_limits[0]
    )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_value=min_colour_value, max_value=max_colour_value,
        padding=0.05, orientation_string='horizontal',
        extend_min=True, extend_max=True
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:d}'.format(int(numpy.round(v))) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    if title_string is not None:
        axes_object.set_title(title_string)

    if panel_letter is not None:
        plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(panel_letter)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


def _run(example_metafile_names, subset_names, output_dir_name):
    """Plots spatial distribution of examples.

    This is effectively the main method.

    :param example_metafile_names: See documentation at top of file.
    :param subset_names: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_subsets = len(example_metafile_names)
    expected_dim = numpy.array([num_subsets], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(subset_names), exact_dimensions=expected_dim
    )

    subset_names_abbrev = [
        n.replace('_', '-').lower() for n in subset_names
    ]
    subset_names_verbose = [n.replace('_', ' ') for n in subset_names]

    panel_file_names = [
        '{0:s}/spatial_distribution_{1:s}.jpg'.format(output_dir_name, s)
        for s in subset_names_abbrev
    ]
    panel_letter = None

    for i in range(num_subsets):
        this_count_matrix = _compute_one_distribution(example_metafile_names[i])

        if panel_letter is None:
            panel_letter = 'a'
        else:
            panel_letter = chr(ord(panel_letter) + 1)

        _plot_one_distribution(
            count_matrix=this_count_matrix,
            output_file_name=panel_file_names[i],
            title_string=subset_names_verbose[i],
            panel_letter=panel_letter
        )
        print('\n')

    concat_file_name = '{0:s}/spatial_distribution_concat.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    num_panel_rows = int(numpy.ceil(
        numpy.sqrt(num_subsets)
    ))
    num_panel_columns = int(numpy.floor(
        float(num_subsets) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_metafile_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        subset_names=getattr(INPUT_ARG_OBJECT, SUBSET_NAMES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
