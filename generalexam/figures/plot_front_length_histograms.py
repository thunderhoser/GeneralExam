"""Creates histograms of warm-front and cold-front lengths."""

import glob
import numpy
import pandas
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import number_rounding as rounder
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from generalexam.ge_io import fronts_io
from generalexam.ge_utils import front_utils
from generalexam.plotting import front_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_KM = 1e-3
X_COORDS_COLUMN = 'x_coords_metres'
Y_COORDS_COLUMN = 'y_coords_metres'

NUM_BINS = 20
MIN_HISTOGRAM_LENGTH_METRES = 0.
MAX_HISTOGRAM_LENGTH_METRES = 2e6

Y_TICK_SPACING = 0.02
LINE_WIDTH = 2.
LINE_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_FILE_PATTERN = (
    '/localdata/ryan.lagerquist/general_exam/fronts/polylines/old/*.p')

OUTPUT_DIR_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'histograms')
WARM_FRONT_FILE_NAME = '{0:s}/warm_front_length_histogram.jpg'.format(
    OUTPUT_DIR_NAME)
COLD_FRONT_FILE_NAME = '{0:s}/cold_front_length_histogram.jpg'.format(
    OUTPUT_DIR_NAME)
CONCAT_FILE_NAME = '{0:s}/front_length_histograms.jpg'.format(OUTPUT_DIR_NAME)


def _project_fronts_latlng_to_narr(front_line_table):
    """Projects fronts from lat-long to NARR (x-y) coordinates.

    P = number of points in a given front

    :param front_line_table: See doc for `fronts_io.write_polylines_to_file`.
    :return: front_line_table: Same as input, but with the following extra
        columns.
    front_line_table.x_coords_metres: length-P numpy array of x-coordinates.
    front_line_table.y_coords_metres: length-P numpy array of y-coordinates.
    """

    num_fronts = len(front_line_table.index)
    projection_object = nwp_model_utils.init_model_projection(
        nwp_model_utils.NARR_MODEL_NAME)

    x_coords_by_front_metres = [numpy.array([])] * num_fronts
    y_coords_by_front_metres = [numpy.array([])] * num_fronts

    for i in range(num_fronts):
        if numpy.mod(i, 1000) == 0:
            print (
                'Have projected {0:d} of {1:d} fronts to NARR coordinates...'
            ).format(i, num_fronts)

        (x_coords_by_front_metres[i], y_coords_by_front_metres[i]
        ) = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=front_line_table[
                front_utils.LATITUDES_COLUMN].values[i],
            longitudes_deg=front_line_table[
                front_utils.LONGITUDES_COLUMN].values[i],
            projection_object=projection_object,
            model_name=nwp_model_utils.NARR_MODEL_NAME)

    print 'Projected all {0:d} fronts to NARR coordinates!'.format(num_fronts)
    return front_line_table.assign(**{
        X_COORDS_COLUMN: x_coords_by_front_metres,
        Y_COORDS_COLUMN: y_coords_by_front_metres
    })


def _get_front_lengths(front_line_table):
    """Returns length of each front.

    N = number of fronts

    :param front_line_table: N-row pandas DataFrame created by
        `_project_fronts_latlng_to_narr`.
    :return: front_lengths_metres: length-N numpy array of front lengths.
    """

    num_fronts = len(front_line_table.index)
    front_lengths_metres = numpy.full(num_fronts, numpy.nan)

    for i in range(num_fronts):
        if numpy.mod(i, 1000) == 0:
            print 'Have computed length for {0:d} of {1:d} fronts...'.format(
                i, num_fronts)

        this_closed_flag = front_utils._is_polyline_closed(
            latitudes_deg=front_line_table[
                front_utils.LATITUDES_COLUMN].values[i],
            longitudes_deg=front_line_table[
                front_utils.LONGITUDES_COLUMN].values[i])

        if this_closed_flag:
            continue

        this_linestring_object = front_utils._create_linestring(
            x_coords_metres=front_line_table[X_COORDS_COLUMN].values[i],
            y_coords_metres=front_line_table[Y_COORDS_COLUMN].values[i])
        front_lengths_metres[i] = this_linestring_object.length

    print 'Computed lengths of all {0:d} fronts!'.format(num_fronts)
    return front_lengths_metres


def _plot_histogram(num_fronts_by_bin, front_type_string, output_file_name):
    """Plots histogram of either cold-front or warm-front lengths.

    B = number of bins

    :param num_fronts_by_bin: length-B numpy array with number of fronts in each
        bin.
    :param front_type_string: Front type (must be accepted by
        `front_utils.check_front_type`).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    bin_frequencies = (
        num_fronts_by_bin.astype(float) / numpy.sum(num_fronts_by_bin)
    )

    num_bins = len(num_fronts_by_bin)
    bin_edges_metres = numpy.linspace(
        MIN_HISTOGRAM_LENGTH_METRES, MAX_HISTOGRAM_LENGTH_METRES,
        num=num_bins + 1)
    bin_width_metres = bin_edges_metres[1] - bin_edges_metres[0]
    bin_centers_metres = bin_edges_metres[:-1] + bin_width_metres / 2

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))

    if front_type_string == front_utils.WARM_FRONT_STRING_ID:
        this_colour = front_plotting.DEFAULT_WARM_FRONT_COLOUR
    else:
        this_colour = front_plotting.DEFAULT_COLD_FRONT_COLOUR

    axes_object.bar(
        bin_centers_metres * METRES_TO_KM, bin_frequencies,
        bin_width_metres * METRES_TO_KM, color=this_colour,
        edgecolor=LINE_COLOUR, linewidth=LINE_WIDTH)

    max_y_tick_value = rounder.floor_to_nearest(
        1.05 * numpy.max(bin_frequencies), Y_TICK_SPACING)
    num_y_ticks = 1 + int(numpy.round(max_y_tick_value / Y_TICK_SPACING))
    y_tick_values = numpy.linspace(0., max_y_tick_value, num=num_y_ticks)
    pyplot.yticks(y_tick_values)

    axes_object.set_xlim(
        MIN_HISTOGRAM_LENGTH_METRES * METRES_TO_KM,
        MAX_HISTOGRAM_LENGTH_METRES * METRES_TO_KM)
    axes_object.set_ylim(0., 1.05 * numpy.max(bin_frequencies))

    x_tick_values, _ = pyplot.xticks()
    x_tick_labels = []
    for i in range(len(x_tick_values)):
        this_label = '{0:d}'.format(int(numpy.round(x_tick_values[i])))
        if i == len(x_tick_values) - 1:
            this_label += '+'

        x_tick_labels.append(this_label)

    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.xlabel('Front length (km)')
    pyplot.ylabel('Frequency')

    print 'Saving figure to: "{0:s}"...'.format(output_file_name)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    pyplot.savefig(output_file_name, dpi=OUTPUT_RESOLUTION_DPI)
    pyplot.close()

    imagemagick_utils.trim_whitespace(input_file_name=output_file_name,
                                      output_file_name=output_file_name)


def _run():
    """Creates histograms of warm-front and cold-front lengths.

    This is effectively the main method.
    """

    input_file_names = glob.glob(INPUT_FILE_PATTERN)
    num_files = len(input_file_names)
    list_of_front_line_tables = [pandas.DataFrame()] * num_files

    for i in range(num_files):
        print 'Reading data from: "{0:s}"...'.format(input_file_names[i])

        list_of_front_line_tables[i] = fronts_io.read_polylines_from_file(
            input_file_names[i])
        if i == 0:
            continue

        list_of_front_line_tables[i] = list_of_front_line_tables[i].align(
            list_of_front_line_tables[0], axis=1
        )[0]

    print SEPARATOR_STRING
    front_line_table = pandas.concat(
        list_of_front_line_tables, axis=0, ignore_index=True)

    front_line_table = _project_fronts_latlng_to_narr(front_line_table)
    print SEPARATOR_STRING

    front_lengths_metres = _get_front_lengths(front_line_table)
    print SEPARATOR_STRING

    nan_flags = numpy.isnan(front_lengths_metres)
    warm_front_flags = numpy.array(
        [s == front_utils.WARM_FRONT_STRING_ID for s in
         front_line_table[front_utils.FRONT_TYPE_COLUMN].values]
    )
    cold_front_flags = numpy.array(
        [s == front_utils.COLD_FRONT_STRING_ID for s in
         front_line_table[front_utils.FRONT_TYPE_COLUMN].values]
    )

    warm_front_indices = numpy.where(
        numpy.logical_and(warm_front_flags, numpy.invert(nan_flags))
    )[0]
    cold_front_indices = numpy.where(
        numpy.logical_and(cold_front_flags, numpy.invert(nan_flags))
    )[0]

    warm_front_lengths_metres = front_lengths_metres[warm_front_indices]
    cold_front_lengths_metres = front_lengths_metres[cold_front_indices]

    print (
        'Number of fronts = {0:d} ... warm fronts with defined length = {1:d} '
        '... cold fronts with defined length = {2:d}'
    ).format(len(front_lengths_metres), len(warm_front_lengths_metres),
             len(cold_front_lengths_metres))

    _, num_warm_fronts_by_bin = histograms.create_histogram(
        input_values=warm_front_lengths_metres, num_bins=NUM_BINS,
        min_value=MIN_HISTOGRAM_LENGTH_METRES,
        max_value=MAX_HISTOGRAM_LENGTH_METRES)
    print 'Sum of bin counts for warm fronts = {0:d}'.format(
        numpy.sum(num_warm_fronts_by_bin))

    _plot_histogram(num_fronts_by_bin=num_warm_fronts_by_bin,
                    front_type_string=front_utils.WARM_FRONT_STRING_ID,
                    output_file_name=WARM_FRONT_FILE_NAME)

    _, num_cold_fronts_by_bin = histograms.create_histogram(
        input_values=cold_front_lengths_metres, num_bins=NUM_BINS,
        min_value=MIN_HISTOGRAM_LENGTH_METRES,
        max_value=MAX_HISTOGRAM_LENGTH_METRES)
    print 'Sum of bin counts for cold fronts = {0:d}'.format(
        numpy.sum(num_cold_fronts_by_bin))

    _plot_histogram(num_fronts_by_bin=num_cold_fronts_by_bin,
                    front_type_string=front_utils.COLD_FRONT_STRING_ID,
                    output_file_name=COLD_FRONT_FILE_NAME)

    print 'Concatenating figures to: "{0:s}"...'.format(CONCAT_FILE_NAME)
    imagemagick_utils.concatenate_images(
        input_file_names=[WARM_FRONT_FILE_NAME, COLD_FRONT_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME, num_panel_rows=1,
        num_panel_columns=2, output_size_pixels=OUTPUT_SIZE_PIXELS)


if __name__ == '__main__':
    _run()
