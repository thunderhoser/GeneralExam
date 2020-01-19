"""Plots number of WF and CF labels at each grid cell over a time period."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_utils import climatology_utils as climo_utils
from generalexam.plotting import prediction_plotting
from generalexam.scripts import plot_gridded_stats

TIME_INTERVAL_SEC = 10800
MASK_IF_NUM_LABELS_BELOW = 100

NUM_PARALLELS = 8
NUM_MERIDIANS = 8
BORDER_COLOUR = numpy.full(3, 0.)

MIN_LATITUDE_DEG = 5.
MIN_LONGITUDE_DEG = 200.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 310.

TITLE_FONT_SIZE = 16
TITLE_TIME_FORMAT = '%Y-%m-%d-%H'
FIGURE_RESOLUTION_DPI = 300

COUNT_FILE_ARG_NAME = 'input_count_file_name'
MONTE_CARLO_FILE_ARG_NAME = 'input_monte_carlo_file_name'
LATLNG_GRID_ARG_NAME = 'use_latlng_grid'
WF_COLOUR_MAP_ARG_NAME = 'wf_colour_map_name'
CF_COLOUR_MAP_ARG_NAME = 'cf_colour_map_name'
PLOT_FREQUENCY_ARG_NAME = 'plot_frequency'
DIFFERENCE_CMAP_ARG_NAME = 'diff_colour_map_name'
MAX_PERCENTILE_ARG_NAME = 'max_colour_percentile'
WF_COLOUR_MAX_ARG_NAME = 'wf_colour_max'
CF_COLOUR_MAX_ARG_NAME = 'cf_colour_max'
MC_COLOUR_MAXIMA_ARG_NAME = 'monte_carlo_colour_maxima'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

COUNT_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_gridded_counts`).  If you want to plot results of '
    'a Monte Carlo significance test, leave this argument alone.'
)
MONTE_CARLO_FILE_HELP_STRING = (
    'Path to input file (will be read by '
    '`climatology_utils.read_monte_carlo_test`).  If you just want to plot '
    'counts, leave this argument alone.'
)
LATLNG_GRID_HELP_STRING = (
    'Boolean flag.  If 1, will plot on lat-long grid.  If 0, will plot on '
    'native grid.'
)
WF_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for warm-front counts.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
CF_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for cold-front counts.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)

DIFFERENCE_CMAP_HELP_STRING = (
    'Name of colour map for composite difference (used only if `{0:s}` is not '
    'empty).  Must be accepted by `matplotlib.pyplot.get_cmap`.'
).format(MONTE_CARLO_FILE_ARG_NAME)

PLOT_FREQUENCY_HELP_STRING = (
    'Boolean flag.  If 1, will plot frequency (fraction of time steps with '
    'front).  If 0, will plot raw count (number of fronts).'
)

MAX_PERCENTILE_HELP_STRING = (
    'Percentile used to set max value in each colour scheme.  If you want to '
    'set max values directly, make this negative and use other input args.'
).format(MAX_PERCENTILE_ARG_NAME)

WF_COLOUR_MAX_HELP_STRING = (
    '[used only if `{0:s}` is negative] Max value for warm-front map (count or '
    'frequency).'
)
CF_COLOUR_MAX_HELP_STRING = (
    '[used only if `{0:s}` is negative] Max value for cold-front map (count or '
    'frequency).'
)
MC_COLOUR_MAXIMA_HELP_STRING = (
    '[used only if `{0:s}` is negative] List of max values for the two types of'
    ' Monte Carlo maps.  This list should have 2 elements [means, differences].'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + COUNT_FILE_ARG_NAME, type=str, required=False, default='',
    help=COUNT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MONTE_CARLO_FILE_ARG_NAME, type=str, required=False, default='',
    help=MONTE_CARLO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATLNG_GRID_ARG_NAME, type=int, required=False, default=0,
    help=LATLNG_GRID_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WF_COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlOrRd',
    help=WF_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CF_COLOUR_MAP_ARG_NAME, type=str, required=False, default='YlGnBu',
    help=CF_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DIFFERENCE_CMAP_ARG_NAME, type=str, required=False, default='bwr',
    help=DIFFERENCE_CMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FREQUENCY_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_FREQUENCY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WF_COLOUR_MAX_ARG_NAME, type=float, required=False, default=0.,
    help=WF_COLOUR_MAX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CF_COLOUR_MAX_ARG_NAME, type=float, required=False, default=0.,
    help=CF_COLOUR_MAX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MC_COLOUR_MAXIMA_ARG_NAME, type=float, nargs=2, required=False,
    default=numpy.full(2, 0.), help=MC_COLOUR_MAXIMA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_front_type(
        count_or_frequency_matrix, use_latlng_grid, colour_map_object,
        plot_frequency, title_string, output_file_name,
        max_colour_percentile=None, max_colour_value=None):
    """Plots gridded counts or frequencies for one front type.

    :param count_or_frequency_matrix: 2-D numpy array with number or frequency
        of fronts at each grid cell.
    :param use_latlng_grid: See documentation at top of file.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param plot_frequency: Same.
    :param title_string: Title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    :param max_colour_percentile: [may be None]
        Max percentile in colour scheme.  The max value will be the [q]th
        percentile of all values in `count_or_frequency_matrix`, where q =
        `max_colour_percentile`.
    :param max_colour_value: [used only if `max_colour_percentile is None`]
        Max value in colour scheme.
    """

    if use_latlng_grid:
        num_grid_rows = count_or_frequency_matrix.shape[0]
        num_grid_columns = count_or_frequency_matrix.shape[1]
        grid_name = nwp_model_utils.dimensions_to_grid(
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )

        latitude_matrix_deg, longitude_matrix_deg = (
            nwp_model_utils.get_latlng_grid_point_matrices(
                model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name)
        )

        x_matrix_metres, y_matrix_metres = nwp_model_utils.project_latlng_to_xy(
            latitudes_deg=latitude_matrix_deg,
            longitudes_deg=longitude_matrix_deg,
            model_name=nwp_model_utils.NARR_MODEL_NAME, grid_name=grid_name
        )

        matrix_to_plot = numpy.ma.masked_where(
            numpy.isnan(count_or_frequency_matrix), count_or_frequency_matrix
        )

        if max_colour_percentile is not None:
            max_colour_value = numpy.nanpercentile(
                count_or_frequency_matrix, max_colour_percentile
            )

        colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

        _, axes_object, basemap_object = (
            plotting_utils.create_equidist_cylindrical_map(
                min_latitude_deg=3., max_latitude_deg=82.,
                min_longitude_deg=178., max_longitude_deg=322.)
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
            num_parallels=NUM_PARALLELS, line_colour=BORDER_COLOUR
        )
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            num_meridians=NUM_MERIDIANS, line_colour=BORDER_COLOUR
        )

        basemap_object.pcolormesh(
            x_matrix_metres, y_matrix_metres, matrix_to_plot,
            cmap=colour_map_object, norm=colour_norm_object,
            vmin=0., vmax=max_colour_value, shading='flat',
            edgecolors='None', ax=axes_object, zorder=-1e12, alpha=1.
        )
    else:
        basemap_dict = plot_gridded_stats._plot_basemap(
            count_or_frequency_matrix
        )
        axes_object = basemap_dict[plot_gridded_stats.AXES_OBJECT_KEY]
        basemap_object = basemap_dict[plot_gridded_stats.BASEMAP_OBJECT_KEY]

        full_grid_name = basemap_dict[plot_gridded_stats.FULL_GRID_NAME_KEY]
        full_grid_row_limits = basemap_dict[
            plot_gridded_stats.FULL_GRID_ROWS_KEY
        ]
        full_grid_column_limits = basemap_dict[
            plot_gridded_stats.FULL_GRID_COLUMNS_KEY
        ]

        matrix_to_plot = basemap_dict[plot_gridded_stats.SUBGRID_DATA_KEY]

        if max_colour_percentile is not None:
            max_colour_value = numpy.nanpercentile(
                matrix_to_plot, max_colour_percentile
            )

        colour_norm_object = pyplot.Normalize(vmin=0, vmax=max_colour_value)

        prediction_plotting.plot_gridded_counts(
            count_or_frequency_matrix=matrix_to_plot,
            axes_object=axes_object, basemap_object=basemap_object,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            full_grid_name=full_grid_name,
            first_row_in_full_grid=full_grid_row_limits[0],
            first_column_in_full_grid=full_grid_column_limits[0]
        )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', padding=0.05,
        extend_min=False, extend_max=True, fraction_of_axis_length=0.9
    )

    tick_values = colour_bar_object.ax.get_xticks()
    colour_bar_object.ax.set_xticks(tick_values)

    if plot_frequency:
        tick_strings = ['{0:.2f}'.format(x) for x in tick_values]
    else:
        tick_strings = ['{0:.1f}'.format(x) for x in tick_values]

    colour_bar_object.ax.set_xticklabels(tick_strings)
    pyplot.title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    pyplot.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close()


def _run(count_file_name, monte_carlo_file_name, use_latlng_grid,
         wf_colour_map_name, cf_colour_map_name, diff_colour_map_name,
         plot_frequency, max_colour_percentile, wf_colour_max, cf_colour_max,
         monte_carlo_colour_maxima, output_dir_name):
    """Plots number of WF and CF labels at each grid cell over a time period.

    This is effectively the main method.

    :param count_file_name: See documentation at top of file.
    :param monte_carlo_file_name: Same.
    :param use_latlng_grid: Same.
    :param wf_colour_map_name: Same.
    :param cf_colour_map_name: Same.
    :param diff_colour_map_name: Same.
    :param plot_frequency: Same.
    :param max_colour_percentile: Same.
    :param wf_colour_max: Same.
    :param cf_colour_max: Same.
    :param monte_carlo_colour_maxima: Same.
    :param output_dir_name: Same.
    """

    if max_colour_percentile <= 0:
        max_colour_percentile = None
    else:
        error_checking.assert_is_greater(max_colour_percentile, 50.)
        error_checking.assert_is_leq(max_colour_percentile, 100.)

    if count_file_name in ['', 'None']:
        count_file_name = None
    if monte_carlo_file_name in ['', 'None']:
        monte_carlo_file_name = None

    wf_colour_map_object = pyplot.get_cmap(wf_colour_map_name)
    cf_colour_map_object = pyplot.get_cmap(cf_colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if count_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(count_file_name))
        front_count_dict = climo_utils.read_gridded_counts(count_file_name)

        first_time_unix_sec = front_count_dict[climo_utils.FIRST_TIME_KEY]
        last_time_unix_sec = front_count_dict[climo_utils.LAST_TIME_KEY]

        first_time_string = time_conversion.unix_sec_to_string(
            first_time_unix_sec, TITLE_TIME_FORMAT
        )
        last_time_string = time_conversion.unix_sec_to_string(
            last_time_unix_sec, TITLE_TIME_FORMAT
        )

        if plot_frequency:
            warm_front_matrix = front_count_dict[climo_utils.NUM_WF_LABELS_KEY]
            cold_front_matrix = front_count_dict[climo_utils.NUM_CF_LABELS_KEY]

            num_times = len(front_count_dict[climo_utils.PREDICTION_FILES_KEY])
            warm_front_matrix = warm_front_matrix.astype(float) / num_times
            cold_front_matrix = cold_front_matrix.astype(float) / num_times

            wf_title_string = 'Frequency'
        else:
            warm_front_matrix = front_count_dict[climo_utils.NUM_UNIQUE_WF_KEY]
            cold_front_matrix = front_count_dict[climo_utils.NUM_UNIQUE_CF_KEY]

            first_year = int(first_time_string[:4])
            first_year_first_time_unix_sec = (
                time_conversion.first_and_last_times_in_year(first_year)[0]
            )

            last_year = int(last_time_string[:4])
            last_year_last_time_unix_sec = (
                time_conversion.first_and_last_times_in_year(last_year)[1]
            )
            last_year_last_time_unix_sec += 1 - TIME_INTERVAL_SEC

            file_has_full_years = (
                first_time_unix_sec == first_year_first_time_unix_sec and
                last_time_unix_sec == last_year_last_time_unix_sec
            )

            if file_has_full_years:
                num_years = last_year - first_year + 1
                warm_front_matrix = warm_front_matrix / num_years
                cold_front_matrix = cold_front_matrix / num_years
                wf_title_string = 'Annual number'
            else:
                wf_title_string = 'Number'

        wf_title_string += ' of warm fronts from {0:s} to {1:s}'.format(
            first_time_string, last_time_string
        )

        hours = front_count_dict[climo_utils.HOURS_KEY]
        if hours is not None:
            wf_title_string += '; hours {0:s}'.format(
                climo_utils.hours_to_string(hours)[0]
            )

        months = front_count_dict[climo_utils.MONTHS_KEY]
        if months is not None:
            wf_title_string += '; months {0:s}'.format(
                climo_utils.months_to_string(months)[0]
            )

        wf_output_file_name = '{0:s}/warm_front_{1:s}.jpg'.format(
            output_dir_name, 'frequency' if plot_frequency else 'count'
        )

        _plot_one_front_type(
            count_or_frequency_matrix=warm_front_matrix,
            use_latlng_grid=use_latlng_grid,
            colour_map_object=wf_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=wf_colour_max, plot_frequency=plot_frequency,
            title_string=wf_title_string, output_file_name=wf_output_file_name
        )

        cf_title_string = wf_title_string.replace('warm', 'cold')
        cf_output_file_name = '{0:s}/cold_front_{1:s}.jpg'.format(
            output_dir_name, 'frequency' if plot_frequency else 'count'
        )

        _plot_one_front_type(
            count_or_frequency_matrix=cold_front_matrix,
            use_latlng_grid=use_latlng_grid,
            colour_map_object=cf_colour_map_object,
            max_colour_percentile=max_colour_percentile,
            max_colour_value=cf_colour_max, plot_frequency=plot_frequency,
            title_string=cf_title_string, output_file_name=cf_output_file_name
        )

        return

    print('Reading data from: "{0:s}"...'.format(monte_carlo_file_name))
    monte_carlo_dict = climo_utils.read_monte_carlo_test(monte_carlo_file_name)
    property_name = monte_carlo_dict[climo_utils.PROPERTY_NAME_KEY]

    this_title_string = None
    this_output_file_name = None
    colour_map_object = None

    diff_colour_map_object = pyplot.get_cmap(diff_colour_map_name)

    if property_name == climo_utils.WF_FREQ_PROPERTY_NAME:
        this_title_string = 'WF frequency for baseline composite'
        this_output_file_name = '{0:s}/wf_frequency_baseline.jpg'.format(
            output_dir_name
        )

        colour_map_object = wf_colour_map_object

    elif property_name == climo_utils.CF_FREQ_PROPERTY_NAME:
        this_title_string = 'CF frequency for baseline composite'
        this_output_file_name = '{0:s}/cf_frequency_baseline.jpg'.format(
            output_dir_name
        )

        colour_map_object = cf_colour_map_object

    num_labels_matrix = monte_carlo_dict[climo_utils.NUM_LABELS_MATRIX_KEY]
    baseline_freq_matrix = monte_carlo_dict[climo_utils.BASELINE_MATRIX_KEY]
    baseline_freq_matrix[
        num_labels_matrix < MASK_IF_NUM_LABELS_BELOW
    ] = numpy.nan

    trial_freq_matrix = monte_carlo_dict[climo_utils.TRIAL_MATRIX_KEY]
    trial_freq_matrix[num_labels_matrix < MASK_IF_NUM_LABELS_BELOW] = numpy.nan

    significance_matrix = monte_carlo_dict[climo_utils.SIGNIFICANCE_MATRIX_KEY]
    significance_matrix[num_labels_matrix < MASK_IF_NUM_LABELS_BELOW] = False

    if max_colour_percentile is None:
        max_colour_value = monte_carlo_colour_maxima[0]
    else:
        concat_freq_matrix = numpy.concatenate(
            (baseline_freq_matrix, trial_freq_matrix), axis=0
        )
        max_colour_value = numpy.nanpercentile(
            concat_freq_matrix, max_colour_percentile
        )

    _plot_one_front_type(
        count_or_frequency_matrix=baseline_freq_matrix,
        use_latlng_grid=use_latlng_grid,
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        plot_frequency=True, title_string=this_title_string,
        output_file_name=this_output_file_name
    )

    this_title_string = this_title_string.replace('baseline', 'trial')
    this_output_file_name = this_output_file_name.replace(
        'baseline.jpg', 'trial.jpg'
    )

    _plot_one_front_type(
        count_or_frequency_matrix=trial_freq_matrix,
        use_latlng_grid=use_latlng_grid,
        colour_map_object=colour_map_object, max_colour_value=max_colour_value,
        plot_frequency=True, title_string=this_title_string,
        output_file_name=this_output_file_name
    )

    this_title_string = this_title_string.replace(' for trial composite', '')
    this_title_string = (
        'Composite difference (trial minus baseline) for {0:s}'
    ).format(this_title_string)

    this_output_file_name = this_output_file_name.replace(
        'trial.jpg', 'difference.jpg'
    )

    plot_gridded_stats._plot_monte_carlo_diff(
        difference_matrix=trial_freq_matrix - baseline_freq_matrix,
        significance_matrix=significance_matrix,
        colour_map_object=diff_colour_map_object,
        max_colour_percentile=max_colour_percentile,
        max_colour_value=monte_carlo_colour_maxima[1],
        title_string=this_title_string,
        output_file_name=this_output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_file_name=getattr(INPUT_ARG_OBJECT, COUNT_FILE_ARG_NAME),
        monte_carlo_file_name=getattr(
            INPUT_ARG_OBJECT, MONTE_CARLO_FILE_ARG_NAME
        ),
        use_latlng_grid=bool(getattr(
            INPUT_ARG_OBJECT, LATLNG_GRID_ARG_NAME
        )),
        wf_colour_map_name=getattr(INPUT_ARG_OBJECT, WF_COLOUR_MAP_ARG_NAME),
        cf_colour_map_name=getattr(INPUT_ARG_OBJECT, CF_COLOUR_MAP_ARG_NAME),
        diff_colour_map_name=getattr(
            INPUT_ARG_OBJECT, DIFFERENCE_CMAP_ARG_NAME
        ),
        plot_frequency=bool(getattr(INPUT_ARG_OBJECT, PLOT_FREQUENCY_ARG_NAME)),
        max_colour_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        wf_colour_max=getattr(INPUT_ARG_OBJECT, WF_COLOUR_MAX_ARG_NAME),
        cf_colour_max=getattr(INPUT_ARG_OBJECT, CF_COLOUR_MAX_ARG_NAME),
        monte_carlo_colour_maxima=numpy.array(
            getattr(INPUT_ARG_OBJECT, MC_COLOUR_MAXIMA_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
