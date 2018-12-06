"""Plotting methods for saliency maps."""

import numpy
from gewittergefahr.gg_utils import error_checking

DEFAULT_LINE_WIDTH = 2


def plot_2d_grid(saliency_matrix_2d, axes_object, colour_map_object,
                 max_absolute_contour_level, contour_interval,
                 line_width=DEFAULT_LINE_WIDTH):
    """Plots 2-D saliency map (for one field at one time) with line contours.

    M = number of rows in grid
    N = number of columns in grid

    Positive saliency values will be plotted with solid contour lines, and
    negative values will be plotted with dashed lines.

    :param saliency_matrix_2d: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.  This colour
        map will be applied to absolute values, rather than signed values.  In
        other words, this colour map will be duplicated and flipped, to create a
        diverging colour map.  Thus, `colour_map_object` itself should be a
        sequential colour map, not a diverging one.  However, this is not
        enforced by the code, so do whatever you want.
    :param max_absolute_contour_level: Max absolute saliency value to plot.  The
        min and max values, respectively, will be
        `-1 * max_absolute_contour_level` and `max_absolute_contour_level`.
    :param contour_interval: Saliency interval between successive contours.
    :param line_width: Width of contour lines.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_2d)
    error_checking.assert_is_numpy_array(saliency_matrix_2d, num_dimensions=2)

    error_checking.assert_is_greater(contour_interval, 0.)
    error_checking.assert_is_greater(
        max_absolute_contour_level, contour_interval)

    num_grid_rows = saliency_matrix_2d.shape[0]
    num_grid_columns = saliency_matrix_2d.shape[1]
    y_coord_vector = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float)
    x_coord_vector = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float)

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(
        x_coord_vector, y_coord_vector)
    x_coord_matrix += 0.5
    y_coord_matrix += 0.5

    half_num_contours = int(numpy.round(
        1 + max_absolute_contour_level / contour_interval
    ))

    # Plot positive values.
    these_contour_levels = numpy.linspace(
        0., max_absolute_contour_level, num=half_num_contours)

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix_2d,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='solid')

    # Plot negative values.
    these_contour_levels = these_contour_levels[1:]

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -saliency_matrix_2d,
        these_contour_levels, cmap=colour_map_object,
        vmin=numpy.min(these_contour_levels),
        vmax=numpy.max(these_contour_levels), linewidths=line_width,
        linestyles='dashed')


def plot_many_2d_grids(
        saliency_matrix_3d, axes_objects_2d_list, colour_map_object,
        max_absolute_contour_level, contour_interval,
        line_width=DEFAULT_LINE_WIDTH):
    """Plots many 2-D saliency maps (many fields at the same time).

    The saliency map for each field will be one panel in a paneled figure.

    M = number of spatial rows
    N = number of spatial columns
    C = number of channels (predictors)

    :param saliency_matrix_3d: M-by-N-by-P numpy array of saliency values.
    :param axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
        [i]th row and [j]th column.
    :param colour_map_object: See doc for `plot_2d_grid`.
    :param max_absolute_contour_level: Same.
    :param contour_interval: Same.
    :param line_width: Width of contour lines.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix_3d)
    error_checking.assert_is_numpy_array(saliency_matrix_3d, num_dimensions=3)
    num_predictors = saliency_matrix_3d.shape[-1]

    num_panel_rows = len(axes_objects_2d_list)
    num_panel_columns = len(axes_objects_2d_list[0])

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns)
        )

        plot_2d_grid(
            saliency_matrix_2d=saliency_matrix_3d[..., k],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=colour_map_object,
            max_absolute_contour_level=max_absolute_contour_level,
            contour_interval=contour_interval, line_width=line_width)
