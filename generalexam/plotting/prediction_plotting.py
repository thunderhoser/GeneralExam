"""Plotting methods for model predictions."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from gewittergefahr.gg_utils import error_checking
from generalexam.plotting import narr_plotting

DEFAULT_GRID_OPACITY = 0.5


def _get_default_colour_map():
    """Returns default colour map for probability.

    N = number of colours

    :return: colour_map_object: Instance of `matplotlib.colors.ListedColormap`.
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :return: colour_bounds: length-(N + 1) numpy array of colour boundaries.
        colour_bounds[0] and colour_bounds[1] are the boundaries for the 1st
        colour; colour_bounds[1] and colour_bounds[2] are the boundaries for the
        2nd colour; ...; colour_bounds[i] and colour_bounds[i + 1] are the
        boundaries for the (i + 1)th colour.
    """

    main_colour_list = [
        numpy.array([0., 90., 50.]), numpy.array([35., 139., 69.]),
        numpy.array([65., 171., 93.]), numpy.array([116., 196., 118.]),
        numpy.array([161., 217., 155.]), numpy.array([8., 69., 148.]),
        numpy.array([33., 113., 181.]), numpy.array([66., 146., 198.]),
        numpy.array([107., 174., 214.]), numpy.array([158., 202., 225.]),
        numpy.array([74., 20., 134.]), numpy.array([106., 81., 163.]),
        numpy.array([128., 125., 186.]), numpy.array([158., 154., 200.]),
        numpy.array([188., 189., 220.]), numpy.array([153., 0., 13.]),
        numpy.array([203., 24., 29.]), numpy.array([239., 59., 44.]),
        numpy.array([251., 106., 74.]), numpy.array([252., 146., 114.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.linspace(0.05, 0.95, num=19)
    main_colour_bounds = numpy.concatenate((
        numpy.array([0.01]), main_colour_bounds, numpy.array([1.])))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([2.])))
    return colour_map_object, colour_norm_object, colour_bounds


def plot_narr_grid(
        predicted_target_matrix, axes_object, basemap_object,
        first_row_in_narr_grid=0, first_column_in_narr_grid=0,
        opacity=DEFAULT_GRID_OPACITY):
    """Plots colour map of predicted probabilities.

    This method plots data over a contiguous subset of the NARR grid, which need
    not be *strictly* a subset.  In other words, the "subset" could be the full
    NARR grid.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param predicted_target_matrix: M-by-N numpy array, where
        predicted_target_matrix[i, j] is the predicted probability of a front
        passing through grid cell [i, j].
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param first_row_in_narr_grid: Row 0 in the subgrid is row
        `first_row_in_narr_grid` in the full NARR grid.
    :param first_column_in_narr_grid: Column 0 in the subgrid is row
        `first_column_in_narr_grid` in the full NARR grid.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_numpy_array(
        predicted_target_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(
        predicted_target_matrix, 0., allow_nan=False)
    error_checking.assert_is_leq_numpy_array(
        predicted_target_matrix, 1., allow_nan=False)

    colour_map_object, _, colour_bounds = _get_default_colour_map()

    narr_plotting.plot_xy_grid(
        data_matrix=predicted_target_matrix, axes_object=axes_object,
        basemap_object=basemap_object, colour_map=colour_map_object,
        colour_minimum=colour_bounds[1], colour_maximum=colour_bounds[-2],
        first_row_in_narr_grid=first_row_in_narr_grid,
        first_column_in_narr_grid=first_column_in_narr_grid, opacity=opacity)
