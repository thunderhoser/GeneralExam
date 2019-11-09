"""Plotting methods for learning examples."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

METRES_PER_SECOND_TO_KT = 3.6 / 1.852

DEFAULT_BARB_LENGTH = 6
DEFAULT_EMPTY_BARB_RADIUS = 0.1
DEFAULT_WIND_BARB_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def plot_2d_grid(
        predictor_matrix, colour_map_object, colour_norm_object=None,
        min_colour_value=None, max_colour_value=None, axes_object=None,
        opacity=1.):
    """Plots predictor as 2-D colour map.

    M = number of rows in grid
    N = number of columns in grid

    :param predictor_matrix: M-by-N numpy array of predictor values.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.BoundaryNorm` or similar).
    :param min_colour_value: [used only if `colour_norm_object is None`]
        Minimum value in colour scheme.
    :param max_colour_value: [used only if `colour_norm_object is None`]
        Max value in colour scheme.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param opacity: Opacity for colour map (in range 0...1).
    :return: axes_object: See input doc.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    if colour_norm_object is None:
        error_checking.assert_is_greater(max_colour_value, min_colour_value)
        colour_norm_object = None
    else:
        if hasattr(colour_norm_object, 'boundaries'):
            min_colour_value = colour_norm_object.boundaries[0]
            max_colour_value = colour_norm_object.boundaries[-1]
        else:
            min_colour_value = colour_norm_object.vmin
            max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        predictor_matrix, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e12, alpha=opacity)

    axes_object.set_xticks([])
    axes_object.set_yticks([])
    return axes_object


def plot_wind_barbs(
        u_wind_matrix_m_s01, v_wind_matrix_m_s01, axes_object=None,
        plot_every=1, barb_colour=DEFAULT_WIND_BARB_COLOUR,
        barb_length=DEFAULT_BARB_LENGTH, fill_empty_barb=True,
        empty_barb_radius=DEFAULT_EMPTY_BARB_RADIUS):
    """Plots wind barbs in 2-D space.

    M = number of rows in grid
    N = number of columns in grid

    :param u_wind_matrix_m_s01: M-by-N numpy array of u-wind components (metres
        per second).
    :param v_wind_matrix_m_s01: Same but for v-wind.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If None, will create new
        axes.
    :param plot_every: Will plot wind barb every K grid cells, where
        K = `plot_every`.
    :param barb_colour: Wind-barb colour (length-3 numpy array of values in
        range 0...1).
    :param barb_length: Wind-barb length.
    :param fill_empty_barb: Boolean flag.  If True, empty wind barbs will be
        plotted as filled circles.  If False, they will be unfilled circles.
    :param empty_barb_radius: Radius of circle for empty wind barbs.
    :return: axes_object: See input doc.
    """

    error_checking.assert_is_real_numpy_array(u_wind_matrix_m_s01)
    error_checking.assert_is_real_numpy_array(v_wind_matrix_m_s01)
    error_checking.assert_is_numpy_array(u_wind_matrix_m_s01, num_dimensions=2)
    error_checking.assert_is_numpy_array(
        v_wind_matrix_m_s01,
        exact_dimensions=numpy.array(u_wind_matrix_m_s01.shape, dtype=int)
    )

    error_checking.assert_is_integer(plot_every)
    error_checking.assert_is_geq(plot_every, 1)
    error_checking.assert_is_numpy_array(
        barb_colour, exact_dimensions=numpy.array([3], dtype=int)
    )

    colour_map_object = matplotlib.colors.ListedColormap([barb_colour])
    colour_map_object.set_under(barb_colour)
    colour_map_object.set_over(barb_colour)

    num_grid_rows = u_wind_matrix_m_s01.shape[0]
    num_grid_columns = u_wind_matrix_m_s01.shape[1]
    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns - 1, dtype=float
    )
    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows - 1, dtype=float
    )

    x_coord_matrix, y_coord_matrix = grids.xy_vectors_to_matrices(
        x_unique_metres=x_coords_unique, y_unique_metres=y_coords_unique)

    # TODO(thunderhoser): Make sure things aren't upside-down.

    x_coords = numpy.ravel(x_coord_matrix[::plot_every, ::plot_every])
    y_coords = numpy.ravel(y_coord_matrix[::plot_every, ::plot_every])
    u_winds_m_s01 = numpy.ravel(u_wind_matrix_m_s01[::plot_every, ::plot_every])
    v_winds_m_s01 = numpy.ravel(v_wind_matrix_m_s01[::plot_every, ::plot_every])

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    wind_speeds_m_s01 = numpy.sqrt(u_winds_m_s01 ** 2 + v_winds_m_s01 ** 2)

    print(x_coords.shape)
    print(y_coords.shape)
    print(u_winds_m_s01.shape)
    print(v_winds_m_s01.shape)
    print(wind_speeds_m_s01.shape)

    axes_object.barbs(
        x_coords, y_coords,
        u_winds_m_s01 * METRES_PER_SECOND_TO_KT,
        v_winds_m_s01 * METRES_PER_SECOND_TO_KT,
        wind_speeds_m_s01 * METRES_PER_SECOND_TO_KT,
        sizes={'emptybarb': empty_barb_radius},
        length=barb_length, fill_empty=fill_empty_barb, rounding=False,
        cmap=colour_map_object, clim=numpy.array([-1, 0.]), linewidth=2
    )

    axes_object.set_xticks([])
    axes_object.set_yticks([])
    return axes_object
