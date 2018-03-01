"""Plotting methods for model predictions."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils
from generalexam.plotting import narr_plotting

DEFAULT_GRID_OPACITY = 0.5

ANY_FRONT_STRING_ID = 'any'
VALID_STRING_IDS = [
    ANY_FRONT_STRING_ID, front_utils.WARM_FRONT_STRING_ID,
    front_utils.COLD_FRONT_STRING_ID]


def _check_front_type(front_string_id):
    """Ensures that front type is valid.

    :param front_string_id: String ID for front type.
    :raises: ValueError: if front type is unrecognized.
    """

    error_checking.assert_is_string(front_string_id)
    if front_string_id not in VALID_STRING_IDS:
        error_string = (
            '\n\n{0:s}\nValid front types (listed above) do not include '
            '"{1:s}".').format(VALID_STRING_IDS, front_string_id)
        raise ValueError(error_string)


def get_any_front_colour_map():
    """Returns colour map for "probability of any front" (warm or cold).

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
        probability_matrix, front_string_id, axes_object, basemap_object,
        first_row_in_narr_grid=0, first_column_in_narr_grid=0,
        opacity=DEFAULT_GRID_OPACITY):
    """Plots frontal-probability map on NARR grid.

    This method plots data over a contiguous subset of the NARR grid, which need
    not be *strictly* a subset.  In other words, the "subset" could be the full
    NARR grid.

    M = number of rows (unique grid-point y-coordinates)
    N = number of columns (unique grid-point x-coordinates)

    :param probability_matrix: M-by-N numpy array, where
        predicted_target_matrix[i, j] is the predicted probability of a front
        passing through grid cell [i, j].
    :param front_string_id: Type of fronts predicted in `probability_matrix`.
        May be "warm", "cold", or "any".
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param first_row_in_narr_grid: Row 0 in the subgrid is row
        `first_row_in_narr_grid` in the full NARR grid.
    :param first_column_in_narr_grid: Column 0 in the subgrid is row
        `first_column_in_narr_grid` in the full NARR grid.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_numpy_array(probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(
        probability_matrix, 0., allow_nan=False)
    error_checking.assert_is_leq_numpy_array(
        probability_matrix, 1., allow_nan=False)

    _check_front_type(front_string_id)

    if front_string_id == ANY_FRONT_STRING_ID:
        colour_map_object, _, colour_bounds = get_any_front_colour_map()
        colour_minimum = colour_bounds[1]
        colour_maximum = colour_bounds[-2]
    elif front_string_id == front_utils.WARM_FRONT_STRING_ID:
        colour_map_object = pyplot.cm.Reds
        colour_minimum = 0.
        colour_maximum = 1.
    else:
        colour_map_object = pyplot.cm.Blues
        colour_minimum = 0.
        colour_maximum = 1.

    narr_plotting.plot_xy_grid(
        data_matrix=probability_matrix, axes_object=axes_object,
        basemap_object=basemap_object, colour_map=colour_map_object,
        colour_minimum=colour_minimum, colour_maximum=colour_maximum,
        first_row_in_narr_grid=first_row_in_narr_grid,
        first_column_in_narr_grid=first_column_in_narr_grid, opacity=opacity)
