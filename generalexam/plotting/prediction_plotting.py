"""Plotting methods for model predictions."""

import numpy
import matplotlib.colors
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import nwp_plotting
from generalexam.ge_utils import front_utils

DEFAULT_GRID_OPACITY = 0.5

ANY_FRONT_STRING = 'any'
VALID_STRING_IDS = [
    ANY_FRONT_STRING, front_utils.WARM_FRONT_STRING,
    front_utils.COLD_FRONT_STRING
]


def _check_front_type(front_string_id):
    """Ensures that front type is valid.

    :param front_string_id: String ID for front type.
    :raises: ValueError: if front type is unrecognized.
    """

    error_checking.assert_is_string(front_string_id)

    if front_string_id not in VALID_STRING_IDS:
        error_string = (
            '\n{0:s}\nValid front types (listed above) do not include "{1:s}".'
        ).format(VALID_STRING_IDS, front_string_id)

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


def get_cold_front_colour_map():
    """Returns colour map for cold-front probability.

    :return: colour_map_object: See documentation for
        `get_any_front_colour_map`.
    :return: colour_norm_object: Same.
    :return: colour_bounds: Same.
    """

    main_colour_list = [
        numpy.array([247., 251., 255.]), numpy.array([222., 235., 247.]),
        numpy.array([198., 219., 239.]), numpy.array([158., 202., 225.]),
        numpy.array([133., 189., 220.]), numpy.array([107., 174., 214.]),
        numpy.array([66., 146., 198.]), numpy.array([33., 113., 181.]),
        numpy.array([8., 81., 156.]), numpy.array([8., 48., 107.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.linspace(0.1, 0.9, num=9)
    main_colour_bounds = numpy.concatenate((
        numpy.array([0.01]), main_colour_bounds, numpy.array([1.])))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([2.])))
    return colour_map_object, colour_norm_object, colour_bounds


def get_warm_front_colour_map():
    """Returns colour map for warm-front probability.

    :return: colour_map_object: See documentation for
        `get_any_front_colour_map`.
    :return: colour_norm_object: Same.
    :return: colour_bounds: Same.
    """

    main_colour_list = [
        numpy.array([255., 245., 240.]), numpy.array([254., 224., 210.]),
        numpy.array([252., 187., 161.]), numpy.array([252., 146., 114.]),
        numpy.array([252., 126., 93.]), numpy.array([251., 106., 74.]),
        numpy.array([239., 59., 44.]), numpy.array([203., 24., 29.]),
        numpy.array([165., 15., 21.]), numpy.array([103., 0., 13.])]

    for i in range(len(main_colour_list)):
        main_colour_list[i] /= 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.array([1., 1., 1.]))
    colour_map_object.set_over(numpy.array([1., 1., 1.]))

    main_colour_bounds = numpy.linspace(0.1, 0.9, num=9)
    main_colour_bounds = numpy.concatenate((
        numpy.array([0.01]), main_colour_bounds, numpy.array([1.])))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        main_colour_bounds, colour_map_object.N)

    colour_bounds = numpy.concatenate((
        numpy.array([0.]), main_colour_bounds, numpy.array([2.])))
    return colour_map_object, colour_norm_object, colour_bounds


def plot_gridded_probs(
        probability_matrix, front_string_id, axes_object, basemap_object,
        full_grid_name, first_row_in_full_grid=0, first_column_in_full_grid=0,
        opacity=DEFAULT_GRID_OPACITY):
    """Plots gridded front probabilities.

    M = number of rows in grid
    N = number of columns in grid

    :param probability_matrix: M-by-N numpy array of predicted front
        probabilities.
    :param front_string_id: Type of fronts predicted in `probability_matrix`.
        Must be accepted by `_check_front_type`.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param basemap_object: Will be used to convert between lat-long and x-y
        (projection) coordinates (instance of `mpl_toolkits.basemap.Basemap`).
    :param full_grid_name: Name of full grid (must be accepted by
        `nwp_model_utils.check_grid_name`).
    :param first_row_in_full_grid: First row in full grid.  In other words,
        row 0 in `probability_matrix` = row `first_row_in_full_grid` in the
        full grid.
    :param first_column_in_full_grid: Same but for column.
    :param opacity: Opacity for colour map (in range 0...1).
    """

    error_checking.assert_is_numpy_array(probability_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(
        probability_matrix, 0., allow_nan=False)
    error_checking.assert_is_leq_numpy_array(
        probability_matrix, 1., allow_nan=False)

    _check_front_type(front_string_id)

    if front_string_id == ANY_FRONT_STRING:
        colour_map_object, _, colour_bounds = get_any_front_colour_map()
    elif front_string_id == front_utils.WARM_FRONT_STRING:
        colour_map_object, _, colour_bounds = get_warm_front_colour_map()
    else:
        colour_map_object, _, colour_bounds = get_cold_front_colour_map()

    nwp_plotting.plot_subgrid(
        field_matrix=probability_matrix,
        model_name=nwp_model_utils.NARR_MODEL_NAME, axes_object=axes_object,
        basemap_object=basemap_object, colour_map_object=colour_map_object,
        min_colour_value=colour_bounds[1],
        max_colour_value=colour_bounds[-2], grid_id=full_grid_name,
        first_row_in_full_grid=first_row_in_full_grid,
        first_column_in_full_grid=first_column_in_full_grid, opacity=opacity)


def plot_gridded_counts(
        count_or_frequency_matrix, axes_object, basemap_object,
        colour_map_object, full_grid_name, colour_norm_object=None,
        first_row_in_full_grid=0, first_column_in_full_grid=0):
    """Plots gridded front counts.

    M = number of rows in grid
    N = number of columns in grid

    :param count_or_frequency_matrix: M-by-N numpy array with raw counts or
        frequencies.
    :param axes_object: See doc for `plot_gridded_probs`.
    :param basemap_object: Same.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param full_grid_name: See doc for `plot_gridded_probs`.
    :param colour_norm_object: Colour-normalizer (instance of
        `matplotlib.colors.Normalize`).  Used to convert from time to colour.
    :param first_row_in_full_grid: See doc for `plot_gridded_probs`.
    :param first_column_in_full_grid: Same.
    """

    error_checking.assert_is_numpy_array(
        count_or_frequency_matrix, num_dimensions=2)
    error_checking.assert_is_geq_numpy_array(count_or_frequency_matrix, 0.)

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    nwp_plotting.plot_subgrid(
        field_matrix=count_or_frequency_matrix,
        model_name=nwp_model_utils.NARR_MODEL_NAME, grid_id=full_grid_name,
        axes_object=axes_object, basemap_object=basemap_object,
        colour_map_object=colour_map_object,
        min_colour_value=min_colour_value, max_colour_value=max_colour_value,
        first_row_in_full_grid=first_row_in_full_grid,
        first_column_in_full_grid=first_column_in_full_grid, opacity=1.)
