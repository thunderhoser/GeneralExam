"""Plotting methods for learning examples."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from generalexam.ge_io import processed_narr_io

WIND_NAMES = [
    processed_narr_io.U_WIND_GRID_RELATIVE_NAME,
    processed_narr_io.V_WIND_GRID_RELATIVE_NAME
]

DEFAULT_WIND_SPEED_SCALING_FACTOR = 0.2
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15


def _create_paneled_figure(
        num_panel_rows, num_panel_columns, horizontal_space_fraction=0.1,
        vertical_space_fraction=0.1):
    """Creates paneled figure.

    :param num_panel_rows: Number of rows.
    :param num_panel_columns: Number of columns.
    :param horizontal_space_fraction: Horizontal space between adjacent panels
        (as fraction of panel size).
    :param vertical_space_fraction: Vertical space between adjacent panels
        (as fraction of panel size).
    :return: figure_object: Instance of `matplotlib.figure.Figure`.
    :return: axes_objects_2d_list: 2-D list, where axes_objects_2d_list[i][j] is
        the handle (instance of `matplotlib.axes._subplots.AxesSubplot`) for the
        [i]th row and [j]th column.
    """

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_geq(num_panel_rows, 1)
    error_checking.assert_is_integer(num_panel_columns)
    error_checking.assert_is_geq(num_panel_columns, 1)
    error_checking.assert_is_geq(horizontal_space_fraction, 0.)
    error_checking.assert_is_less_than(horizontal_space_fraction, 1.)
    error_checking.assert_is_geq(vertical_space_fraction, 0.)
    error_checking.assert_is_less_than(vertical_space_fraction, 1.)

    figure_object, axes_objects_2d_list = pyplot.subplots(
        num_panel_rows, num_panel_columns, sharex=True, sharey=True,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if num_panel_rows == num_panel_columns == 1:
        axes_objects_2d_list = [[axes_objects_2d_list]]
    elif num_panel_columns == 1:
        axes_objects_2d_list = [[a] for a in axes_objects_2d_list]
    elif num_panel_rows == 1:
        axes_objects_2d_list = [axes_objects_2d_list]

    pyplot.subplots_adjust(
        left=0.02, bottom=0.02, right=0.98, top=0.95,
        hspace=vertical_space_fraction, wspace=horizontal_space_fraction)

    return figure_object, axes_objects_2d_list


def plot_predictor_2d(
        predictor_matrix, colour_map_object, colour_norm_object=None,
        min_colour_value=None, max_colour_value=None, axes_object=None):
    """Plots predictor variable on 2-D grid.

    If `colour_norm_object is None`, both `min_colour_value` and
    `max_colour_value` must be specified.

    M = number of rows in grid
    N = number of columns in grid

    :param predictor_matrix: M-by-N numpy array of predictor values.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
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
        edgecolors='None')

    axes_object.set_xticks([])
    axes_object.set_yticks([])

    return plotting_utils.add_colour_bar(
        axes_object_or_list=axes_object, values_to_colour=predictor_matrix,
        colour_map=colour_map_object, colour_norm_object=colour_norm_object,
        orientation='vertical')


def plot_wind_2d(u_wind_matrix, v_wind_matrix, axes_object=None,
                 scaling_factor=DEFAULT_WIND_SPEED_SCALING_FACTOR):
    """Plots wind velocity on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param u_wind_matrix: M-by-N numpy array of eastward components.
    :param v_wind_matrix: M-by-N numpy array of northward components.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param scaling_factor: Scaling factor for wind speed (necessary because
        input matrices are probably in z-score units).  A speed of
        `wind_speed_scaling_factor` will correspond to a long line (which is
        usually 10 kt).
    """

    if axes_object is None:
        _, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    num_grid_rows = u_wind_matrix.shape[0]
    num_grid_columns = u_wind_matrix.shape[1]

    x_coords_unique = numpy.linspace(
        0, num_grid_columns, num=num_grid_columns + 1, dtype=float)
    x_coords_unique = x_coords_unique[:-1]
    x_coords_unique = x_coords_unique + numpy.diff(x_coords_unique[:2]) / 2

    y_coords_unique = numpy.linspace(
        0, num_grid_rows, num=num_grid_rows + 1, dtype=float)
    y_coords_unique = y_coords_unique[:-1]
    y_coords_unique = y_coords_unique + numpy.diff(y_coords_unique[:2]) / 2

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords_unique,
                                                    y_coords_unique)

    speed_matrix_m_s01 = numpy.sqrt(u_wind_matrix ** 2 + v_wind_matrix ** 2)
    multiplier = 10. / scaling_factor

    axes_object.barbs(
        x_coord_matrix, y_coord_matrix,
        u_wind_matrix * multiplier, v_wind_matrix * multiplier,
        speed_matrix_m_s01 * multiplier, color='k', length=6,
        sizes={'emptybarb': 0.1}, fill_empty=True, rounding=False)

    axes_object.set_xlim(0, num_grid_columns)
    axes_object.set_ylim(0, num_grid_rows)


def _check_input_args_many_predictors(
        predictor_matrix, predictor_names, cmap_object_by_predictor,
        cnorm_object_by_predictor, min_colour_value_by_predictor,
        max_colour_value_by_predictor, plot_wind_barbs):
    """Error-checks input arguments for `plot_many_predictors*`.

    :param predictor_matrix: See doc for `plot_many_predictors_sans_barbs` or
        `plot_many_predictors_with_barbs`.
    :param predictor_names: Same.
    :param cmap_object_by_predictor: Same.
    :param cnorm_object_by_predictor: Same.
    :param min_colour_value_by_predictor: Same.
    :param max_colour_value_by_predictor: Same.
    :param plot_wind_barbs: Boolean flag.  If True, wind velocity will be
        plotted with barbs.  If False, wind velocity will be plotted with two
        colour maps (one for u-component, one for v-component).
    :return: cnorm_object_by_predictor: See input documentation.
    :raises: TypeError: if `cmap_object_by_predictor` or
        `cnorm_object_by_predictor` is not a list.
    :raises: ValueError: if `cmap_object_by_predictor` or
        `cnorm_object_by_predictor` has a different length than
        `predictor_names`.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=3)
    num_predictors = predictor_matrix.shape[-1]

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names),
        exact_dimensions=numpy.array([num_predictors])
    )

    error_checking.assert_is_boolean(plot_wind_barbs)
    if plot_wind_barbs:
        u_wind_index, v_wind_index = get_wind_indices(predictor_names)
    else:
        u_wind_index = -1
        v_wind_index = -1

    if not isinstance(cmap_object_by_predictor, list):
        error_string = (
            'cmap_object_by_predictor should be a list.  Instead, got type '
            '"{0:s}".'
        ).format(type(cmap_object_by_predictor).__name__)

        raise TypeError(error_string)

    if len(cmap_object_by_predictor) != num_predictors:
        error_string = (
            'Length of cmap_object_by_predictor ({0:d}) should equal number of '
            'predictors ({1:d}).'
        ).format(len(cmap_object_by_predictor), num_predictors)

        raise ValueError(error_string)

    if cnorm_object_by_predictor is None:
        error_checking.assert_is_numpy_array(
            min_colour_value_by_predictor,
            exact_dimensions=numpy.array([num_predictors])
        )

        error_checking.assert_is_numpy_array(
            max_colour_value_by_predictor,
            exact_dimensions=numpy.array([num_predictors])
        )

        cnorm_object_by_predictor = []

        for k in range(num_predictors):
            if k in [u_wind_index, v_wind_index]:
                cnorm_object_by_predictor.append(None)
                continue

            error_checking.assert_is_greater(max_colour_value_by_predictor[k],
                                             min_colour_value_by_predictor[k])

            this_colour_norm_object = matplotlib.colors.Normalize(
                vmin=min_colour_value_by_predictor[k],
                vmax=max_colour_value_by_predictor[k], clip=False)
            cnorm_object_by_predictor.append(this_colour_norm_object)

    if not isinstance(cnorm_object_by_predictor, list):
        error_string = (
            'cnorm_object_by_predictor should be a list.  Instead, got type'
            ' "{0:s}".'
        ).format(type(cnorm_object_by_predictor).__name__)

        raise TypeError(error_string)

    if len(cnorm_object_by_predictor) != num_predictors:
        error_string = (
            'Length of cnorm_object_by_predictor ({0:d}) should equal '
            'number of predictors ({1:d}).'
        ).format(len(cnorm_object_by_predictor), num_predictors)

        raise ValueError(error_string)

    return cnorm_object_by_predictor


def get_wind_indices(predictor_names):
    """Returns array indices of u-wind and v-wind.

    :param predictor_names: 1-D list of predictor names (must be accepted by
        `processed_narr_io.check_field_name`).
    :return: u_wind_index: Array index of u-wind.
    :return: v_wind_index: Array index of v-wind.
    :raises: ValueError: if either u-wind or v-wind cannot be found.
    """

    try:
        u_wind_index = predictor_names.index(
            processed_narr_io.U_WIND_GRID_RELATIVE_NAME)
    except ValueError:
        error_string = (
            '\n{0:s}\nPredictor names (shown above) must include "{1:s}".'
        ).format(str(predictor_names),
                 processed_narr_io.U_WIND_GRID_RELATIVE_NAME)

        raise ValueError(error_string)

    try:
        v_wind_index = predictor_names.index(
            processed_narr_io.V_WIND_GRID_RELATIVE_NAME)
    except ValueError:
        error_string = (
            '\n{0:s}\nPredictor names (shown above) must include "{1:s}".'
        ).format(str(predictor_names),
                 processed_narr_io.V_WIND_GRID_RELATIVE_NAME)

        raise ValueError(error_string)

    return u_wind_index, v_wind_index


def plot_many_predictors_sans_barbs(
        predictor_matrix, predictor_names, cmap_object_by_predictor,
        cnorm_object_by_predictor=None, min_colour_value_by_predictor=None,
        max_colour_value_by_predictor=None):
    """Plots many predictor variables on 2-D grid; no wind barbs overlain.

    In this case, both u-wind and v-wind are plotted as separate maps.

    M = number of rows in grid
    N = number of columns in grid
    C = number of predictors

    :param predictor_matrix: M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param cmap_object_by_predictor: length-C list of colour maps (instances of
        `matplotlib.pyplot.cm`).
    :param cnorm_object_by_predictor: length-C list of colour-normalization
        schemes (instances of `matplotlib.colors.BoundaryNorm`).
    :param min_colour_value_by_predictor:
        [used only if `cnorm_object_by_predictor is None`]
        length-C numpy array with minimum value in each colour scheme.
    :param max_colour_value_by_predictor:
        [used only if `cnorm_object_by_predictor is None`]
        length-C numpy array with max value in each colour scheme.
    :return: figure_object: See doc for `_create_paneled_figure`.
    :return: axes_objects_2d_list: Same.
    """

    cnorm_object_by_predictor = _check_input_args_many_predictors(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        cmap_object_by_predictor=cmap_object_by_predictor,
        cnorm_object_by_predictor=cnorm_object_by_predictor,
        min_colour_value_by_predictor=min_colour_value_by_predictor,
        max_colour_value_by_predictor=max_colour_value_by_predictor,
        plot_wind_barbs=False)

    num_predictors = len(predictor_names)
    num_panel_rows = int(numpy.floor(numpy.sqrt(num_predictors)))
    num_panel_columns = int(numpy.ceil(float(num_predictors) / num_panel_rows))

    figure_object, axes_objects_2d_list = _create_paneled_figure(
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns)

    for i in range(num_panel_rows):
        for j in range(num_panel_columns):
            this_linear_index = i * num_panel_columns + j
            if this_linear_index >= num_predictors:
                break

            this_colour_bar_object = plot_predictor_2d(
                predictor_matrix=predictor_matrix[..., this_linear_index],
                colour_map_object=cmap_object_by_predictor[this_linear_index],
                colour_norm_object=cnorm_object_by_predictor[this_linear_index],
                axes_object=axes_objects_2d_list[i][j])

            this_colour_bar_object.set_label(predictor_names[this_linear_index])

    return figure_object, axes_objects_2d_list


def plot_many_predictors_with_barbs(
        predictor_matrix, predictor_names, cmap_object_by_predictor,
        wind_speed_scaling_factor=DEFAULT_WIND_SPEED_SCALING_FACTOR,
        cnorm_object_by_predictor=None, min_colour_value_by_predictor=None,
        max_colour_value_by_predictor=None):
    """Plots many predictor variables on 2-D grid with wind barbs overlain.

    :param predictor_matrix: See doc for `plot_many_predictors_sans_barbs`.
    :param predictor_names: Same.
    :param cmap_object_by_predictor: Same as `plot_many_predictors_sans_barbs`,
        except that entries for u-wind and v-wind are ignored.
    :param wind_speed_scaling_factor: See doc for `plot_wind_2d`.
    :param cnorm_object_by_predictor: See doc for `cmap_object_by_predictor`.
    :param min_colour_value_by_predictor: Same.
    :param max_colour_value_by_predictor: Same.
    :return: figure_object: See doc for `_create_paneled_figure`.
    :return: axes_objects_2d_list: Same.
    """

    cnorm_object_by_predictor = _check_input_args_many_predictors(
        predictor_matrix=predictor_matrix, predictor_names=predictor_names,
        cmap_object_by_predictor=cmap_object_by_predictor,
        cnorm_object_by_predictor=cnorm_object_by_predictor,
        min_colour_value_by_predictor=min_colour_value_by_predictor,
        max_colour_value_by_predictor=max_colour_value_by_predictor,
        plot_wind_barbs=True)

    u_wind_index, v_wind_index = get_wind_indices(predictor_names)
    u_wind_matrix = predictor_matrix[..., u_wind_index]
    v_wind_matrix = predictor_matrix[..., v_wind_index]

    non_wind_predictor_names = [
        p for p in predictor_names if p not in WIND_NAMES
    ]

    figure_object, axes_objects_2d_list = _create_paneled_figure(
        num_panel_rows=len(non_wind_predictor_names), num_panel_columns=1)

    for k in range(len(non_wind_predictor_names)):
        this_index = predictor_names.index(non_wind_predictor_names[k])

        this_colour_bar_object = plot_predictor_2d(
            predictor_matrix=predictor_matrix[..., this_index],
            colour_map_object=cmap_object_by_predictor[this_index],
            colour_norm_object=cnorm_object_by_predictor[this_index],
            axes_object=axes_objects_2d_list[k][0])

        this_colour_bar_object.set_label(predictor_names[this_index])

        plot_wind_2d(u_wind_matrix=u_wind_matrix, v_wind_matrix=v_wind_matrix,
                     axes_object=axes_objects_2d_list[k][0],
                     scaling_factor=wind_speed_scaling_factor)

    return figure_object, axes_objects_2d_list
