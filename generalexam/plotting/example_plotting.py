"""Plotting methods for learning examples."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot
import matplotlib.colors
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils

TITLE_FONT_SIZE = 20
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


def plot_2d_grid(
        predictor_matrix_2d, axes_object, colour_map_object,
        colour_norm_object=None, min_colour_value=None, max_colour_value=None,
        annotation_string=None):
    """Plots 2-D grid (one field at one time) as colour map.

    M = number of rows in grid
    N = number of columns in grid

    :param predictor_matrix_2d: M-by-N numpy array of values to plot (either
        normalized or unnormalized -- this method doesn't care).
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param colour_map_object: Instance of `matplotlib.pyplot.cm`.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`.
    :param min_colour_value: [used only if `colour_norm_object is None`]
        Minimum value in colour scheme.
    :param max_colour_value: [used only if `colour_norm_object is None`]
        Max value in colour scheme.
    :param annotation_string: Annotation (printed in the bottom-center of the
        map).  For no annotation, leave this alone.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix_2d)
    error_checking.assert_is_numpy_array(predictor_matrix_2d, num_dimensions=2)

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
        predictor_matrix_2d, cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None')

    if annotation_string is not None:
        error_checking.assert_is_string(annotation_string)
        axes_object.text(
            0.5, 0.01, annotation_string, fontsize=20, color='k',
            horizontalalignment='center', verticalalignment='bottom',
            transform=axes_object.transAxes)

    axes_object.set_xticks([])
    axes_object.set_yticks([])


def plot_many_2d_grids(
        predictor_matrix_3d, predictor_names, num_panel_rows,
        cmap_object_by_predictor, cnorm_object_by_predictor=None,
        min_colour_value_by_predictor=None, max_colour_value_by_predictor=None):
    """Plots many 2-D grids (many fields at the same time).

    Each field will be one panel in a paneled figure.

    M = number of spatial rows
    N = number of spatial columns
    C = number of channels (predictors)

    :param predictor_matrix_3d: M-by-N-by-P numpy array of values to plot (may
        be normalized or unnormalized).
    :param predictor_names: length-P list of predictor names (will be used as
        panel titles).
    :param num_panel_rows: Number of rows in paneled figure.
    :param cmap_object_by_predictor: length-P list of colour maps (instances of
        `matplotlib.pyplot.cm`).
    :param cnorm_object_by_predictor: length-P list of colour-normalization
        schemes (instances of `matplotlib.colors.BoundaryNorm`).
    :param min_colour_value_by_predictor:
        [used only if `cnorm_object_by_predictor is None`]
        length-P numpy array with minimum value in each colour scheme.
    :param max_colour_value_by_predictor:
        [used only if `cnorm_object_by_predictor is None`]
        length-P numpy array with max value in each colour scheme.
    :return: figure_object: See doc for `_create_paneled_figure`.
    :return: axes_objects_2d_list: Same.
    :raises: TypeError: if `cmap_object_by_predictor` is not a list;
        if `cnorm_object_by_predictor` is neither None nor list.
    :raises: ValueError: if `cmap_object_by_predictor` or
        `cnorm_object_by_predictor` has a different length than the last axis of
        `predictor_matrix_3d`.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix_3d)
    error_checking.assert_is_numpy_array(predictor_matrix_3d, num_dimensions=3)
    num_predictors = predictor_matrix_3d.shape[-1]

    error_checking.assert_is_string_list(predictor_names)
    error_checking.assert_is_numpy_array(
        numpy.array(predictor_names),
        exact_dimensions=numpy.array([num_predictors])
    )

    error_checking.assert_is_integer(num_panel_rows)
    error_checking.assert_is_greater(num_panel_rows, 0)

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
            error_checking.assert_is_greater(max_colour_value_by_predictor[k],
                                             min_colour_value_by_predictor[k])

            cnorm_object_by_predictor.append(
                matplotlib.colors.Normalize(
                    vmin=min_colour_value_by_predictor[k],
                    vmax=max_colour_value_by_predictor[k], clip=False)
            )

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

    num_panel_columns = int(numpy.ceil(float(num_predictors) / num_panel_rows))
    figure_object, axes_objects_2d_list = _create_paneled_figure(
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        horizontal_space_fraction=0.05)

    for k in range(num_predictors):
        this_panel_row, this_panel_column = numpy.unravel_index(
            k, (num_panel_rows, num_panel_columns)
        )

        plot_2d_grid(
            predictor_matrix_2d=predictor_matrix_3d[..., k],
            axes_object=axes_objects_2d_list[this_panel_row][this_panel_column],
            colour_map_object=cmap_object_by_predictor[k],
            colour_norm_object=cnorm_object_by_predictor[k])

        axes_objects_2d_list[this_panel_row][this_panel_column].set_title(
            predictor_names[k], fontsize=TITLE_FONT_SIZE)

        plotting_utils.add_colour_bar(
            axes_object_or_list=axes_objects_2d_list[
                this_panel_row][this_panel_column],
            values_to_colour=predictor_matrix_3d[..., k],
            colour_map=cmap_object_by_predictor[k],
            colour_norm_object=cnorm_object_by_predictor[k],
            orientation='vertical')

    return figure_object, axes_objects_2d_list
