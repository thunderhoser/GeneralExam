"""Plots 1-D version of TFP (thermal front parameter) schematic."""

import numpy
import matplotlib.pyplot as pyplot

NUM_GRID_POINTS = 1001
FIRST_POINT_IN_FRONT = 400
LAST_POINT_IN_FRONT = 600
MIN_TEMPERATURE_KELVINS = 0.
DEFAULT_GRADIENT_KELVINS_PT01 = 0.1
FRONT_GRADIENT_KELVINS_PT01 = 1.

TEMPERATURE_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
FIRST_DERIV_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
SECOND_DERIV_COLOUR = numpy.array([178, 223, 138], dtype=float) / 255
TFP_COLOUR = numpy.array([51, 160, 44], dtype=float) / 255

# TEMPERATURE_COLOUR = numpy.array([1, 115, 178], dtype=float) / 255
# SECOND_DERIV_COLOUR = numpy.array([222, 143, 5], dtype=float) / 255
# FIRST_DERIV_COLOUR = numpy.array([2, 158, 115], dtype=float) / 255
# TFP_COLOUR = numpy.array([213, 94, 0], dtype=float) / 255

TEMPERATURE_LEGEND_STRING = r'Thermal variable ($\tau$)'
FIRST_DERIV_LEGEND_STRING = (
    r'First deriv: $\vec{\nabla} \tau \cdot \hat{x} = '
    r'\frac{\partial \tau}{\partial x}$'
)
SECOND_DERIV_LEGEND_STRING = (
    r'Second deriv: '
    r'$\vec{\nabla} \left \| \left \| \vec{\nabla} \tau \right \| \right \| '
    r'\cdot \hat{x} = \frac{\partial}{\partial x} '
    r'\left \| \frac{\partial \tau}{\partial x} \right \|$'
)
TFP_LEGEND_STRING = 'TFP'

SOLID_LINE_WIDTH = 4
DASHED_LINE_WIDTH = 2

FONT_SIZE = 20
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 600
OUTPUT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'tfp_schematic/tfp_schematic_1d.jpg')


def _create_temperature_grid():
    """Creates 1-D temperature grid.

    P = number of grid points

    :return: temperatures_kelvins: length-P numpy array of temperatures.
    """

    temperatures_kelvins = numpy.full(NUM_GRID_POINTS, MIN_TEMPERATURE_KELVINS)

    for i in range(1, NUM_GRID_POINTS):
        if FIRST_POINT_IN_FRONT < i <= LAST_POINT_IN_FRONT:
            this_diff_kelvins = FRONT_GRADIENT_KELVINS_PT01 + 0.
        else:
            this_diff_kelvins = DEFAULT_GRADIENT_KELVINS_PT01 + 0.

        temperatures_kelvins[i] = (
            temperatures_kelvins[i - 1] + this_diff_kelvins
        )

    return temperatures_kelvins


def _run():
    """Plots 1-D version of TFP (thermal front parameter) schematic.

    This is effectively the main method.
    """

    temperatures_kelvins = _create_temperature_grid()
    first_derivs_kelvins_pt01 = numpy.gradient(temperatures_kelvins)
    second_derivs_kelvins_pt01 = numpy.gradient(
        numpy.absolute(first_derivs_kelvins_pt01)
    )

    this_ratio = (
        numpy.max(temperatures_kelvins) /
        numpy.max(first_derivs_kelvins_pt01)
    )

    first_derivs_unitless = first_derivs_kelvins_pt01 * this_ratio

    this_ratio = (
        numpy.max(temperatures_kelvins) /
        numpy.max(second_derivs_kelvins_pt01)
    )

    second_derivs_unitless = second_derivs_kelvins_pt01 * this_ratio

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    temperature_handle = axes_object.plot(
        temperatures_kelvins, color=TEMPERATURE_COLOUR, linestyle='solid',
        linewidth=SOLID_LINE_WIDTH
    )[0]

    second_deriv_handle = axes_object.plot(
        second_derivs_unitless, color=SECOND_DERIV_COLOUR, linestyle='solid',
        linewidth=SOLID_LINE_WIDTH
    )[0]

    first_deriv_handle = axes_object.plot(
        first_derivs_unitless, color=FIRST_DERIV_COLOUR, linestyle='dashed',
        linewidth=DASHED_LINE_WIDTH
    )[0]

    tfp_deriv_handle = axes_object.plot(
        -1 * second_derivs_unitless, color=TFP_COLOUR, linestyle='dashed',
        linewidth=DASHED_LINE_WIDTH
    )[0]

    axes_object.set_yticks([0])
    axes_object.set_xticks([], [])

    x_label_string = r'$x$-coordinate (increasing to the right)'
    axes_object.set_xlabel(x_label_string)

    legend_handles = [
        temperature_handle, first_deriv_handle, second_deriv_handle,
        tfp_deriv_handle
    ]

    legend_strings = [
        TEMPERATURE_LEGEND_STRING, FIRST_DERIV_LEGEND_STRING,
        SECOND_DERIV_LEGEND_STRING, TFP_LEGEND_STRING
    ]

    axes_object.legend(legend_handles, legend_strings, loc='lower right')

    print 'Saving figure to file: "{0:s}"...'.format(OUTPUT_FILE_NAME)
    pyplot.savefig(OUTPUT_FILE_NAME, dpi=FIGURE_RESOLUTION_DPI)
    pyplot.close()


if __name__ == '__main__':
    _run()
