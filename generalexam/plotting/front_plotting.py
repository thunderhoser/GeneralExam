"""Plotting methods for warm and cold fronts."""

import numpy
from generalexam.ge_utils import front_utils
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

DEFAULT_WARM_FRONT_COLOUR = numpy.array([228., 26., 28.]) / 255
DEFAULT_COLD_FRONT_COLOUR = numpy.array([31., 120., 180.]) / 255
DEFAULT_LINE_WIDTH = 2.
DEFAULT_LINE_STYLE = 'solid'


def plot_front(
        latitudes_deg, longitudes_deg, basemap_object, axes_object,
        front_type=None, line_colour=None, line_width=DEFAULT_LINE_WIDTH,
        line_style=DEFAULT_LINE_STYLE):
    """Plots either warm or cold front.

    N = number of points in front (polyline)

    :param latitudes_deg: 1-D numpy array of latitudes (deg N).
    :param longitudes_deg: 1-D numpy array of longitudes (deg N).
    :param basemap_object: Instance of `mpl_toolkits.basemap.Basemap`.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
    :param front_type: Type of front (string).  Used only to determine line
        colour (if `line_colour` is left as None).
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
        Defaults to `DEFAULT_WARM_FRONT_COLOUR` or `DEFAULT_COLD_FRONT_COLOUR`.
    :param line_width: Line width (real positive number).
    :param line_style: Line style (in any format accepted by
        `matplotlib.lines`).
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg)
    error_checking.assert_is_numpy_array(latitudes_deg, num_dimensions=1)
    num_points = len(latitudes_deg)

    longitudes_deg = lng_conversion.convert_lng_positive_in_west(longitudes_deg)
    error_checking.assert_is_numpy_array(
        longitudes_deg, exact_dimensions=numpy.array([num_points]))

    if line_colour is None:
        front_utils.check_front_type(front_type)
        if front_type == front_utils.WARM_FRONT_STRING_ID:
            line_colour = DEFAULT_WARM_FRONT_COLOUR
        else:
            line_colour = DEFAULT_COLD_FRONT_COLOUR

    x_coords_metres, y_coords_metres = basemap_object(
        longitudes_deg, latitudes_deg)
    axes_object.plot(
        x_coords_metres, y_coords_metres, color=line_colour,
        linestyle=line_style, linewidth=line_width)
