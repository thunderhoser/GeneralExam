"""Plots predicted and observed fronts for one time step."""

import pickle
import argparse
import numpy
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import nwp_plotting
from generalexam.evaluation import object_based_evaluation as obe
from generalexam.plotting import front_plotting

# TODO(thunderhoser): Make this two time steps.

MIN_LATITUDE_DEG = 20.
MIN_LONGITUDE_DEG = 220.
MAX_LATITUDE_DEG = 80.
MAX_LONGITUDE_DEG = 290.
PARALLEL_SPACING_DEG = 10.
MERIDIAN_SPACING_DEG = 20.

PRESSURE_LEVEL_MB = 1000
WIND_FIELD_NAMES = [
    processed_narr_io.U_WIND_EARTH_RELATIVE_NAME,
    processed_narr_io.V_WIND_EARTH_RELATIVE_NAME
]
NARR_FIELD_NAMES = WIND_FIELD_NAMES + [processed_narr_io.WET_BULB_THETA_NAME]

FRONT_LINE_WIDTH = 8
BORDER_COLOUR = numpy.full(3, 152. / 255)
WARM_FRONT_COLOUR = numpy.array([217., 95., 2.]) / 255
COLD_FRONT_COLOUR = numpy.array([117., 112., 179.]) / 255

THERMAL_COLOUR_MAP_OBJECT = pyplot.cm.YlGn
MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

WIND_COLOUR_MAP_OBJECT = pyplot.cm.binary
WIND_BARB_LENGTH = 8
EMPTY_WIND_BARB_RADIUS = 0.1
MIN_COLOUR_WIND_SPEED_KT = -1.
MAX_COLOUR_WIND_SPEED_KT = 0.
PLOT_EVERY_KTH_WIND_BARB = 2

OUTPUT_RESOLUTION_DPI = 600
OUTPUT_SIZE_PIXELS = int(1e7)

VALID_TIMES_ARG_NAME = 'valid_time_strings'
VALID_TIMES_HELP_STRING = (
    'List of two valid times (format "yyyy-mm-dd-HH").  Predictions and '
    'observations will be plotted for each valid time, with the top (bottom) '
    'row of the figure showing the first (second) valid time.')

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIMES_ARG_NAME, type=str, nargs='+', required=True,
    help=VALID_TIMES_HELP_STRING)

