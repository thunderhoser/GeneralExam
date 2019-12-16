"""Processes ERA5 orography data.

Specifically, this script does the following:

[1] Converts values from geopotential to geopotential height.
[2] Interpolates values to NARR grid.
[3] Saves interpolated geopotential height to new file.
"""

import argparse
import numpy
from netCDF4 import Dataset
from gewittergefahr.gg_utils import nwp_model_utils
from generalexam.ge_io import predictor_io
from generalexam.ge_io import era5_input
from generalexam.ge_utils import predictor_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RAW_FIELD_NAME = 'z'
GRAV_ACCELERATION_M_S02 = 9.80665

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  This should be a NetCDF file downloaded from the '
    'Copernicus climate-data store, where the variable name for surface '
    'geopotential is "z".'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Processed values will be written here by '
    '`predictor_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING)

INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING)


def _run(input_file_name, output_file_name):
    """Processes ERA5 orography data.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    print('Reading orography from: "{0:s}"...'.format(input_file_name))
    dataset_object = Dataset(input_file_name)

    height_matrix_m_asl = numpy.array(
        dataset_object.variables[RAW_FIELD_NAME][:]
    )
    height_matrix_m_asl = (
        numpy.expand_dims(height_matrix_m_asl, axis=-1) /
        GRAV_ACCELERATION_M_S02
    )

    source_latitudes_deg = numpy.array(
        dataset_object.variables[era5_input.NETCDF_LATITUDES_KEY][:]
    )
    source_longitudes_deg = numpy.array(
        dataset_object.variables[era5_input.NETCDF_LONGITUDES_KEY][:]
    )
    source_latitudes_deg = source_latitudes_deg[::-1]

    source_latitudes_deg = source_latitudes_deg[1:]
    height_matrix_m_asl = height_matrix_m_asl[:, 1:, ...]

    print((
        'Minimum height before interp = {0:.1f} m ASL ... max = {1:.1f} ... '
        'mean = {2:.1f} ... median = {3:.1f}'
    ).format(
        numpy.min(height_matrix_m_asl), numpy.max(height_matrix_m_asl),
        numpy.mean(height_matrix_m_asl), numpy.median(height_matrix_m_asl)
    ))
    print(SEPARATOR_STRING)

    predictor_dict = {
        predictor_utils.DATA_MATRIX_KEY: height_matrix_m_asl,
        predictor_utils.VALID_TIMES_KEY: numpy.array([0], dtype=int),
        predictor_utils.LATITUDES_KEY: source_latitudes_deg,
        predictor_utils.LONGITUDES_KEY: source_longitudes_deg,
        predictor_utils.FIELD_NAMES_KEY: [predictor_utils.HEIGHT_NAME],
        predictor_utils.PRESSURE_LEVELS_KEY: numpy.array(
            [predictor_utils.DUMMY_SURFACE_PRESSURE_MB], dtype=int
        )
    }

    predictor_dict = era5_input.interp_to_narr_grid(
        predictor_dict=predictor_dict,
        grid_name=nwp_model_utils.NAME_OF_EXTENDED_221GRID)

    height_matrix_m_asl = predictor_dict[predictor_utils.DATA_MATRIX_KEY]

    print((
        'Minimum height after interp = {0:.1f} m ASL ... max = {1:.1f} ... '
        'mean = {2:.1f} ... median = {3:.1f}'
    ).format(
        numpy.min(height_matrix_m_asl), numpy.max(height_matrix_m_asl),
        numpy.mean(height_matrix_m_asl), numpy.median(height_matrix_m_asl)
    ))

    print(SEPARATOR_STRING)
    print('Writing processed orography to: "{0:s}"...'.format(output_file_name))
    predictor_io.write_file(
        netcdf_file_name=output_file_name, predictor_dict=predictor_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
