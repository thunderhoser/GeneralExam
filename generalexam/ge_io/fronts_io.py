"""IO methods for front labels."""

import os.path
import numpy
import pandas
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

SENTINEL_VALUE = -9999.
TIME_FORMAT = '%Y%m%d%H'

FRONT_DIMENSION_KEY = 'front'
VERTEX_DIMENSION_KEY = 'vertex'
FRONT_TYPE_CHAR_DIM_KEY = 'front_type_character'
LATITUDE_MATRIX_KEY = 'latitude_matrix_deg'
LONGITUDE_MATRIX_KEY = 'longitude_matrix_deg'
FRONT_TYPES_KEY = 'front_type_strings'

COLD_FRONT_PIXEL_DIM_KEY = 'cold_front_pixel'
WARM_FRONT_PIXEL_DIM_KEY = 'warm_front_pixel'


def find_polyline_file(top_directory_name, valid_time_unix_sec,
                       raise_error_if_missing=True):
    """Finds NetCDF file with polylines at one time step.

    :param top_directory_name: Name of top-level directory with polyline files.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: polyline_file_name: Path to polyline line.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT)

    polyline_file_name = '{0:s}/{1:s}/frontal_polylines_{2:s}.nc'.format(
        top_directory_name, valid_time_string[:6], valid_time_string)

    if raise_error_if_missing and not os.path.isfile(polyline_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            polyline_file_name)
        raise ValueError(error_string)

    return polyline_file_name


def write_polylines_to_file(polyline_table, valid_time_unix_sec,
                            netcdf_file_name):
    """Writes polylines at one time step to NetCDF file.

    P = number of points in a given front

    :param polyline_table: pandas DataFrame with the following columns.  Each
        row is one front.
    polyline_table.front_type_string: Front type ("warm" or "cold").
    polyline_table.latitudes_deg: length-P numpy array of latitudes (deg N)
        along front.
    polyline_table.longitudes_deg: length-P numpy array of longitudes (deg E)
        along front.

    :param valid_time_unix_sec: Valid time.
    :param netcdf_file_name: Path to output file.
    """

    # Check input args.
    error_checking.assert_is_integer(valid_time_unix_sec)

    num_fronts = len(polyline_table.index)
    max_num_vertices_in_front = 0

    for i in range(num_fronts):
        front_utils.check_front_type_string(
            polyline_table[front_utils.FRONT_TYPE_COLUMN].values[i]
        )

        error_checking.assert_is_valid_lat_numpy_array(
            polyline_table[front_utils.LATITUDES_COLUMN].values[i])
        error_checking.assert_is_valid_lng_numpy_array(
            polyline_table[front_utils.LONGITUDES_COLUMN].values[i],
            positive_in_west_flag=True
        )

        this_num_vertices = len(
            polyline_table[front_utils.LATITUDES_COLUMN].values[i]
        )
        these_expected_dim = numpy.array([this_num_vertices], dtype=int)

        error_checking.assert_is_numpy_array(
            polyline_table[front_utils.LONGITUDES_COLUMN].values[i],
            exact_dimensions=these_expected_dim)

        max_num_vertices_in_front = max(
            [max_num_vertices_in_front, this_num_vertices]
        )

    # Reformat vertex lists into matrices.
    latitude_matrix_deg = numpy.full(
        (num_fronts, max_num_vertices_in_front), SENTINEL_VALUE)
    longitude_matrix_deg = numpy.full(
        (num_fronts, max_num_vertices_in_front), SENTINEL_VALUE)

    for i in range(num_fronts):
        this_num_vertices = len(
            polyline_table[front_utils.LATITUDES_COLUMN].values[i]
        )

        latitude_matrix_deg[i, :this_num_vertices] = polyline_table[
            front_utils.LATITUDES_COLUMN].values[i]
        longitude_matrix_deg[i, :this_num_vertices] = polyline_table[
            front_utils.LONGITUDES_COLUMN].values[i]

    # Open NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Write global attributes and dimensions.
    dataset_object.setncattr(front_utils.TIME_COLUMN, valid_time_unix_sec)
    dataset_object.createDimension(FRONT_DIMENSION_KEY, num_fronts)
    dataset_object.createDimension(VERTEX_DIMENSION_KEY,
                                   max_num_vertices_in_front)

    num_front_type_chars = numpy.max(numpy.array(
        [len(f) for f in front_utils.VALID_FRONT_TYPE_STRINGS]
    ))

    dataset_object.createDimension(FRONT_TYPE_CHAR_DIM_KEY,
                                   num_front_type_chars)

    # Write vertex latitudes and longitudes.
    dataset_object.createVariable(
        LATITUDE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(FRONT_DIMENSION_KEY, VERTEX_DIMENSION_KEY)
    )
    dataset_object.variables[LATITUDE_MATRIX_KEY][:] = latitude_matrix_deg

    dataset_object.createVariable(
        LONGITUDE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(FRONT_DIMENSION_KEY, VERTEX_DIMENSION_KEY)
    )
    dataset_object.variables[LONGITUDE_MATRIX_KEY][:] = longitude_matrix_deg

    # Write front types.
    this_string_type = 'S{0:d}'.format(num_front_type_chars)
    front_types_char_array = netCDF4.stringtochar(numpy.array(
        polyline_table[front_utils.FRONT_TYPE_COLUMN].values,
        dtype=this_string_type
    ))

    dataset_object.createVariable(
        FRONT_TYPES_KEY, datatype='S1',
        dimensions=(FRONT_DIMENSION_KEY, FRONT_TYPE_CHAR_DIM_KEY)
    )
    dataset_object.variables[FRONT_TYPES_KEY][:] = numpy.array(
        front_types_char_array)

    dataset_object.close()


def read_polylines_from_file(netcdf_file_name):
    """Reads polylines at one time step from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: polyline_table: See doc for `write_polylines_to_file`.
    :return: valid_time_unix_sec: Write valid time.
    """

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    valid_time_unix_sec = int(numpy.round(
        getattr(dataset_object, front_utils.TIME_COLUMN)
    ))

    latitude_matrix_deg = numpy.array(
        dataset_object.variables[LATITUDE_MATRIX_KEY][:]
    )
    longitude_matrix_deg = numpy.array(
        dataset_object.variables[LONGITUDE_MATRIX_KEY][:]
    )
    front_type_strings = [
        str(f) for f in netCDF4.chartostring(
            dataset_object.variables[FRONT_TYPES_KEY][:])
    ]

    dataset_object.close()

    polyline_table = pandas.DataFrame.from_dict(
        {front_utils.FRONT_TYPE_COLUMN: front_type_strings}
    )

    nested_array = polyline_table[[
        front_utils.FRONT_TYPE_COLUMN, front_utils.FRONT_TYPE_COLUMN
    ]].values.tolist()

    argument_dict = {
        front_utils.LATITUDES_COLUMN: nested_array,
        front_utils.LONGITUDES_COLUMN: nested_array
    }

    polyline_table = polyline_table.assign(**argument_dict)
    num_fronts = len(front_type_strings)

    for i in range(num_fronts):
        these_latitudes_deg = latitude_matrix_deg[i, ...]
        these_longitudes_deg = longitude_matrix_deg[i, ...]

        good_indices = numpy.where(these_latitudes_deg > SENTINEL_VALUE + 1)[0]

        polyline_table[front_utils.LATITUDES_COLUMN].values[i] = (
            these_latitudes_deg[good_indices]
        )
        polyline_table[front_utils.LONGITUDES_COLUMN].values[i] = (
            these_longitudes_deg[good_indices]
        )

    return polyline_table, valid_time_unix_sec


def find_gridded_file(top_directory_name, valid_time_unix_sec,
                      raise_error_if_missing=True):
    """Finds NetCDF file with gridded labels at one time step.

    :param top_directory_name: See doc for `find_polyline_file`.
    :param valid_time_unix_sec: Same.
    :param raise_error_if_missing: Same.
    :return: gridded_file_name: Path to gridded line.  If file is missing and
        `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: See doc for `find_polyline_file`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT)

    gridded_file_name = '{0:s}/{1:s}/frontal_grid_{2:s}.nc'.format(
        top_directory_name, valid_time_string[:6], valid_time_string)

    if raise_error_if_missing and not os.path.isfile(gridded_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            gridded_file_name)
        raise ValueError(error_string)

    return gridded_file_name


def write_grid_to_file(gridded_label_table, netcdf_file_name):
    """Writes gridded labels (for one time step) to NetCDF file.

    C = number of pixels intersected by any cold front
    W = number of pixels intersected by any warm front

    :param gridded_label_table: pandas DataFrame with the following columns (and
        only one row).
    gridded_label_table.cold_front_rows: length-C numpy array with row indices
        of cold-frontal pixels.
    gridded_label_table.cold_front_columns: Same but for columns.
    gridded_label_table.warm_front_rows: length-W numpy array with row indices
        of warm-frontal pixels.
    gridded_label_table.warm_front_columns: Same but for columns.
    gridded_label_table.valid_time_unix_sec: Valid time.
    gridded_label_table.dilation_distance_metres: Dilation distance used to
        convert polylines to grid.
    gridded_label_table.model_name: Name of model used to create grid (must be
        accepted by `_check_model_name`).

    :param netcdf_file_name: Path to output file.
    """

    valid_time_unix_sec = gridded_label_table[front_utils.TIME_COLUMN].values[0]
    model_name = gridded_label_table[front_utils.MODEL_NAME_COLUMN].values[0]
    dilation_distance_metres = gridded_label_table[
        front_utils.DILATION_DISTANCE_COLUMN
    ].values[0]

    cold_front_rows = gridded_label_table[
        front_utils.COLD_FRONT_ROWS_COLUMN].values[0]
    cold_front_columns = gridded_label_table[
        front_utils.COLD_FRONT_COLUMNS_COLUMN].values[0]
    warm_front_rows = gridded_label_table[
        front_utils.WARM_FRONT_ROWS_COLUMN].values[0]
    warm_front_columns = gridded_label_table[
        front_utils.WARM_FRONT_COLUMNS_COLUMN].values[0]

    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_geq(dilation_distance_metres, 0)
    front_utils.check_nwp_model_name(model_name)

    error_checking.assert_is_integer_numpy_array(cold_front_rows)
    error_checking.assert_is_geq_numpy_array(cold_front_rows, 0)

    num_cold_front_pixels = len(cold_front_rows)
    these_expected_dim = numpy.array([num_cold_front_pixels], dtype=int)

    error_checking.assert_is_integer_numpy_array(cold_front_columns)
    error_checking.assert_is_geq_numpy_array(cold_front_columns, 0)
    error_checking.assert_is_numpy_array(
        cold_front_columns, exact_dimensions=these_expected_dim)

    error_checking.assert_is_integer_numpy_array(warm_front_rows)
    error_checking.assert_is_geq_numpy_array(warm_front_rows, 0)

    num_warm_front_pixels = len(warm_front_rows)
    these_expected_dim = numpy.array([num_warm_front_pixels], dtype=int)

    error_checking.assert_is_integer_numpy_array(warm_front_columns)
    error_checking.assert_is_geq_numpy_array(warm_front_columns, 0)
    error_checking.assert_is_numpy_array(
        warm_front_columns, exact_dimensions=these_expected_dim)

    # Open NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Write global attributes and dimensions.
    dataset_object.setncattr(front_utils.TIME_COLUMN, valid_time_unix_sec)
    dataset_object.setncattr(
        front_utils.DILATION_DISTANCE_COLUMN, dilation_distance_metres)
    dataset_object.setncattr(front_utils.MODEL_NAME_COLUMN, model_name)

    dataset_object.createDimension(
        COLD_FRONT_PIXEL_DIM_KEY, num_cold_front_pixels)
    dataset_object.createDimension(
        WARM_FRONT_PIXEL_DIM_KEY, num_warm_front_pixels)

    dataset_object.createVariable(
        front_utils.COLD_FRONT_ROWS_COLUMN, datatype=numpy.int32,
        dimensions=COLD_FRONT_PIXEL_DIM_KEY)
    dataset_object.variables[
        front_utils.COLD_FRONT_ROWS_COLUMN][:] = cold_front_rows

    dataset_object.createVariable(
        front_utils.COLD_FRONT_COLUMNS_COLUMN, datatype=numpy.int32,
        dimensions=COLD_FRONT_PIXEL_DIM_KEY)
    dataset_object.variables[
        front_utils.COLD_FRONT_COLUMNS_COLUMN][:] = cold_front_columns

    dataset_object.createVariable(
        front_utils.WARM_FRONT_ROWS_COLUMN, datatype=numpy.int32,
        dimensions=WARM_FRONT_PIXEL_DIM_KEY)
    dataset_object.variables[
        front_utils.WARM_FRONT_ROWS_COLUMN][:] = warm_front_rows

    dataset_object.createVariable(
        front_utils.WARM_FRONT_COLUMNS_COLUMN, datatype=numpy.int32,
        dimensions=WARM_FRONT_PIXEL_DIM_KEY)
    dataset_object.variables[
        front_utils.WARM_FRONT_COLUMNS_COLUMN][:] = warm_front_columns

    dataset_object.close()


def read_grid_from_file(netcdf_file_name):
    """Reads gridded labels (for one time step) from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: gridded_label_table: See doc for `write_grid_to_file`.
    """

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    valid_time_unix_sec = int(numpy.round(
        getattr(dataset_object, front_utils.TIME_COLUMN)
    ))
    dilation_distance_metres = float(
        getattr(dataset_object, front_utils.DILATION_DISTANCE_COLUMN)
    )
    model_name = str(
        getattr(dataset_object, front_utils.MODEL_NAME_COLUMN)
    )

    gridded_label_dict = {
        front_utils.TIME_COLUMN: numpy.array([valid_time_unix_sec], dtype=int),
        front_utils.DILATION_DISTANCE_COLUMN:
            numpy.array([dilation_distance_metres], dtype=float),
        front_utils.MODEL_NAME_COLUMN: [model_name]
    }

    gridded_label_table = pandas.DataFrame.from_dict(gridded_label_dict)

    nested_array = gridded_label_table[[
        front_utils.TIME_COLUMN, front_utils.TIME_COLUMN
    ]].values.tolist()

    argument_dict = {
        front_utils.COLD_FRONT_ROWS_COLUMN: nested_array,
        front_utils.COLD_FRONT_COLUMNS_COLUMN: nested_array,
        front_utils.WARM_FRONT_ROWS_COLUMN: nested_array,
        front_utils.WARM_FRONT_COLUMNS_COLUMN: nested_array
    }

    gridded_label_table = gridded_label_table.assign(**argument_dict)

    gridded_label_table[front_utils.COLD_FRONT_ROWS_COLUMN].values[
        0
    ] = numpy.round(numpy.array(
        dataset_object.variables[front_utils.COLD_FRONT_ROWS_COLUMN][:]
    )).astype(int)

    gridded_label_table[front_utils.COLD_FRONT_COLUMNS_COLUMN].values[
        0
    ] = numpy.round(numpy.array(
        dataset_object.variables[front_utils.COLD_FRONT_COLUMNS_COLUMN][:]
    )).astype(int)

    gridded_label_table[front_utils.WARM_FRONT_ROWS_COLUMN].values[
        0
    ] = numpy.round(numpy.array(
        dataset_object.variables[front_utils.WARM_FRONT_ROWS_COLUMN][:]
    )).astype(int)

    gridded_label_table[front_utils.WARM_FRONT_COLUMNS_COLUMN].values[
        0
    ] = numpy.round(numpy.array(
        dataset_object.variables[front_utils.WARM_FRONT_COLUMNS_COLUMN][:]
    )).astype(int)

    dataset_object.close()
    return gridded_label_dict
