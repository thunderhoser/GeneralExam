"""IO methods for predictors (processed NARR or ERA5 grids)."""

import copy
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import predictor_utils

TIME_FORMAT = '%Y%m%d%H'

VALID_TIME_KEY = 'valid_time_unix_sec'
ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
FIELD_DIMENSION_KEY = 'field'
FIELD_NAME_CHAR_DIM_KEY = 'field_name_character'


def find_file(
        top_directory_name, valid_time_unix_sec, raise_error_if_missing=True):
    """Finds predictor file (NetCDF with all data at one time step).

    :param top_directory_name: Name of top-level directory with predictor files.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: predictor_file_name: Path to predictor file.  If file is missing
        and `raise_error_if_missing = False`, this will be the expected path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT)

    # TODO(thunderhoser): Remove "era5" from file names.
    predictor_file_name = '{0:s}/{1:s}/era5_processed_{2:s}.nc'.format(
        top_directory_name, valid_time_string[:6], valid_time_string)

    if raise_error_if_missing and not os.path.isfile(predictor_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            predictor_file_name)
        raise ValueError(error_string)

    return predictor_file_name


def write_file(netcdf_file_name, predictor_dict):
    """Writes data to predictor file (NetCDF with all data at one time step).

    :param netcdf_file_name: Path to output file.
    :param predictor_dict: See doc for `check_predictor_dict`.
    :raises: ValueError: if `predictor_dict` contains more than one time step.
    """

    num_times = len(predictor_dict[predictor_utils.VALID_TIMES_KEY])
    if num_times > 1:
        error_string = (
            'Dictionary should contain one time step, not {0:d}.'
        ).format(num_times)

        raise ValueError(error_string)

    predictor_utils.check_predictor_dict(predictor_dict)
    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY][0, ...]

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes.
    dataset_object.setncattr(
        VALID_TIME_KEY, predictor_dict[predictor_utils.VALID_TIMES_KEY][0]
    )

    # Set dimensions.
    dataset_object.createDimension(
        ROW_DIMENSION_KEY,
        predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY,
        predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[1]
    )
    dataset_object.createDimension(
        FIELD_DIMENSION_KEY,
        predictor_dict[predictor_utils.DATA_MATRIX_KEY].shape[2]
    )

    num_field_name_chars = numpy.max(numpy.array(
        [len(f) for f in predictor_dict[predictor_utils.FIELD_NAMES_KEY]]
    ))

    dataset_object.createDimension(FIELD_NAME_CHAR_DIM_KEY,
                                   num_field_name_chars)

    # Add data matrix.
    dataset_object.createVariable(
        predictor_utils.DATA_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
                    FIELD_DIMENSION_KEY)
    )

    dataset_object.variables[predictor_utils.DATA_MATRIX_KEY][:] = (
        predictor_dict[predictor_utils.DATA_MATRIX_KEY]
    )

    # Add latitudes.
    if predictor_dict[predictor_utils.LATITUDES_KEY] is not None:
        dataset_object.createVariable(
            predictor_utils.LATITUDES_KEY, datatype=numpy.float32,
            dimensions=ROW_DIMENSION_KEY)

        dataset_object.variables[predictor_utils.LATITUDES_KEY][:] = (
            predictor_dict[predictor_utils.LATITUDES_KEY]
        )

    # Add longitudes.
    if predictor_dict[predictor_utils.LONGITUDES_KEY] is not None:
        dataset_object.createVariable(
            predictor_utils.LONGITUDES_KEY, datatype=numpy.float32,
            dimensions=COLUMN_DIMENSION_KEY)

        dataset_object.variables[predictor_utils.LONGITUDES_KEY][:] = (
            predictor_dict[predictor_utils.LONGITUDES_KEY]
        )

    # Add pressure levels.
    dataset_object.createVariable(
        predictor_utils.PRESSURE_LEVELS_KEY, datatype=numpy.int32,
        dimensions=FIELD_DIMENSION_KEY)

    dataset_object.variables[predictor_utils.PRESSURE_LEVELS_KEY][:] = (
        predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY]
    )

    # Add field names.
    this_string_type = 'S{0:d}'.format(num_field_name_chars)
    field_names_char_array = netCDF4.stringtochar(numpy.array(
        predictor_dict[predictor_utils.FIELD_NAMES_KEY], dtype=this_string_type
    ))

    dataset_object.createVariable(
        predictor_utils.FIELD_NAMES_KEY, datatype='S1',
        dimensions=(FIELD_DIMENSION_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[predictor_utils.FIELD_NAMES_KEY][:] = numpy.array(
        field_names_char_array)

    dataset_object.close()


def read_file(
        netcdf_file_name, metadata_only=False, pressure_levels_to_keep_mb=None,
        field_names_to_keep=None):
    """Reads predictor file (NetCDF with all data at one time step).

    C = number of predictors to keep

    :param netcdf_file_name: Path to input file.
    :param metadata_only: Boolean flag.  If True, will read only metadata
        (everything except the big data matrix).
    :param pressure_levels_to_keep_mb: [used only if `metadata_only == False`]
        length-C numpy array of pressure levels (millibars) to read.  Use 1013
        to denote surface.  If you want to read all pressure levels, make this
        None.
    :param field_names_to_keep: [used only if `metadata_only == False`]
        length-C list of field names to read.  If you want to read all fields,
        make this None.
    :return: predictor_dict: See doc for `write_file`.  If
        `metadata_only = True`, this dictionary will not contain "data_matrix".
    """

    error_checking.assert_is_boolean(metadata_only)

    dataset_object = netcdf_io.open_netcdf(
        netcdf_file_name=netcdf_file_name, raise_error_if_fails=True)

    valid_time_unix_sec = int(numpy.round(
        getattr(dataset_object, VALID_TIME_KEY)
    ))
    valid_times_unix_sec = numpy.array([valid_time_unix_sec], dtype=int)

    field_names = [
        str(f) for f in netCDF4.chartostring(
            dataset_object.variables[predictor_utils.FIELD_NAMES_KEY][:]
        )
    ]

    pressure_levels_mb = numpy.array(
        dataset_object.variables[predictor_utils.PRESSURE_LEVELS_KEY][:],
        dtype=int
    )

    is_old_file = (len(field_names) != len(pressure_levels_mb))
    if is_old_file:
        pressure_levels_mb = numpy.full(
            len(field_names), pressure_levels_mb[0], dtype=int
        )

    predictor_dict = {
        predictor_utils.VALID_TIMES_KEY: valid_times_unix_sec,
        predictor_utils.FIELD_NAMES_KEY: field_names,
        predictor_utils.PRESSURE_LEVELS_KEY: pressure_levels_mb
    }

    if predictor_utils.LATITUDES_KEY in dataset_object.variables:
        predictor_dict.update({
            predictor_utils.LATITUDES_KEY: numpy.array(
                dataset_object.variables[predictor_utils.LATITUDES_KEY][:]
            ),
            predictor_utils.LONGITUDES_KEY: numpy.array(
                dataset_object.variables[predictor_utils.LONGITUDES_KEY][:]
            )
        })
    else:
        predictor_dict.update({
            predictor_utils.LATITUDES_KEY: None,
            predictor_utils.LONGITUDES_KEY: None
        })

    if metadata_only:
        return predictor_dict

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.array(
        dataset_object.variables[predictor_utils.DATA_MATRIX_KEY][:]
    )
    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.expand_dims(
        predictor_dict[predictor_utils.DATA_MATRIX_KEY], axis=0)

    if is_old_file:
        predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_dict[
            predictor_utils.DATA_MATRIX_KEY][..., 0, :]

    if field_names_to_keep is None and pressure_levels_to_keep_mb is None:
        field_names_to_keep = copy.deepcopy(
            predictor_dict[predictor_utils.FIELD_NAMES_KEY]
        )
        pressure_levels_to_keep_mb = (
            predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] + 0
        )

    pressure_levels_to_keep_mb = numpy.round(
        pressure_levels_to_keep_mb
    ).astype(int)

    error_checking.assert_is_numpy_array(
        numpy.array(field_names_to_keep), num_dimensions=1)

    num_fields_to_keep = len(field_names_to_keep)
    error_checking.assert_is_numpy_array(
        pressure_levels_to_keep_mb,
        exact_dimensions=numpy.array([num_fields_to_keep], dtype=int)
    )

    field_indices = [
        numpy.where(numpy.logical_and(
            numpy.array(predictor_dict[predictor_utils.FIELD_NAMES_KEY]) == f,
            predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] == p
        ))[0][0]
        for f, p in zip(field_names_to_keep, pressure_levels_to_keep_mb)
    ]

    predictor_dict[
        predictor_utils.PRESSURE_LEVELS_KEY] = pressure_levels_to_keep_mb
    predictor_dict[predictor_utils.FIELD_NAMES_KEY] = field_names_to_keep

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_dict[
        predictor_utils.DATA_MATRIX_KEY][..., field_indices]

    return predictor_dict
