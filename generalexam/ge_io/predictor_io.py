"""IO methods for predictors (processed NARR or ERA5 grids)."""

import pickle
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_io import netcdf_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import nwp_model_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import predictor_utils

TIME_FORMAT = '%Y%m%d%H'

VALID_TIME_KEY = 'valid_time_unix_sec'
ROW_DIMENSION_KEY = 'grid_row'
COLUMN_DIMENSION_KEY = 'grid_column'
FIELD_DIMENSION_KEY = 'field'
FIELD_NAME_CHAR_DIM_KEY = 'field_name_character'

THIS_DIR_NAME = os.path.dirname(__file__)
PARENT_DIR_NAME = '/'.join(THIS_DIR_NAME.split('/')[:-1])
OROGRAPHY_FILE_NAME = '{0:s}/era5_orography.nc'.format(PARENT_DIR_NAME)

DATASET_OBJECT = netcdf_io.open_netcdf(OROGRAPHY_FILE_NAME)
FULL_OROGRAPHY_MATRIX_M_ASL = numpy.array(
    DATASET_OBJECT.variables[predictor_utils.DATA_MATRIX_KEY][:]
)
FULL_OROGRAPHY_MATRIX_M_ASL = numpy.expand_dims(
    FULL_OROGRAPHY_MATRIX_M_ASL, axis=0
)
DATASET_OBJECT.close()


def _add_orography(predictor_dict):
    """Adds orography (surface height above sea level) as predictor.

    :param predictor_dict: See doc for `predictor_utils.check_predictor_dict`.
    :return: predictor_dict: Same but with extra predictor.
    """

    predictor_matrix = predictor_dict[predictor_utils.DATA_MATRIX_KEY]
    grid_name = nwp_model_utils.dimensions_to_grid(
        num_rows=predictor_matrix.shape[1],
        num_columns=predictor_matrix.shape[2]
    )

    if grid_name == nwp_model_utils.NAME_OF_EXTENDED_221GRID:
        new_predictor_matrix = FULL_OROGRAPHY_MATRIX_M_ASL
    else:
        new_predictor_matrix = (
            FULL_OROGRAPHY_MATRIX_M_ASL[:, 100:-100, 100:-100, :]
        )

    num_times = predictor_matrix.shape[0]
    new_predictor_matrix = numpy.repeat(
        new_predictor_matrix, repeats=num_times, axis=0
    )
    predictor_matrix = numpy.concatenate(
        (predictor_matrix, new_predictor_matrix), axis=-1
    )

    dummy_pressures_mb = numpy.array(
        [predictor_utils.DUMMY_SURFACE_PRESSURE_MB], dtype=int
    )

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = predictor_matrix
    predictor_dict[predictor_utils.FIELD_NAMES_KEY].append(
        predictor_utils.HEIGHT_NAME
    )
    predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] = numpy.concatenate((
        predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY], dummy_pressures_mb
    ), axis=0)

    return predictor_dict


def _subset_channels(predictor_dict, metadata_only, field_names,
                     pressure_levels_mb):
    """Subsets predictors by channel.

    C = number of channels to keep

    :param predictor_dict: See doc for `predictor_utils.check_predictor_dict`.
    :param metadata_only: Boolean flag.  If True, will change only metadata in
        `predictor_dict`.
    :param field_names: length-C list of field names.
    :param pressure_levels_mb: length-C numpy array of pressure levels
        (millibars).
    :return: example_dict: Same as input but maybe with different channels.
    """

    # Check input args.
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1
    )

    error_checking.assert_is_numpy_array(pressure_levels_mb)
    pressure_levels_mb = numpy.round(pressure_levels_mb).astype(int)

    num_channels = len(field_names)
    these_expected_dim = numpy.array([num_channels], dtype=int)
    error_checking.assert_is_numpy_array(
        pressure_levels_mb, exact_dimensions=these_expected_dim
    )

    orography_flags = numpy.logical_and(
        numpy.array(field_names) == predictor_utils.HEIGHT_NAME,
        pressure_levels_mb == predictor_utils.DUMMY_SURFACE_PRESSURE_MB
    )
    add_orography = numpy.any(orography_flags)

    if add_orography:
        these_indices = numpy.where(numpy.invert(orography_flags))[0]
        field_names = [field_names[k] for k in these_indices]
        pressure_levels_mb = pressure_levels_mb[these_indices]

    # Do the subsetting.
    all_field_names = numpy.array(
        predictor_dict[predictor_utils.FIELD_NAMES_KEY]
    )
    all_pressure_levels_mb = predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY]

    indices_to_keep = [
        numpy.where(numpy.logical_and(
            all_field_names == n, all_pressure_levels_mb == l
        ))[0][0]
        for n, l in zip(field_names, pressure_levels_mb)
    ]

    predictor_dict[predictor_utils.FIELD_NAMES_KEY] = field_names
    predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] = pressure_levels_mb

    if metadata_only:
        if add_orography:
            dummy_pressures_mb = numpy.array(
                [predictor_utils.DUMMY_SURFACE_PRESSURE_MB], dtype=int
            )

            predictor_dict[predictor_utils.FIELD_NAMES_KEY].append(
                predictor_utils.HEIGHT_NAME
            )
            predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY] = (
                numpy.concatenate((
                    predictor_dict[predictor_utils.PRESSURE_LEVELS_KEY],
                    dummy_pressures_mb
                ), axis=0)
            )

        return predictor_dict

    predictor_dict[predictor_utils.DATA_MATRIX_KEY] = (
        predictor_dict[predictor_utils.DATA_MATRIX_KEY][..., indices_to_keep]
    )

    if not add_orography:
        return predictor_dict

    return _add_orography(predictor_dict)


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
    :param predictor_dict: See doc for `predictor_utils.check_predictor_dict`.
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

    if not metadata_only:
        predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.array(
            dataset_object.variables[predictor_utils.DATA_MATRIX_KEY][:]
        )
        predictor_dict[predictor_utils.DATA_MATRIX_KEY] = numpy.expand_dims(
            predictor_dict[predictor_utils.DATA_MATRIX_KEY], axis=0
        )

        if is_old_file:
            predictor_dict[predictor_utils.DATA_MATRIX_KEY] = (
                predictor_dict[predictor_utils.DATA_MATRIX_KEY][..., 0, :]
            )

    if field_names_to_keep is None or pressure_levels_to_keep_mb is None:
        return predictor_dict

    return _subset_channels(
        predictor_dict=predictor_dict, metadata_only=metadata_only,
        field_names=field_names_to_keep,
        pressure_levels_mb=pressure_levels_to_keep_mb)


def write_normalization_params(mean_value_dict, standard_deviation_dict,
                               pickle_file_name):
    """Writes normalization params to Pickle file.

    :param mean_value_dict: Dictionary of mean values.  Each key is a tuple with
        (predictor_name, pressure_level_mb), where the pressure level must be an
        integer.
    :param standard_deviation_dict: Same but for standard deviations.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(mean_value_dict, pickle_file_handle)
    pickle.dump(standard_deviation_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_normalization_params(pickle_file_name):
    """Reads normalization params from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: mean_value_dict: See doc for `write_normalization_params`.
    :return: standard_deviation_dict: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    mean_value_dict = pickle.load(pickle_file_handle)
    standard_deviation_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return mean_value_dict, standard_deviation_dict
