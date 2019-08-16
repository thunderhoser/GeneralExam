"""Methods for creating climatology of fronts."""

import glob
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from generalexam.ge_utils import front_utils

FRONT_LABELS_STRING = 'front_labels'
FRONT_PROPERTIES_STRING = 'front_properties'
FRONT_COUNTS_STRING = 'front_counts'
FRONT_STATS_STRING = 'front_statistics'
BASIC_FILE_TYPE_STRINGS = [FRONT_LABELS_STRING, FRONT_PROPERTIES_STRING]
AGGREGATED_FILE_TYPE_STRINGS = [FRONT_COUNTS_STRING, FRONT_STATS_STRING]

WINTER_STRING = 'winter'
SPRING_STRING = 'spring'
SUMMER_STRING = 'summer'
FALL_STRING = 'fall'

VALID_SEASON_STRINGS = [
    WINTER_STRING, SPRING_STRING, SUMMER_STRING, FALL_STRING
]

FILE_NAME_TIME_FORMAT = '%Y-%m-%d-%H%M%S'
YEAR_MONTH_REGEX = '[0-9][0-9][0-9][0-9][0-1][0-9]'
FILE_NAME_TIME_REGEX = (
    '[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]-[0-2][0-9][0-5][0-9][0-5][0-9]'
)

ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
FRONT_LABELS_KEY = 'label_matrix'
UNIQUE_FRONT_LABELS_KEY = 'unique_label_matrix'
PREDICTION_FILE_KEY = 'prediction_file_name'
SEPARATION_TIME_KEY = 'separation_time_sec'

PREDICTION_FILE_DIM_KEY = 'prediction_file'
PREDICTION_FILE_CHAR_DIM_KEY = 'prediction_file_char'
PREDICTION_FILES_KEY = 'prediction_file_names'
FIRST_TIME_KEY = 'first_time_unix_sec'
LAST_TIME_KEY = 'last_time_unix_sec'
HOURS_KEY = 'hours'
MONTHS_KEY = 'months'

WARM_FRONT_LENGTHS_KEY = 'wf_length_matrix_metres'
WARM_FRONT_AREAS_KEY = 'wf_area_matrix_m2'
COLD_FRONT_LENGTHS_KEY = 'cf_length_matrix_metres'
COLD_FRONT_AREAS_KEY = 'cf_area_matrix_m2'

WF_LENGTH_PROPERTY_NAME = 'wf_length'
WF_AREA_PROPERTY_NAME = 'wf_area'
WF_FREQ_PROPERTY_NAME = 'wf_frequency'
CF_LENGTH_PROPERTY_NAME = 'cf_length'
CF_AREA_PROPERTY_NAME = 'cf_area'
CF_FREQ_PROPERTY_NAME = 'cf_frequency'

VALID_PROPERTY_NAMES = [
    WF_LENGTH_PROPERTY_NAME, WF_AREA_PROPERTY_NAME, WF_FREQ_PROPERTY_NAME,
    CF_LENGTH_PROPERTY_NAME, CF_AREA_PROPERTY_NAME, CF_FREQ_PROPERTY_NAME
]

NUM_WF_LABELS_KEY = 'num_wf_labels_matrix'
NUM_UNIQUE_WF_KEY = 'num_unique_wf_matrix'
NUM_CF_LABELS_KEY = 'num_cf_labels_matrix'
NUM_UNIQUE_CF_KEY = 'num_unique_cf_matrix'

MEAN_WF_LENGTHS_KEY = 'mean_wf_length_matrix_metres'
MEAN_WF_AREAS_KEY = 'mean_wf_area_matrix_m2'
MEAN_CF_LENGTHS_KEY = 'mean_cf_length_matrix_metres'
MEAN_CF_AREAS_KEY = 'mean_cf_area_matrix_m2'

PROPERTY_NAME_KEY = 'property_name'
NUM_ITERATIONS_KEY = 'num_iterations'
CONFIDENCE_LEVEL_KEY = 'confidence_level'
FIRST_GRID_ROW_KEY = 'first_grid_row'
FIRST_GRID_COLUMN_KEY = 'first_grid_column'

BASELINE_INPUT_FILE_DIM_KEY = 'baseline_input_file'
TRIAL_INPUT_FILE_DIM_KEY = 'trial_input_file'
INPUT_FILE_CHAR_DIM_KEY = 'input_file_char'

BASELINE_MATRIX_KEY = 'baseline_mean_matrix'
TRIAL_MATRIX_KEY = 'trial_mean_matrix'
SIGNIFICANCE_MATRIX_KEY = 'significance_matrix'
BASELINE_INPUT_FILES_KEY = 'baseline_input_file_names'
TRIAL_INPUT_FILES_KEY = 'trial_input_file_names'


def _check_season(season_string):
    """Error-checks season.

    :param season_string: Season.
    :raises: ValueError: if `season_string not in VALID_SEASON_STRINGS`.
    """

    error_checking.assert_is_string(season_string)

    if season_string not in VALID_SEASON_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid seasons (listed above) do not include "{1:s}".'
        ).format(str(VALID_SEASON_STRINGS), season_string)

        raise ValueError(error_string)


def _check_basic_file_type(file_type_string):
    """Error-checks basic-file type.

    :param file_type_string: File type.
    :raises: ValueError: if `file_type_string not in BASIC_FILE_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(file_type_string)

    if file_type_string not in BASIC_FILE_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nBasic file types (listed above) do not include "{1:s}".'
        ).format(str(BASIC_FILE_TYPE_STRINGS), file_type_string)

        raise ValueError(error_string)


def _check_aggregated_file_type(file_type_string):
    """Error-checks aggregated-file type.

    :param file_type_string: File type.
    :raises: ValueError: if
        `file_type_string not in AGGREGATED_FILE_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(file_type_string)

    if file_type_string not in AGGREGATED_FILE_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nAggregated file types (listed above) do not include '
            '"{1:s}".'
        ).format(str(AGGREGATED_FILE_TYPE_STRINGS), file_type_string)

        raise ValueError(error_string)


def _check_property(property_name):
    """Error-checks name of front property.

    :param property_name: Name of property.
    :raises: ValueError: if `property_name not in VALID_PROPERTY_NAMES`.
    """

    error_checking.assert_is_string(property_name)

    if property_name not in VALID_PROPERTY_NAMES:
        error_string = (
            '\n\n{0:s}\nValid properties (listed above) do not include "{1:s}".'
        ).format(str(VALID_PROPERTY_NAMES), property_name)

        raise ValueError(error_string)


def _check_aggregated_file_metadata(first_time_unix_sec, last_time_unix_sec,
                                    prediction_file_names, hours, months):
    """Error-checks metadata for aggregated file.

    An "aggregated file" is one with front counts or statistics.

    :param first_time_unix_sec: First time in period.
    :param last_time_unix_sec: Last time in period.
    :param prediction_file_names: 1-D list of paths to input files for counts or
        statistics (readable by `prediction_io.read_file`).
    :param hours: 1-D numpy array of hours used for counts or statistics.  If
        all hours were used, leave this as None.
    :param months: Same but for months.
    :return: hours: Same as input, except that None is changed to [-1].
    :return: months: Same as input, except that None is changed to [-1].
    """

    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(prediction_file_names), num_dimensions=1
    )

    if hours is None:
        hours = numpy.array([-1], dtype=int)
    else:
        check_hours(hours)

    if months is None:
        months = numpy.array([-1], dtype=int)
    else:
        check_months(months)

    return hours, months


def _read_hours_and_months_from_agg_file(dataset_object):
    """Reads hours and months from aggregated file.

    An "aggregated file" is one with front counts or statistics.

    :param dataset_object: File handle (instance of `netCDF4.Dataset`).
    :return: hours: 1-D numpy array of hours used for counts or statistics.
        `None` means that all hours were used.
    :return: months: Same but for months.
    """

    hours = getattr(dataset_object, HOURS_KEY)
    if isinstance(hours, numpy.ndarray):
        hours = hours.astype(int)
    else:
        hours = numpy.array([hours], dtype=int)

    if len(hours) == 1 and hours[0] == -1:
        hours = None

    months = getattr(dataset_object, MONTHS_KEY)
    if isinstance(months, numpy.ndarray):
        months = months.astype(int)
    else:
        months = numpy.array([months], dtype=int)

    if len(months) == 1 and months[0] == -1:
        months = None

    return hours, months


def _compare_hour_sets(first_hours, second_hours):
    """Ensures that two sets of hours are equal.

    :param first_hours: 1-D numpy array of hours (integers in range 0...23).
        May also be None.
    :param second_hours: Same.
    """

    if first_hours is None and second_hours is None:
        return

    check_hours(first_hours)
    assert numpy.array_equal(first_hours, second_hours)


def _compare_month_sets(first_months, second_months):
    """Ensures that two sets of months are equal.

    :param first_months: 1-D numpy array of months (integers in range 1...12).
        May also be None.
    :param second_months: Same.
    """

    if first_months is None and second_months is None:
        return

    check_months(first_months)
    assert numpy.array_equal(first_months, second_months)


def _exact_times_to_hours(exact_times_unix_sec):
    """Converts exact times to hours.

    N = number of times

    :param exact_times_unix_sec: length-N numpy array of exact times.
    :return: hours: length-N numpy array of hours (in range 0...23).
    """

    error_checking.assert_is_integer_numpy_array(exact_times_unix_sec)
    error_checking.assert_is_numpy_array(exact_times_unix_sec, num_dimensions=1)

    hours = [
        int(time_conversion.unix_sec_to_string(t, '%H'))
        for t in exact_times_unix_sec
    ]

    return numpy.array(hours, dtype=int)


def _exact_times_to_months(exact_times_unix_sec):
    """Converts exact times to months.

    N = number of times

    :param exact_times_unix_sec: length-N numpy array of exact times.
    :return: months: length-N numpy array of months (in range 1...12).
    """

    error_checking.assert_is_integer_numpy_array(exact_times_unix_sec)
    error_checking.assert_is_numpy_array(exact_times_unix_sec, num_dimensions=1)

    months = [
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in exact_times_unix_sec
    ]

    return numpy.array(months, dtype=int)


def _apply_sep_time_one_front_type(
        front_type_enums, valid_times_unix_sec, separation_time_sec,
        relevant_front_type_enum):
    """Applies separation time for one front type (warm or cold).

    :param front_type_enums: Same as input arg for `apply_separation_time`,
        except this method assumes the array is sorted by increasing time.
    :param valid_times_unix_sec: Same as input arg for `apply_separation_time`,
        except this method assumes the array is sorted by increasing time.
    :param separation_time_sec: See doc for `apply_separation_time`.
    :param relevant_front_type_enum: Will apply separation time only for this
        front type (must be in `front_utils.VALID_FRONT_TYPE_ENUMS`).
    :return: front_type_enums: Same as input, except that some labels of
        `relevant_front_type_enum` may have been changed to "no front".
    """

    time_steps_sec = numpy.diff(valid_times_unix_sec)
    smallest_time_step_sec = numpy.min(time_steps_sec)

    while True:
        orig_front_type_enums = front_type_enums + 0

        frontal_flags = front_type_enums == relevant_front_type_enum

        prev_non_frontal_flags = numpy.logical_or(
            time_steps_sec > smallest_time_step_sec,
            front_type_enums[:-1] != relevant_front_type_enum
        )

        prev_non_frontal_flags = numpy.concatenate((
            numpy.array([1], dtype=bool),
            prev_non_frontal_flags
        ))

        front_start_indices = numpy.where(
            numpy.logical_and(frontal_flags, prev_non_frontal_flags)
        )[0]

        for k in front_start_indices:
            if front_type_enums[k] != relevant_front_type_enum:
                continue

            these_indices = numpy.where(numpy.logical_and(
                front_type_enums == relevant_front_type_enum,
                valid_times_unix_sec <
                valid_times_unix_sec[k] + separation_time_sec
            ))[0]

            these_indices = these_indices[these_indices > k]
            if len(these_indices) == 0:
                continue

            front_type_enums[these_indices] = front_utils.NO_FRONT_ENUM

        if numpy.array_equal(front_type_enums, orig_front_type_enums):
            break

    return front_type_enums


def check_hours(hours):
    """Error-checks list of hours.

    :param hours: 1-D numpy array of hours (in range 0...23).
    """

    error_checking.assert_is_numpy_array(hours, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(hours)
    error_checking.assert_is_geq_numpy_array(hours, 0)
    error_checking.assert_is_leq_numpy_array(hours, 23)


def check_months(months):
    """Error-checks list of months.

    :param months: 1-D numpy array of months (in range 1...12).
    """

    error_checking.assert_is_numpy_array(months, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(months)
    error_checking.assert_is_geq_numpy_array(months, 1)
    error_checking.assert_is_leq_numpy_array(months, 12)


def season_to_months(season_string):
    """Returns months in season.

    :param season_string: Season (must be accepted by `_check_season`).
    :return: months: 1-D numpy array of months (in range 1...12).
    """

    _check_season(season_string)

    if season_string == WINTER_STRING:
        return numpy.array([12, 1, 2], dtype=int)
    if season_string == SPRING_STRING:
        return numpy.array([3, 4, 5], dtype=int)
    if season_string == SUMMER_STRING:
        return numpy.array([6, 7, 8], dtype=int)

    return numpy.array([9, 10, 11], dtype=int)


def filter_by_hour(all_times_unix_sec, hours_to_keep):
    """Filters times by hour.

    :param all_times_unix_sec: 1-D numpy array of times.
    :param hours_to_keep: 1-D numpy array of hours to keep (integers in 0...23).
    :return: indices_to_keep: 1-D numpy array of indices to keep.
        If `k in indices_to_keep`, then unix_times_sec[k] has one of the desired
        hours.
    """

    check_hours(hours_to_keep)
    all_hours = _exact_times_to_hours(all_times_unix_sec)

    num_times = len(all_times_unix_sec)
    keep_time_flags = numpy.full(num_times, False, dtype=bool)

    for this_hour in hours_to_keep:
        keep_time_flags = numpy.logical_or(
            keep_time_flags, all_hours == this_hour
        )

    return numpy.where(keep_time_flags)[0]


def filter_by_month(all_times_unix_sec, months_to_keep):
    """Filters times by month.

    :param all_times_unix_sec: 1-D numpy array of times.
    :param months_to_keep: 1-D numpy array of months to keep (integers in
        1...12).
    :return: indices_to_keep: 1-D numpy array of indices to keep.
        If `k in indices_to_keep`, then unix_times_sec[k] has one of the desired
        months.
    """

    check_months(months_to_keep)
    all_months = _exact_times_to_months(all_times_unix_sec)

    num_times = len(all_times_unix_sec)
    keep_time_flags = numpy.full(num_times, False, dtype=bool)

    for this_month in months_to_keep:
        keep_time_flags = numpy.logical_or(
            keep_time_flags, all_months == this_month
        )

    return numpy.where(keep_time_flags)[0]


def months_to_string(months):
    """Converts list of months to two strings.

    For example, if the months are {1, 2, 12}, this method will return the
    following strings.

    - Verbose string: "Jan, Feb, Dec"
    - Abbrev: "jfd"

    :param months: 1-D numpy array of months (integers in 1...12).
    :return: verbose_string: See general discussion above.
    :return: abbrev_string: See general discussion above.
    """

    check_months(months)

    dummy_time_strings = [
        '4055-{0:02d}'.format(m) for m in numpy.sort(months)
    ]

    dummy_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(s, '%Y-%m')
        for s in dummy_time_strings
    ], dtype=int)

    month_strings = [
        time_conversion.unix_sec_to_string(t, '%b')
        for t in dummy_times_unix_sec
    ]

    verbose_string = ', '.join(month_strings)
    abbrev_string = ''.join([s[0].lower() for s in month_strings])
    return verbose_string, abbrev_string


def hours_to_string(hours):
    """Converts list of hours to two strings.

    For example, if the hours are {12, 13, 14, 15, 16, 17}, this method will
    return the following strings.

    - Verbose string: "12, 13, 14, 15, 16, 17 UTC"
    - Abbrev: "12-13-14-15-16-17utc"

    :param hours: 1-D numpy array of hours (in range 0...23).
    :return: verbose_string: See general discussion above.
    :return: abbrev_string: See general discussion above.
    """

    check_hours(hours)

    dummy_time_strings = [
        '4055-01-01-{0:02d}'.format(h) for h in numpy.sort(hours)
    ]

    dummy_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(s, '%Y-%m-%d-%H')
        for s in dummy_time_strings
    ], dtype=int)

    hour_strings = [
        time_conversion.unix_sec_to_string(t, '%H')
        for t in dummy_times_unix_sec
    ]

    verbose_string = '{0:s} UTC'.format(', '.join(hour_strings))
    abbrev_string = '{0:s}utc'.format('-'.join(hour_strings))
    return verbose_string, abbrev_string


def apply_separation_time(front_type_enums, valid_times_unix_sec,
                          separation_time_sec):
    """Applies separation time to front labels.

    T = number of time steps

    :param front_type_enums: length-T numpy array of front labels (must be in
        `front_utils.VALID_FRONT_TYPE_ENUMS`).
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param separation_time_sec: Separation time.
    :return: front_type_enums: Same as input but with two exceptions.
        [1] Some labels may have been changed to "no front".
        [2] Sorted by increasing time.

    :return: valid_times_unix_sec: Same as input but sorted by increasing time.
    """

    error_checking.assert_is_integer_numpy_array(front_type_enums)
    error_checking.assert_is_numpy_array(front_type_enums, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(
        front_type_enums, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS)
    )
    error_checking.assert_is_leq_numpy_array(
        front_type_enums, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS)
    )

    num_times = len(front_type_enums)
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec,
        exact_dimensions=numpy.array([num_times], dtype=int)
    )

    error_checking.assert_is_integer(separation_time_sec)
    error_checking.assert_is_greater(separation_time_sec, 0)

    sort_indices = numpy.argsort(valid_times_unix_sec)
    valid_times_unix_sec = valid_times_unix_sec[sort_indices]
    front_type_enums = front_type_enums[sort_indices]

    front_type_enums = _apply_sep_time_one_front_type(
        front_type_enums=front_type_enums,
        valid_times_unix_sec=valid_times_unix_sec,
        separation_time_sec=separation_time_sec,
        relevant_front_type_enum=front_utils.WARM_FRONT_ENUM)

    front_type_enums = _apply_sep_time_one_front_type(
        front_type_enums=front_type_enums,
        valid_times_unix_sec=valid_times_unix_sec,
        separation_time_sec=separation_time_sec,
        relevant_front_type_enum=front_utils.COLD_FRONT_ENUM)

    return front_type_enums, valid_times_unix_sec


def find_basic_file(directory_name, file_type_string, valid_time_unix_sec,
                    raise_error_if_missing=True):
    """Locates file with gridded front labels or properties.

    :param directory_name: Directory name.
    :param file_type_string: See doc for `_check_basic_file_type`.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: netcdf_file_name: File path.  If file is missing and
        `raise_error_if_missing = False`, this is the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    _check_basic_file_type(file_type_string)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    netcdf_file_name = '{0:s}/{1:s}/{2:s}_{3:s}.nc'.format(
        directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, '%Y%m'),
        file_type_string.replace('_', '-'),
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, FILE_NAME_TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isfile(netcdf_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            netcdf_file_name)
        raise ValueError(error_string)

    return netcdf_file_name


def basic_file_name_to_time(netcdf_file_name):
    """Parses time from file name.

    :param netcdf_file_name: See doc for `find_basic_file`.
    :return: valid_time_unix_sec: Valid time.
    """

    error_checking.assert_is_string(netcdf_file_name)

    pathless_file_name = os.path.split(netcdf_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    return time_conversion.string_to_unix_sec(
        extensionless_file_name.split('_')[-1], FILE_NAME_TIME_FORMAT
    )


def find_many_basic_files(
        directory_name, file_type_string, first_time_unix_sec,
        last_time_unix_sec, hours_to_keep=None, months_to_keep=None,
        raise_error_if_none_found=True):
    """Finds many files with gridded front labels or properties.

    :param directory_name: Directory name.
    :param file_type_string: See doc for `_check_basic_file_type`.
    :param first_time_unix_sec: First time.  Will look for files only in period
        `first_time_unix_sec`...`last_time_unix_sec`.
    :param last_time_unix_sec: See above.
    :param hours_to_keep: 1-D numpy array of integers in range 0...23.  Will
        look for files only at these hours.  If None, will look for files at all
        hours.
    :param months_to_keep: 1-D numpy array of integers in range 1...12.  Will
        look for files only in these months.  If None, will look for files in
        all months.
    :param raise_error_if_none_found: Boolean flag.  If all files are missing
        and `raise_error_if_none_found = True`, this method will error out.
    :return: netcdf_file_names: 1-D list of file paths.  If no files were found
        and `raise_error_if_none_found = False`, this is an empty list.
    :raises: ValueError: if no files were found and
        `raise_error_if_none_found = True`.
    """

    error_checking.assert_is_string(directory_name)
    _check_basic_file_type(file_type_string)
    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_none_found)

    if hours_to_keep is not None:
        check_hours(hours_to_keep)
    if months_to_keep is not None:
        check_months(months_to_keep)

    glob_pattern = '{0:s}/{1:s}/{2:s}_{3:s}.nc'.format(
        directory_name, YEAR_MONTH_REGEX,
        file_type_string.replace('_', '-'), FILE_NAME_TIME_REGEX
    )

    netcdf_file_names = glob.glob(glob_pattern)

    if len(netcdf_file_names) > 0:
        valid_times_unix_sec = numpy.array(
            [basic_file_name_to_time(f) for f in netcdf_file_names], dtype=int
        )

        good_indices = numpy.where(numpy.logical_and(
            valid_times_unix_sec >= first_time_unix_sec,
            valid_times_unix_sec <= last_time_unix_sec
        ))[0]

        netcdf_file_names = [netcdf_file_names[k] for k in good_indices]
        valid_times_unix_sec = valid_times_unix_sec[good_indices]

    if hours_to_keep is not None and len(netcdf_file_names) > 0:
        good_indices = filter_by_hour(
            all_times_unix_sec=valid_times_unix_sec,
            hours_to_keep=hours_to_keep)

        netcdf_file_names = [netcdf_file_names[k] for k in good_indices]
        valid_times_unix_sec = valid_times_unix_sec[good_indices]

    if months_to_keep is not None and len(netcdf_file_names) > 0:
        good_indices = filter_by_month(
            all_times_unix_sec=valid_times_unix_sec,
            months_to_keep=months_to_keep)

        netcdf_file_names = [netcdf_file_names[k] for k in good_indices]
        # valid_times_unix_sec = valid_times_unix_sec[good_indices]

    if raise_error_if_none_found and len(netcdf_file_names) == 0:
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            glob_pattern)
        raise ValueError(error_string)

    if len(netcdf_file_names) > 0:
        netcdf_file_names.sort()

    return netcdf_file_names


def average_many_property_files(property_file_names):
    """Averages gridded front properties over many files.

    This method averages each property, at each grid cell, independently.

    :param property_file_names: 1-D list of paths to input files (will be read
        by `read_gridded_properties`).
    :return: front_statistic_dict: Dictionary with the following keys.
    front_property_dict["mean_wf_length_matrix_metres"]: See doc for
        `write_gridded_stats`.
    front_property_dict["mean_wf_area_matrix_m2"]: Same.
    front_property_dict["mean_cf_length_matrix_metres"]: Same.
    front_property_dict["mean_cf_area_matrix_m2"]: Same.
    front_property_dict["prediction_file_names"]: Same.
    """

    error_checking.assert_is_string_list(property_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(property_file_names), num_dimensions=1
    )

    num_wf_labels_matrix = None
    sum_wf_length_matrix_metres = None
    sum_wf_area_matrix_m2 = None
    num_cf_labels_matrix = None
    sum_cf_length_matrix_metres = None
    sum_cf_area_matrix_m2 = None

    prediction_file_names = []

    for i in range(len(property_file_names)):
        print('Reading data from: "{0:s}"...'.format(property_file_names[i]))
        this_property_dict = read_gridded_properties(property_file_names[i])
        prediction_file_names.append(this_property_dict[PREDICTION_FILE_KEY])

        this_num_labels_matrix = numpy.invert(numpy.isnan(
            this_property_dict[WARM_FRONT_LENGTHS_KEY]
        )).astype(int)

        if i == 0:
            num_wf_labels_matrix = numpy.full(this_num_labels_matrix.shape, 0.)
            sum_wf_length_matrix_metres = numpy.full(
                this_num_labels_matrix.shape, 0.)
            sum_wf_area_matrix_m2 = numpy.full(this_num_labels_matrix.shape, 0.)

            num_cf_labels_matrix = numpy.full(this_num_labels_matrix.shape, 0.)
            sum_cf_length_matrix_metres = numpy.full(
                this_num_labels_matrix.shape, 0.)
            sum_cf_area_matrix_m2 = numpy.full(this_num_labels_matrix.shape, 0.)

        num_wf_labels_matrix = num_wf_labels_matrix + this_num_labels_matrix
        sum_wf_length_matrix_metres += numpy.nan_to_num(
            this_property_dict[WARM_FRONT_LENGTHS_KEY], nan=0
        )
        sum_wf_area_matrix_m2 += numpy.nan_to_num(
            this_property_dict[WARM_FRONT_AREAS_KEY], nan=0
        )

        this_num_labels_matrix = numpy.invert(numpy.isnan(
            this_property_dict[COLD_FRONT_LENGTHS_KEY]
        )).astype(int)

        num_cf_labels_matrix = num_cf_labels_matrix + this_num_labels_matrix
        sum_cf_length_matrix_metres += numpy.nan_to_num(
            this_property_dict[COLD_FRONT_LENGTHS_KEY], nan=0
        )
        sum_cf_area_matrix_m2 += numpy.nan_to_num(
            this_property_dict[COLD_FRONT_AREAS_KEY], nan=0
        )

    num_wf_labels_matrix[num_wf_labels_matrix == 0] = numpy.nan
    num_cf_labels_matrix[num_cf_labels_matrix == 0] = numpy.nan

    return {
        MEAN_WF_LENGTHS_KEY: sum_wf_length_matrix_metres / num_wf_labels_matrix,
        MEAN_WF_AREAS_KEY: sum_wf_area_matrix_m2 / num_wf_labels_matrix,
        MEAN_CF_LENGTHS_KEY: sum_cf_length_matrix_metres / num_cf_labels_matrix,
        MEAN_CF_AREAS_KEY: sum_cf_area_matrix_m2 / num_cf_labels_matrix,
        PREDICTION_FILES_KEY: prediction_file_names
    }


def write_gridded_labels(
        netcdf_file_name, label_matrix, unique_label_matrix,
        prediction_file_name, separation_time_sec):
    """Writes gridded front labels to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param label_matrix: M-by-N numpy array of front labels (integers).  NaN
        means that there is no reanalysis (predictor) data at the grid cell, so
        it is impossible to tell.
    :param unique_label_matrix: Same but after applying separation time.
    :param prediction_file_name: Path to input file (readable by
        `prediction_io.read_file`).  This is metadata.
    :param separation_time_sec: Separation time (for more details, see doc for
        `apply_separation_time`).  This is metadata.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(
        label_matrix, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS),
        allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        label_matrix, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS),
        allow_nan=True
    )
    error_checking.assert_is_numpy_array(label_matrix, num_dimensions=2)

    error_checking.assert_is_geq_numpy_array(
        unique_label_matrix, numpy.min(front_utils.VALID_FRONT_TYPE_ENUMS),
        allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        unique_label_matrix, numpy.max(front_utils.VALID_FRONT_TYPE_ENUMS),
        allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        unique_label_matrix,
        exact_dimensions=numpy.array(label_matrix.shape, dtype=int)
    )

    error_checking.assert_is_string(prediction_file_name)
    error_checking.assert_is_integer(separation_time_sec)
    error_checking.assert_is_greater(separation_time_sec, 0)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes and dimensions.
    dataset_object.setncattr(PREDICTION_FILE_KEY, str(prediction_file_name))
    dataset_object.setncattr(SEPARATION_TIME_KEY, separation_time_sec)

    dataset_object.createDimension(ROW_DIMENSION_KEY, label_matrix.shape[0])
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, label_matrix.shape[1])

    # Add variables.
    dataset_object.createVariable(
        FRONT_LABELS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[FRONT_LABELS_KEY][:] = label_matrix

    dataset_object.createVariable(
        UNIQUE_FRONT_LABELS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[UNIQUE_FRONT_LABELS_KEY][:] = unique_label_matrix

    dataset_object.close()


def read_gridded_labels(netcdf_file_name):
    """Reads gridded front labels from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: front_label_dict: Dictionary with the following keys.
    front_label_dict["label_matrix"]: See doc for `write_gridded_labels`.
    front_label_dict["unique_label_matrix"]: Same.
    front_label_dict["prediction_file_name"]: Same.
    front_label_dict["separation_time_sec"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    front_label_dict = {
        FRONT_LABELS_KEY: numpy.array(
            dataset_object.variables[FRONT_LABELS_KEY][:]
        ),
        UNIQUE_FRONT_LABELS_KEY: numpy.array(
            dataset_object.variables[UNIQUE_FRONT_LABELS_KEY][:]
        ),
        PREDICTION_FILE_KEY: str(getattr(dataset_object, PREDICTION_FILE_KEY)),
        SEPARATION_TIME_KEY: int(getattr(dataset_object, SEPARATION_TIME_KEY))
    }

    dataset_object.close()

    # for this_key in [FRONT_LABELS_KEY, UNIQUE_FRONT_LABELS_KEY]:
    #     front_label_dict[this_key][front_label_dict[this_key] < 0] = numpy.nan

    return front_label_dict


def write_gridded_properties(
        netcdf_file_name, wf_length_matrix_metres, wf_area_matrix_m2,
        cf_length_matrix_metres, cf_area_matrix_m2, prediction_file_name):
    """Writes gridded front properties to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param wf_length_matrix_metres: M-by-N numpy array with warm-front length at
        each grid cell (NaN if there is no warm front).
    :param wf_area_matrix_m2: Same but for warm-front area.
    :param cf_length_matrix_metres: Same but for cold-front length.
    :param cf_area_matrix_m2: Same but for cold-front area.
    :param prediction_file_name: Path to input file (readable by
        `prediction_io.read_file`).  This is metadata.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(
        wf_length_matrix_metres, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        wf_length_matrix_metres, num_dimensions=2)

    these_expected_dim = numpy.array(wf_length_matrix_metres.shape, dtype=int)

    error_checking.assert_is_geq_numpy_array(
        wf_area_matrix_m2, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        wf_area_matrix_m2, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        cf_length_matrix_metres, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        cf_length_matrix_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        cf_area_matrix_m2, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        cf_area_matrix_m2, exact_dimensions=these_expected_dim)

    error_checking.assert_is_string(prediction_file_name)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes and dimensions.
    dataset_object.setncattr(PREDICTION_FILE_KEY, str(prediction_file_name))

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, wf_length_matrix_metres.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, wf_length_matrix_metres.shape[1]
    )

    # Add variables.
    dataset_object.createVariable(
        WARM_FRONT_LENGTHS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[WARM_FRONT_LENGTHS_KEY][:] = (
        wf_length_matrix_metres
    )

    dataset_object.createVariable(
        WARM_FRONT_AREAS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[WARM_FRONT_AREAS_KEY][:] = wf_area_matrix_m2

    dataset_object.createVariable(
        COLD_FRONT_LENGTHS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[COLD_FRONT_LENGTHS_KEY][:] = (
        cf_length_matrix_metres
    )

    dataset_object.createVariable(
        COLD_FRONT_AREAS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[COLD_FRONT_AREAS_KEY][:] = cf_area_matrix_m2

    dataset_object.close()


def read_gridded_properties(netcdf_file_name):
    """Reads gridded front properties from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: front_property_dict: Dictionary with the following keys.
    front_property_dict["wf_length_matrix_metres"]: See doc for
        `write_gridded_properties`.
    front_property_dict["wf_area_matrix_m2"]: Same.
    front_property_dict["cf_length_matrix_metres"]: Same.
    front_property_dict["cf_area_matrix_m2"]: Same.
    front_property_dict["prediction_file_name"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    front_property_dict = {
        WARM_FRONT_LENGTHS_KEY: numpy.array(
            dataset_object.variables[WARM_FRONT_LENGTHS_KEY][:], dtype=float
        ),
        WARM_FRONT_AREAS_KEY: numpy.array(
            dataset_object.variables[WARM_FRONT_AREAS_KEY][:], dtype=float
        ),
        COLD_FRONT_LENGTHS_KEY: numpy.array(
            dataset_object.variables[COLD_FRONT_LENGTHS_KEY][:], dtype=float
        ),
        COLD_FRONT_AREAS_KEY: numpy.array(
            dataset_object.variables[COLD_FRONT_AREAS_KEY][:], dtype=float
        ),
        PREDICTION_FILE_KEY: str(getattr(dataset_object, PREDICTION_FILE_KEY))
    }

    dataset_object.close()

    # for this_key in [WARM_FRONT_LENGTHS_KEY, WARM_FRONT_AREAS_KEY,
    #                  COLD_FRONT_LENGTHS_KEY, COLD_FRONT_AREAS_KEY]:
    #     front_property_dict[this_key][
    #         front_property_dict[this_key] < 0
    #     ] = numpy.nan

    return front_property_dict


def find_monte_carlo_file(
        directory_name, property_name, first_grid_row, first_grid_column,
        raise_error_if_missing=True):
    """Finds NetCDF file with results of Monte Carlo test.

    :param directory_name: See doc for `write_monte_carlo_test`.
    :param property_name: Same.
    :param first_grid_row: Same.
    :param first_grid_column: Same.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: netcdf_file_name: Path to file with Monte Carlo results.  If file
        is missing and `raise_error_if_missing = False`, this is the *expected*
        path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    _check_property(property_name)
    error_checking.assert_is_integer(first_grid_row)
    error_checking.assert_is_geq(first_grid_row, 0)
    error_checking.assert_is_integer(first_grid_column)
    error_checking.assert_is_geq(first_grid_column, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    netcdf_file_name = (
        '{0:s}/monte-carlo-test_{1:s}_first-row={2:03d}_first-column={3:03d}.nc'
    ).format(
        directory_name, property_name.replace('_', '-'),
        first_grid_row, first_grid_column
    )

    if raise_error_if_missing and not os.path.isfile(netcdf_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            netcdf_file_name)
        raise ValueError(error_string)

    return netcdf_file_name


def find_many_monte_carlo_files(
        directory_name, property_name, raise_error_if_none_found=True):
    """Finds many NetCDF files with results of Monte Carlo test.

    :param directory_name: See doc for `write_monte_carlo_test`.
    :param property_name: Same.
    :param raise_error_if_none_found: Boolean flag.  If all files are missing
        and `raise_error_if_none_found = True`, this method will error out.
    :return: netcdf_file_names: 1-D list of file paths.  If no files were found
        and `raise_error_if_none_found = False`, this is an empty list.
    :raises: ValueError: if no files were found and
        `raise_error_if_none_found = True`.
    """

    error_checking.assert_is_string(directory_name)
    _check_property(property_name)
    error_checking.assert_is_boolean(raise_error_if_none_found)

    glob_pattern = (
        '{0:s}/monte-carlo-test_{1:s}_first-row=[0-9][0-9][0-9]_'
        'first-column=[0-9][0-9][0-9].nc'
    ).format(
        directory_name, property_name.replace('_', '-'),
    )

    netcdf_file_names = glob.glob(glob_pattern)

    if raise_error_if_none_found and len(netcdf_file_names) == 0:
        error_string = 'Could not find any files with pattern: "{0:s}"'.format(
            glob_pattern)
        raise ValueError(error_string)

    if len(netcdf_file_names) > 0:
        netcdf_file_names.sort()

    return netcdf_file_names


def write_monte_carlo_test(
        netcdf_file_name, baseline_mean_matrix, trial_mean_matrix,
        significance_matrix, property_name, baseline_input_file_names,
        trial_input_file_names, num_iterations, confidence_level,
        first_grid_row, first_grid_column):
    """Writes results of Monte Carlo test to NetCDF file.

    M = number of rows in subgrid (the part of the grid sent to this method)
    N = number of rows in subgrid

    :param netcdf_file_name: Path to output file.
    :param baseline_mean_matrix: M-by-N numpy array with mean values in baseline
        set.
    :param trial_mean_matrix: M-by-N numpy array with mean values in trial set.
    :param significance_matrix: M-by-N numpy array of Boolean flags,
        indicating where difference between means is significant.
    :param property_name: Name of property.  Must be accepted by
        `_check_property`.
    :param baseline_input_file_names: 1-D list of paths to input files for first
        composite (readable by `read_gridded_properties`).
    :param trial_input_file_names: Same but for second composite.
    :param num_iterations: Number of iterations for Monte Carlo test.
    :param confidence_level: Confidence level for Monte Carlo test (used to
        determine where difference between means is significant).
    :param first_grid_row: First row in subgrid is [i]th row in full grid, where
        i = `first_grid_row`.
    :param first_grid_column: Same but for first column in subgrid.
    """

    error_checking.assert_is_numpy_array(baseline_mean_matrix, num_dimensions=2)

    expected_dim = numpy.array(baseline_mean_matrix.shape, dtype=int)
    error_checking.assert_is_numpy_array(
        trial_mean_matrix, exact_dimensions=expected_dim)

    error_checking.assert_is_boolean_numpy_array(significance_matrix)
    error_checking.assert_is_numpy_array(
        significance_matrix, exact_dimensions=expected_dim)

    _check_property(property_name)

    error_checking.assert_is_string_list(baseline_input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(baseline_input_file_names), num_dimensions=1
    )

    error_checking.assert_is_string_list(trial_input_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(trial_input_file_names), num_dimensions=1
    )

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_geq(num_iterations, 10)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)
    error_checking.assert_is_integer(first_grid_row)
    error_checking.assert_is_geq(first_grid_row, 0)
    error_checking.assert_is_integer(first_grid_column)
    error_checking.assert_is_geq(first_grid_column, 0)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes and dimensions.
    dataset_object.setncattr(PROPERTY_NAME_KEY, str(property_name))
    dataset_object.setncattr(NUM_ITERATIONS_KEY, num_iterations)
    dataset_object.setncattr(CONFIDENCE_LEVEL_KEY, confidence_level)
    dataset_object.setncattr(FIRST_GRID_ROW_KEY, first_grid_row)
    dataset_object.setncattr(FIRST_GRID_COLUMN_KEY, first_grid_column)

    num_file_name_chars = max([
        len(f) for f in baseline_input_file_names + trial_input_file_names
    ])

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, baseline_mean_matrix.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, baseline_mean_matrix.shape[1]
    )
    dataset_object.createDimension(
        BASELINE_INPUT_FILE_DIM_KEY, len(baseline_input_file_names)
    )
    dataset_object.createDimension(
        TRIAL_INPUT_FILE_DIM_KEY, len(trial_input_file_names)
    )
    dataset_object.createDimension(
        INPUT_FILE_CHAR_DIM_KEY, num_file_name_chars
    )

    dataset_object.createVariable(
        BASELINE_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[BASELINE_MATRIX_KEY][:] = baseline_mean_matrix

    dataset_object.createVariable(
        TRIAL_MATRIX_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[TRIAL_MATRIX_KEY][:] = trial_mean_matrix

    dataset_object.createVariable(
        SIGNIFICANCE_MATRIX_KEY, datatype=numpy.int32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[SIGNIFICANCE_MATRIX_KEY][:] = (
        significance_matrix.astype(int)
    )

    this_string_type = 'S{0:d}'.format(num_file_name_chars)
    this_char_array = netCDF4.stringtochar(numpy.array(
        baseline_input_file_names, dtype=this_string_type
    ))

    dataset_object.createVariable(
        BASELINE_INPUT_FILES_KEY, datatype='S1',
        dimensions=(BASELINE_INPUT_FILE_DIM_KEY, INPUT_FILE_CHAR_DIM_KEY)
    )
    dataset_object.variables[BASELINE_INPUT_FILES_KEY][:] = numpy.array(
        this_char_array)

    this_char_array = netCDF4.stringtochar(numpy.array(
        trial_input_file_names, dtype=this_string_type
    ))

    dataset_object.createVariable(
        TRIAL_INPUT_FILES_KEY, datatype='S1',
        dimensions=(TRIAL_INPUT_FILE_DIM_KEY, INPUT_FILE_CHAR_DIM_KEY)
    )
    dataset_object.variables[TRIAL_INPUT_FILES_KEY][:] = numpy.array(
        this_char_array)

    dataset_object.close()


def read_monte_carlo_test(netcdf_file_name):
    """Reads results of Monte Carlo test from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: monte_carlo_dict: Dictionary with the following keys.
    monte_carlo_dict["baseline_mean_matrix"]: See doc for
        `write_monte_carlo_test`.
    monte_carlo_dict["trial_mean_matrix"]: Same.
    monte_carlo_dict["significance_matrix"]: Same.
    monte_carlo_dict["property_name"]: Same.
    monte_carlo_dict["baseline_input_file_names"]: Same.
    monte_carlo_dict["trial_input_file_names"]: Same.
    monte_carlo_dict["num_iterations"]: Same.
    monte_carlo_dict["confidence_level"]: Same.
    monte_carlo_dict["first_grid_row"]: Same.
    monte_carlo_dict["first_grid_column"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    monte_carlo_dict = {
        PROPERTY_NAME_KEY: str(getattr(dataset_object, PROPERTY_NAME_KEY)),
        NUM_ITERATIONS_KEY: int(getattr(dataset_object, NUM_ITERATIONS_KEY)),
        CONFIDENCE_LEVEL_KEY: getattr(dataset_object, CONFIDENCE_LEVEL_KEY),
        FIRST_GRID_ROW_KEY: int(getattr(dataset_object, FIRST_GRID_ROW_KEY)),
        FIRST_GRID_COLUMN_KEY:
            int(getattr(dataset_object, FIRST_GRID_COLUMN_KEY)),
        BASELINE_MATRIX_KEY: numpy.array(
            dataset_object.variables[BASELINE_MATRIX_KEY][:], dtype=float
        ),
        TRIAL_MATRIX_KEY: numpy.array(
            dataset_object.variables[TRIAL_MATRIX_KEY][:], dtype=float
        ),
        SIGNIFICANCE_MATRIX_KEY: numpy.array(
            dataset_object.variables[SIGNIFICANCE_MATRIX_KEY][:], dtype=bool
        ),
        BASELINE_INPUT_FILES_KEY: [
            str(s) for s in
            netCDF4.chartostring(
                dataset_object.variables[BASELINE_INPUT_FILES_KEY][:]
            )
        ],
        TRIAL_INPUT_FILES_KEY: [
            str(s) for s in
            netCDF4.chartostring(
                dataset_object.variables[TRIAL_INPUT_FILES_KEY][:]
            )
        ]
    }

    dataset_object.close()
    return monte_carlo_dict


def find_aggregated_file(
        directory_name, file_type_string, first_time_unix_sec,
        last_time_unix_sec, hours=None, months=None,
        raise_error_if_missing=True):
    """Locates file with gridded front statistics or counts.

    :param directory_name: Directory name.
    :param file_type_string: See doc for `_check_aggregated_file_type`.
    :param first_time_unix_sec: First time used to create stats.
    :param last_time_unix_sec: Last time used to create stats.
    :param hours: 1-D numpy array of hours for which fronts were counted.  If
        all hours were used, leave this as None.
    :param months: Same but for months.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing = True`, this method will error out.
    :return: netcdf_file_name: Path to file with gridded front statistics or
        counts.  If file is missing and `raise_error_if_missing = False`, this
        is the *expected* path.
    :raises: ValueError: if file is missing and `raise_error_if_missing = True`.
    """

    error_checking.assert_is_string(directory_name)
    _check_aggregated_file_type(file_type_string)
    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_greater(last_time_unix_sec, first_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if hours is None:
        hour_string = 'all'
    else:
        hour_string = hours_to_string(hours)[-1]

    if months is None:
        month_string = 'all'
    else:
        month_string = months_to_string(months)[-1]

    netcdf_file_name = (
        '{0:s}/{1:s}_{2:s}_{3:s}_hours={4:s}_months={5:s}.nc'
    ).format(
        directory_name, file_type_string.replace('_', '-'),
        time_conversion.unix_sec_to_string(
            first_time_unix_sec, FILE_NAME_TIME_FORMAT),
        time_conversion.unix_sec_to_string(
            last_time_unix_sec, FILE_NAME_TIME_FORMAT),
        hour_string, month_string
    )

    if raise_error_if_missing and not os.path.isfile(netcdf_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            netcdf_file_name)
        raise ValueError(error_string)

    return netcdf_file_name


def write_gridded_counts(
        netcdf_file_name, num_wf_labels_matrix, num_unique_wf_matrix,
        num_cf_labels_matrix, num_unique_cf_matrix,
        first_time_unix_sec, last_time_unix_sec, prediction_file_names,
        hours=None, months=None):
    """Writes gridded front counts to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param num_wf_labels_matrix: M-by-N numpy array with number of warm fronts
        at each grid cell (NaN for grid cells with no reanalysis data).
    :param num_unique_wf_matrix: Same but after applying separation time.
    :param num_cf_labels_matrix: Same but for cold fronts.
    :param num_unique_cf_matrix: Same but for cold fronts after applying
        separation time.
    :param first_time_unix_sec: See doc for `_check_aggregated_file_metadata`.
    :param last_time_unix_sec: Same.
    :param prediction_file_names: Same.
    :param hours: Same.
    :param months: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(
        num_wf_labels_matrix, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        num_wf_labels_matrix, num_dimensions=2)

    these_expected_dim = numpy.array(num_wf_labels_matrix.shape, dtype=int)

    error_checking.assert_is_geq_numpy_array(
        num_unique_wf_matrix, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        num_unique_wf_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        num_cf_labels_matrix, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        num_cf_labels_matrix, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        num_unique_cf_matrix, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        num_unique_cf_matrix, exact_dimensions=these_expected_dim)

    hours, months = _check_aggregated_file_metadata(
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        prediction_file_names=prediction_file_names, hours=hours, months=months)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes and dimensions.
    dataset_object.setncattr(FIRST_TIME_KEY, first_time_unix_sec)
    dataset_object.setncattr(LAST_TIME_KEY, last_time_unix_sec)
    dataset_object.setncattr(HOURS_KEY, hours)
    dataset_object.setncattr(MONTHS_KEY, months)

    num_file_name_chars = max([
        len(f) for f in prediction_file_names
    ])

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, num_wf_labels_matrix.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, num_wf_labels_matrix.shape[1]
    )
    dataset_object.createDimension(
        PREDICTION_FILE_DIM_KEY, len(prediction_file_names)
    )
    dataset_object.createDimension(
        PREDICTION_FILE_CHAR_DIM_KEY, num_file_name_chars
    )

    # Add variables.
    dataset_object.createVariable(
        NUM_WF_LABELS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NUM_WF_LABELS_KEY][:] = num_wf_labels_matrix

    dataset_object.createVariable(
        NUM_UNIQUE_WF_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NUM_UNIQUE_WF_KEY][:] = num_unique_wf_matrix

    dataset_object.createVariable(
        NUM_CF_LABELS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NUM_CF_LABELS_KEY][:] = num_cf_labels_matrix

    dataset_object.createVariable(
        NUM_UNIQUE_CF_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[NUM_UNIQUE_CF_KEY][:] = num_unique_cf_matrix

    this_string_type = 'S{0:d}'.format(num_file_name_chars)
    file_names_char_array = netCDF4.stringtochar(numpy.array(
        prediction_file_names, dtype=this_string_type
    ))

    dataset_object.createVariable(
        PREDICTION_FILES_KEY, datatype='S1',
        dimensions=(PREDICTION_FILE_DIM_KEY, PREDICTION_FILE_CHAR_DIM_KEY)
    )
    dataset_object.variables[PREDICTION_FILES_KEY][:] = numpy.array(
        file_names_char_array)

    dataset_object.close()


def read_gridded_counts(netcdf_file_name):
    """Reads gridded front counts from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: front_count_dict: Dictionary with the following keys.
    front_count_dict["num_wf_labels_matrix"]: See doc for
        `write_gridded_counts`.
    front_count_dict["num_unique_wf_matrix"]: Same.
    front_count_dict["num_cf_labels_matrix"]: Same.
    front_count_dict["num_unique_cf_matrix"]: Same.
    front_count_dict["first_time_unix_sec"]: Same.
    front_count_dict["last_time_unix_sec"]: Same.
    front_count_dict["hours"]: Same.
    front_count_dict["months"]: Same.
    front_count_dict["prediction_file_names"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    hours, months = _read_hours_and_months_from_agg_file(dataset_object)

    front_count_dict = {
        FIRST_TIME_KEY: int(getattr(dataset_object, FIRST_TIME_KEY)),
        LAST_TIME_KEY: int(getattr(dataset_object, LAST_TIME_KEY)),
        HOURS_KEY: hours,
        MONTHS_KEY: months,
        NUM_WF_LABELS_KEY: numpy.array(
            dataset_object.variables[NUM_WF_LABELS_KEY][:], dtype=float
        ),
        NUM_UNIQUE_WF_KEY: numpy.array(
            dataset_object.variables[NUM_UNIQUE_WF_KEY][:], dtype=float
        ),
        NUM_CF_LABELS_KEY: numpy.array(
            dataset_object.variables[NUM_CF_LABELS_KEY][:], dtype=float
        ),
        NUM_UNIQUE_CF_KEY: numpy.array(
            dataset_object.variables[NUM_UNIQUE_CF_KEY][:], dtype=float
        ),
        PREDICTION_FILES_KEY: [
            str(s) for s in
            netCDF4.chartostring(
                dataset_object.variables[PREDICTION_FILES_KEY][:]
            )
        ],
    }

    dataset_object.close()
    return front_count_dict


def write_gridded_stats(
        netcdf_file_name, mean_wf_length_matrix_metres, mean_wf_area_matrix_m2,
        mean_cf_length_matrix_metres, mean_cf_area_matrix_m2,
        first_time_unix_sec, last_time_unix_sec, prediction_file_names,
        hours=None, months=None):
    """Writes stats for gridded front properties to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param mean_wf_length_matrix_metres: M-by-N numpy array with mean warm-front
        length at each grid cell (NaN if there are no warm fronts).
    :param mean_wf_area_matrix_m2: Same but for warm-front area.
    :param mean_cf_length_matrix_metres: Same but for cold-front length.
    :param mean_cf_area_matrix_m2: Same but for cold-front area.
    :param first_time_unix_sec: See doc for `_check_aggregated_file_metadata`.
    :param last_time_unix_sec: Same.
    :param prediction_file_names: Same.
    :param hours: Same.
    :param months: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(
        mean_wf_length_matrix_metres, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        mean_wf_length_matrix_metres, num_dimensions=2)

    these_expected_dim = numpy.array(
        mean_wf_length_matrix_metres.shape, dtype=int)

    error_checking.assert_is_geq_numpy_array(
        mean_wf_area_matrix_m2, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        mean_wf_area_matrix_m2, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        mean_cf_length_matrix_metres, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        mean_cf_length_matrix_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_geq_numpy_array(
        mean_cf_area_matrix_m2, 0, allow_nan=True)
    error_checking.assert_is_numpy_array(
        mean_cf_area_matrix_m2, exact_dimensions=these_expected_dim)

    hours, months = _check_aggregated_file_metadata(
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        prediction_file_names=prediction_file_names, hours=hours, months=months)

    # Open file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET')

    # Set global attributes and dimensions.
    dataset_object.setncattr(FIRST_TIME_KEY, first_time_unix_sec)
    dataset_object.setncattr(LAST_TIME_KEY, last_time_unix_sec)
    dataset_object.setncattr(HOURS_KEY, hours)
    dataset_object.setncattr(MONTHS_KEY, months)

    num_file_name_chars = max([
        len(f) for f in prediction_file_names
    ])

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, mean_wf_length_matrix_metres.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, mean_wf_length_matrix_metres.shape[1]
    )
    dataset_object.createDimension(
        PREDICTION_FILE_DIM_KEY, len(prediction_file_names)
    )
    dataset_object.createDimension(
        PREDICTION_FILE_CHAR_DIM_KEY, num_file_name_chars
    )

    # Add variables.
    dataset_object.createVariable(
        MEAN_WF_LENGTHS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MEAN_WF_LENGTHS_KEY][:] = (
        mean_wf_length_matrix_metres
    )

    dataset_object.createVariable(
        MEAN_WF_AREAS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MEAN_WF_AREAS_KEY][:] = mean_wf_area_matrix_m2

    dataset_object.createVariable(
        MEAN_CF_LENGTHS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MEAN_CF_LENGTHS_KEY][:] = (
        mean_cf_length_matrix_metres
    )

    dataset_object.createVariable(
        MEAN_CF_AREAS_KEY, datatype=numpy.float32,
        dimensions=(ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[MEAN_CF_AREAS_KEY][:] = mean_cf_area_matrix_m2

    this_string_type = 'S{0:d}'.format(num_file_name_chars)
    file_names_char_array = netCDF4.stringtochar(numpy.array(
        prediction_file_names, dtype=this_string_type
    ))

    dataset_object.createVariable(
        PREDICTION_FILES_KEY, datatype='S1',
        dimensions=(PREDICTION_FILE_DIM_KEY, PREDICTION_FILE_CHAR_DIM_KEY)
    )
    dataset_object.variables[PREDICTION_FILES_KEY][:] = numpy.array(
        file_names_char_array)

    dataset_object.close()


def read_gridded_stats(netcdf_file_name):
    """Reads stats for gridded front properties from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: front_statistic_dict: Dictionary with the following keys.
    front_statistic_dict["mean_wf_length_matrix_metres"]: See doc for
        `write_gridded_stats`.
    front_statistic_dict["mean_wf_area_matrix_m2"]: Same.
    front_statistic_dict["mean_cf_length_matrix_metres"]: Same.
    front_statistic_dict["mean_cf_area_matrix_m2"]: Same.
    front_statistic_dict["first_time_unix_sec"]: Same.
    front_statistic_dict["last_time_unix_sec"]: Same.
    front_statistic_dict["hours"]: Same.
    front_statistic_dict["months"]: Same.
    front_statistic_dict["prediction_file_names"]: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    hours, months = _read_hours_and_months_from_agg_file(dataset_object)

    front_property_dict = {
        FIRST_TIME_KEY: int(getattr(dataset_object, FIRST_TIME_KEY)),
        LAST_TIME_KEY: int(getattr(dataset_object, LAST_TIME_KEY)),
        HOURS_KEY: hours,
        MONTHS_KEY: months,
        MEAN_WF_LENGTHS_KEY: numpy.array(
            dataset_object.variables[MEAN_WF_LENGTHS_KEY][:], dtype=float
        ),
        MEAN_WF_AREAS_KEY: numpy.array(
            dataset_object.variables[MEAN_WF_AREAS_KEY][:], dtype=float
        ),
        MEAN_CF_LENGTHS_KEY: numpy.array(
            dataset_object.variables[MEAN_CF_LENGTHS_KEY][:], dtype=float
        ),
        MEAN_CF_AREAS_KEY: numpy.array(
            dataset_object.variables[MEAN_CF_AREAS_KEY][:], dtype=float
        ),
        PREDICTION_FILES_KEY: [
            str(s) for s in
            netCDF4.chartostring(
                dataset_object.variables[PREDICTION_FILES_KEY][:]
            )
        ],
    }

    dataset_object.close()
    return front_property_dict
