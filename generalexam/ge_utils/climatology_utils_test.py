"""Unit tests for climatology_utils.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from generalexam.ge_utils import front_utils
from generalexam.ge_utils import climatology_utils as climo_utils

# The following constants are used to test a bunch of methods.
ALL_TIME_STRINGS = [
    '2018-01-01-000000', '2018-02-02-020202', '2018-03-03-030303',
    '2018-04-04-040404', '2018-05-05-050505', '2018-06-06-060606',
    '2018-07-07-070707', '2018-08-08-080808', '2018-09-09-090909',
    '2018-10-10-101010', '2018-11-11-111111', '2018-12-12-121212',
    '2019-12-13-131313', '2020-12-14-141414', '2021-12-15-151515'
]

ALL_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(s, '%Y-%m-%d-%H%M%S')
    for s in ALL_TIME_STRINGS
], dtype=int)

ALL_HOURS = numpy.array(
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int
)
ALL_MONTHS = numpy.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12], dtype=int
)

# The following constants are used to test filter_by_hour.
HOURS_TO_KEEP = numpy.array([1, 2, 3, 5, 8, 13, 21], dtype=int)
HOUR_INDICES_TO_KEEP = numpy.array([1, 2, 4, 7, 12], dtype=int)

# The following constants are used to test filter_by_month.
MONTHS_TO_KEEP = numpy.array([12, 5, 10, 1], dtype=int)
MONTH_INDICES_TO_KEEP = numpy.array([0, 4, 9, 11, 12, 13, 14], dtype=int)

# The following constants are used to test months_to_string.
WINTER_MONTHS = numpy.array([12, 2, 1], dtype=int)
WINTER_STRING_VERBOSE = 'Jan, Feb, Dec'
WINTER_STRING_ABBREV = 'jfd'

SPRING_MONTHS = numpy.array([3, 5, 4], dtype=int)
SPRING_STRING_VERBOSE = 'Mar, Apr, May'
SPRING_STRING_ABBREV = 'mam'

SUMMER_MONTHS = numpy.array([8, 6, 7], dtype=int)
SUMMER_STRING_VERBOSE = 'Jun, Jul, Aug'
SUMMER_STRING_ABBREV = 'jja'

FALL_MONTHS = numpy.array([11, 10, 9], dtype=int)
FALL_STRING_VERBOSE = 'Sep, Oct, Nov'
FALL_STRING_ABBREV = 'son'

# The following constants are used to test hours_to_string.
MORNING_HOURS = numpy.array([11, 10, 9, 8, 7, 6], dtype=int)
MORNING_STRING_VERBOSE = '06, 07, 08, 09, 10, 11 UTC'
MORNING_STRING_ABBREV = '06-07-08-09-10-11utc'

AFTERNOON_HOURS = numpy.array([12, 14, 16, 13, 15, 17], dtype=int)
AFTERNOON_STRING_VERBOSE = '12, 13, 14, 15, 16, 17 UTC'
AFTERNOON_STRING_ABBREV = '12-13-14-15-16-17utc'

EVENING_HOURS = numpy.array([18, 23, 19, 22, 21, 20], dtype=int)
EVENING_STRING_VERBOSE = '18, 19, 20, 21, 22, 23 UTC'
EVENING_STRING_ABBREV = '18-19-20-21-22-23utc'

OVERNIGHT_HOURS = numpy.array([0, 1, 2, 5, 3, 4], dtype=int)
OVERNIGHT_STRING_VERBOSE = '00, 01, 02, 03, 04, 05 UTC'
OVERNIGHT_STRING_ABBREV = '00-01-02-03-04-05utc'

# The following constants are used to test _apply_sep_time_one_front_type and
# apply_separation_time.
VALID_TIME_STRINGS = [
    '4055-01-01-00', '4055-01-01-03', '4055-01-01-06', '4055-01-01-09',
    '4055-01-01-12', '4055-01-01-15', '4055-01-01-18', '4055-01-01-21',
    '4055-01-02-00', '4055-01-02-03', '4055-01-02-09', '4055-01-02-12',
    '4055-01-02-15', '4055-01-02-18', '4055-01-02-21', '4055-01-03-00',
    '4055-02-01-00', '4055-02-01-03', '4055-02-01-09', '4055-02-01-12',
    '4055-02-01-15', '4055-02-01-18', '4055-02-01-21', '4055-02-02-00',
    '4055-02-02-03', '4055-02-02-06', '4055-02-02-09', '4055-02-02-12',
    '4055-02-02-15', '4055-02-02-18', '4055-02-02-21', '4055-02-03-00'
]

VALID_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(s, '%Y-%m-%d-%H')
    for s in VALID_TIME_STRINGS
], dtype=int)

NO_FRONT_ENUM = front_utils.NO_FRONT_ENUM
WARM_FRONT_ENUM = front_utils.WARM_FRONT_ENUM
COLD_FRONT_ENUM = front_utils.COLD_FRONT_ENUM

FRONT_TYPE_ENUMS_BEFORE_SEP = numpy.array([
    WARM_FRONT_ENUM, WARM_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, WARM_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, WARM_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, COLD_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, WARM_FRONT_ENUM
], dtype=int)

SEPARATION_TIME_SEC = 80000

FRONT_TYPE_ENUMS_AFTER_WF_SEP = numpy.array([
    WARM_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, COLD_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM, COLD_FRONT_ENUM,
    COLD_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM
], dtype=int)

FRONT_TYPE_ENUMS_AFTER_CF_SEP = numpy.array([
    WARM_FRONT_ENUM, WARM_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, WARM_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM,
    COLD_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, WARM_FRONT_ENUM
], dtype=int)

FRONT_TYPE_ENUMS_AFTER_BOTH_SEP = numpy.array([
    WARM_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, WARM_FRONT_ENUM, COLD_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM,
    COLD_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, COLD_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM, NO_FRONT_ENUM,
    NO_FRONT_ENUM, NO_FRONT_ENUM, WARM_FRONT_ENUM, NO_FRONT_ENUM
], dtype=int)

# The following constants are used to test find_basic_file and
# basic_file_name_to_time.
DIRECTORY_NAME = 'foo'
BASIC_FILE_TIME_UNIX_SEC = 0
LABEL_FILE_NAME = 'foo/front-labels_1970-01-01-000000.nc'
PROPERTY_FILE_NAME = 'foo/front-properties_1970-01-01-000000.nc'

# The following constants are used to test find_monte_carlo_file.
MONTE_CARLO_PROPERTY_NAME = climo_utils.WARM_FRONT_LENGTHS_KEY
FIRST_MONTE_CARLO_ROW = 10
FIRST_MONTE_CARLO_COLUMN = 20
MONTE_CARLO_FILE_NAME = (
    'foo/monte-carlo-test_wf-length-matrix-metres_first-row=010_'
    'first-column=020.nc')

# The following constants are used to test find_statistic_file.
FIRST_AGG_TIME_UNIX_SEC = 0
LAST_AGG_TIME_UNIX_SEC = 86399
HOURS_IN_AGGREGATE = numpy.array([18, 21], dtype=int)
MONTHS_IN_AGGREGATE = numpy.array([12, 1, 2], dtype=int)

COUNT_FILE_NAME = (
    'foo/front-counts_1970-01-01-000000_1970-01-01-235959_hours=18-21utc_'
    'months=jfd.nc')

STATISTIC_FILE_NAME = (
    'foo/front-statistics_1970-01-01-000000_1970-01-01-235959_hours=18-21utc_'
    'months=jfd.nc')


class ClimatologyUtilsTests(unittest.TestCase):
    """Each method is a unit test for climatology_utils.py."""

    def test_exact_times_to_hours(self):
        """Ensures correct output from _exact_times_to_hours."""

        these_hours = climo_utils._exact_times_to_hours(ALL_TIMES_UNIX_SEC)
        self.assertTrue(numpy.array_equal(these_hours, ALL_HOURS))

    def test_exact_times_to_months(self):
        """Ensures correct output from _exact_times_to_months."""

        these_months = climo_utils._exact_times_to_months(ALL_TIMES_UNIX_SEC)
        self.assertTrue(numpy.array_equal(these_months, ALL_MONTHS))

    def test_filter_by_hour(self):
        """Ensures correct output from filter_by_hour."""

        these_indices = climo_utils.filter_by_hour(
            all_times_unix_sec=ALL_TIMES_UNIX_SEC,
            hours_to_keep=HOURS_TO_KEEP)

        self.assertTrue(numpy.array_equal(
            these_indices, HOUR_INDICES_TO_KEEP
        ))

    def test_filter_by_month(self):
        """Ensures correct output from filter_by_month."""

        these_indices = climo_utils.filter_by_month(
            all_times_unix_sec=ALL_TIMES_UNIX_SEC,
            months_to_keep=MONTHS_TO_KEEP)

        self.assertTrue(numpy.array_equal(
            these_indices, MONTH_INDICES_TO_KEEP
        ))

    def test_months_to_string_winter(self):
        """Ensures correct output from months_to_string.

        In this case the season is winter.
        """

        this_verbose_string, this_abbrev_string = climo_utils.months_to_string(
            WINTER_MONTHS)

        self.assertTrue(this_verbose_string == WINTER_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == WINTER_STRING_ABBREV)

    def test_months_to_string_spring(self):
        """Ensures correct output from months_to_string.

        In this case the season is spring.
        """

        this_verbose_string, this_abbrev_string = climo_utils.months_to_string(
            SPRING_MONTHS)

        self.assertTrue(this_verbose_string == SPRING_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == SPRING_STRING_ABBREV)

    def test_months_to_string_summer(self):
        """Ensures correct output from months_to_string.

        In this case the season is summer.
        """

        this_verbose_string, this_abbrev_string = climo_utils.months_to_string(
            SUMMER_MONTHS)

        self.assertTrue(this_verbose_string == SUMMER_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == SUMMER_STRING_ABBREV)

    def test_months_to_string_fall(self):
        """Ensures correct output from months_to_string.

        In this case the season is fall.
        """

        this_verbose_string, this_abbrev_string = climo_utils.months_to_string(
            FALL_MONTHS)

        self.assertTrue(this_verbose_string == FALL_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == FALL_STRING_ABBREV)

    def test_hours_to_string_morning(self):
        """Ensures correct output from hours_to_string.

        In this case the hours are morning.
        """

        this_verbose_string, this_abbrev_string = climo_utils.hours_to_string(
            MORNING_HOURS)

        self.assertTrue(this_verbose_string == MORNING_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == MORNING_STRING_ABBREV)

    def test_hours_to_string_afternoon(self):
        """Ensures correct output from hours_to_string.

        In this case the hours are afternoon.
        """

        this_verbose_string, this_abbrev_string = climo_utils.hours_to_string(
            AFTERNOON_HOURS)

        self.assertTrue(this_verbose_string == AFTERNOON_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == AFTERNOON_STRING_ABBREV)

    def test_hours_to_string_evening(self):
        """Ensures correct output from hours_to_string.

        In this case the hours are evening.
        """

        this_verbose_string, this_abbrev_string = climo_utils.hours_to_string(
            EVENING_HOURS)

        self.assertTrue(this_verbose_string == EVENING_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == EVENING_STRING_ABBREV)

    def test_hours_to_string_overnight(self):
        """Ensures correct output from hours_to_string.

        In this case the hours are overnight.
        """

        this_verbose_string, this_abbrev_string = climo_utils.hours_to_string(
            OVERNIGHT_HOURS)

        self.assertTrue(this_verbose_string == OVERNIGHT_STRING_VERBOSE)
        self.assertTrue(this_abbrev_string == OVERNIGHT_STRING_ABBREV)

    def test_apply_separation_time_wf(self):
        """Ensures correct output from _apply_sep_time_one_front_type.

        In this case, removing only warm-front labels.
        """

        these_front_type_enums = climo_utils._apply_sep_time_one_front_type(
            front_type_enums=FRONT_TYPE_ENUMS_BEFORE_SEP + 0,
            valid_times_unix_sec=VALID_TIMES_UNIX_SEC,
            separation_time_sec=SEPARATION_TIME_SEC,
            relevant_front_type_enum=WARM_FRONT_ENUM)

        self.assertTrue(numpy.array_equal(
            these_front_type_enums, FRONT_TYPE_ENUMS_AFTER_WF_SEP
        ))

    def test_apply_separation_time_cf(self):
        """Ensures correct output from _apply_sep_time_one_front_type.

        In this case, removing only cold-front labels.
        """

        these_front_type_enums = climo_utils._apply_sep_time_one_front_type(
            front_type_enums=FRONT_TYPE_ENUMS_BEFORE_SEP + 0,
            valid_times_unix_sec=VALID_TIMES_UNIX_SEC,
            separation_time_sec=SEPARATION_TIME_SEC,
            relevant_front_type_enum=COLD_FRONT_ENUM)

        self.assertTrue(numpy.array_equal(
            these_front_type_enums, FRONT_TYPE_ENUMS_AFTER_CF_SEP
        ))

    def test_apply_separation_time(self):
        """Ensures correct output from apply_separation_time."""

        random_indices = numpy.random.permutation(len(VALID_TIMES_UNIX_SEC)) - 1

        these_front_type_enums, these_times_unix_sec = (
            climo_utils.apply_separation_time(
                front_type_enums=FRONT_TYPE_ENUMS_BEFORE_SEP[random_indices],
                valid_times_unix_sec=VALID_TIMES_UNIX_SEC[random_indices],
                separation_time_sec=SEPARATION_TIME_SEC)
        )

        self.assertTrue(numpy.array_equal(
            these_front_type_enums, FRONT_TYPE_ENUMS_AFTER_BOTH_SEP
        ))
        self.assertTrue(numpy.array_equal(
            these_times_unix_sec, VALID_TIMES_UNIX_SEC
        ))

    def test_find_basic_file_labels(self):
        """Ensures correct output from find_basic_file.

        In this case the file type is front labels.
        """

        this_file_name = climo_utils.find_basic_file(
            directory_name=DIRECTORY_NAME,
            file_type_string=climo_utils.FRONT_LABELS_STRING,
            valid_time_unix_sec=BASIC_FILE_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == LABEL_FILE_NAME)

    def test_find_basic_file_properties(self):
        """Ensures correct output from find_basic_file.

        In this case the file type is front properties.
        """

        this_file_name = climo_utils.find_basic_file(
            directory_name=DIRECTORY_NAME,
            file_type_string=climo_utils.FRONT_PROPERTIES_STRING,
            valid_time_unix_sec=BASIC_FILE_TIME_UNIX_SEC,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == PROPERTY_FILE_NAME)

    def test_basic_file_name_to_time_labels(self):
        """Ensures correct output from basic_file_name_to_time.

        In this case the file type is front labels.
        """

        self.assertTrue(
            climo_utils.basic_file_name_to_time(LABEL_FILE_NAME) ==
            BASIC_FILE_TIME_UNIX_SEC
        )

    def test_basic_file_name_to_time_properties(self):
        """Ensures correct output from basic_file_name_to_time.

        In this case the file type is front properties.
        """

        self.assertTrue(
            climo_utils.basic_file_name_to_time(PROPERTY_FILE_NAME) ==
            BASIC_FILE_TIME_UNIX_SEC
        )

    def test_find_monte_carlo_file(self):
        """Ensures correct output from find_monte_carlo_file."""

        this_file_name = climo_utils.find_monte_carlo_file(
            directory_name=DIRECTORY_NAME,
            property_name=MONTE_CARLO_PROPERTY_NAME,
            first_grid_row=FIRST_MONTE_CARLO_ROW,
            first_grid_column=FIRST_MONTE_CARLO_COLUMN,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == MONTE_CARLO_FILE_NAME)

    def test_find_count_file(self):
        """Ensures correct output from find_aggregated_file.

        In this case the file type is front counts.
        """

        this_file_name = climo_utils.find_aggregated_file(
            directory_name=DIRECTORY_NAME,
            file_type_string=climo_utils.FRONT_COUNTS_STRING,
            first_time_unix_sec=FIRST_AGG_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_AGG_TIME_UNIX_SEC,
            hours=HOURS_IN_AGGREGATE, months=MONTHS_IN_AGGREGATE,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == COUNT_FILE_NAME)

    def test_find_statistic_file(self):
        """Ensures correct output from find_aggregated_file.

        In this case the file type is front statistics.
        """

        this_file_name = climo_utils.find_aggregated_file(
            directory_name=DIRECTORY_NAME,
            file_type_string=climo_utils.FRONT_STATS_STRING,
            first_time_unix_sec=FIRST_AGG_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_AGG_TIME_UNIX_SEC,
            hours=HOURS_IN_AGGREGATE, months=MONTHS_IN_AGGREGATE,
            raise_error_if_missing=False)

        self.assertTrue(this_file_name == STATISTIC_FILE_NAME)


if __name__ == '__main__':
    unittest.main()
