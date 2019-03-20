"""Unit tests for cnn.py."""

import unittest
from generalexam.machine_learning import cnn

MODEL_FILE_NAME = 'foo/bar/model.h5'
MODEL_METAFILE_NAME = 'foo/bar/model_metadata.p'


class CnnTests(unittest.TestCase):
    """Each method is a unit test for cnn.py."""

    def test_find_metafile(self):
        """Ensures correct output from find_metafile."""

        this_file_name = cnn.find_metafile(
            model_file_name=MODEL_FILE_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == MODEL_METAFILE_NAME)


if __name__ == '__main__':
    unittest.main()
