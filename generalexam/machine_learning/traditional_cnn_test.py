"""Unit tests for traditional_cnn.py."""

import unittest
from generalexam.machine_learning import traditional_cnn

MODEL_FILE_NAME = 'foo/bar/model.h5'
MODEL_METAFILE_NAME = 'foo/bar/model_metadata.p'


class TraditionalCnnTests(unittest.TestCase):
    """Each method is a unit test for traditional_cnn.py."""

    def test_find_metafile(self):
        """Ensures correct output from find_metafile."""

        this_file_name = traditional_cnn.find_metafile(
            model_file_name=MODEL_FILE_NAME, raise_error_if_missing=False)
        self.assertTrue(this_file_name == MODEL_METAFILE_NAME)


if __name__ == '__main__':
    unittest.main()
