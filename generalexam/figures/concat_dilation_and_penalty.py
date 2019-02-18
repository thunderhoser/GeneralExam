"""Concatenates figures showing dilation procedure and double penalty."""

import os
import numpy
from PIL import Image
from gewittergefahr.plotting import imagemagick_utils

CONCAT_SIZE_PIXELS = int(1e7)

DILATION_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'dilation/dilation.jpg')

DOUBLE_PENALTY_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'double_penalty/double_penalty.jpg')

CONCAT_FILE_NAME = (
    '/localdata/ryan.lagerquist/general_exam/journal_paper/figure_workspace/'
    'dilation/dilation_and_double_penalty.jpg')


def _run():
    """Concatenates figures showing dilation procedure and double penalty.

    This is effectively the main method.
    """

    dilation_image_object = Image.open(DILATION_FILE_NAME)
    double_penalty_image_object = Image.open(DOUBLE_PENALTY_FILE_NAME)

    new_dilation_width_px = double_penalty_image_object.size[0]
    width_ratio = (
        float(new_dilation_width_px) / dilation_image_object.size[0]
    )
    new_dilation_height_px = int(numpy.round(
        dilation_image_object.size[1] * width_ratio
    ))

    small_dilation_file_name = DILATION_FILE_NAME.replace(
        '.jpg', '_resized.jpg')

    print 'Resizing dilation figure to: "{0:s}"...'.format(
        small_dilation_file_name)

    command_string = (
        '/usr/bin/convert "{0:s}" -resize {1:d}x{2:d} "{3:s}"'
    ).format(DILATION_FILE_NAME, new_dilation_width_px,
             new_dilation_height_px, small_dilation_file_name)

    os.system(command_string)

    print (
        'Concatenating dilation and double-penalty figures to: "{0:s}"...'
    ).format(CONCAT_FILE_NAME)

    imagemagick_utils.concatenate_images(
        input_file_names=[small_dilation_file_name, DOUBLE_PENALTY_FILE_NAME],
        output_file_name=CONCAT_FILE_NAME,
        num_panel_rows=2, num_panel_columns=1)

    imagemagick_utils.trim_whitespace(
        input_file_name=CONCAT_FILE_NAME, output_file_name=CONCAT_FILE_NAME)

    imagemagick_utils.resize_image(
        input_file_name=CONCAT_FILE_NAME, output_file_name=CONCAT_FILE_NAME,
        output_size_pixels=CONCAT_SIZE_PIXELS)

    os.remove(small_dilation_file_name)


if __name__ == '__main__':
    _run()
