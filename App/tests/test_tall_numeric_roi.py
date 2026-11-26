import cv2
import numpy as np

from App.ocr.tall_numeric_roi import compact_tall


def test_single_pixel_decimal_is_preserved():
    image = np.full((150, 110), 255, np.uint8)
    cv2.putText(image, '12', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, 2)
    image[80, 65] = 0

    cropped, roi = compact_tall(image)

    assert roi['changed']
    assert np.count_nonzero(cropped == 0) == np.count_nonzero(image == 0)


def test_multiline_number_is_not_cropped_to_one_line():
    image = np.full((150, 100), 255, np.uint8)
    for y in (45, 105):
        cv2.putText(image, '20', (30, y), cv2.FONT_HERSHEY_SIMPLEX, .65, 0, 2)
    assert compact_tall(image) is None
