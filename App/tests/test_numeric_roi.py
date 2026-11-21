import cv2
import numpy as np
from pathlib import Path

from App.ocr.numeric_roi import crop_numeric_ink
from App.ocr.p1_batch import BatchCell, read_batches


def test_rim_crop_retains_leading_digit_and_decimal_punctuation():
    image = np.full((90,130,3),255,np.uint8)
    cv2.putText(image,'1120.7',(0,53),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    text_pixels = np.any(image < 128,axis=2)
    # Separate table fragments near the top/bottom/side cannot define numeric ROI.
    image[0,5:120] = 0
    image[-1,5:120] = 0
    image[10:80,-1] = 0
    crop, info = crop_numeric_ink(image)
    assert info['changed']
    x, y = info['offset']
    h, w = crop.shape[:2]
    assert x == 0  # the leading 1 at the image edge must survive
    text_y,text_x = np.nonzero(text_pixels)
    assert x <= text_x.min() and x+w > text_x.max()
    assert y <= text_y.min() and y+h > text_y.max()
    assert np.array_equal(crop,image[y:y+h,x:x+w])


def test_rim_cleanup_does_not_join_separate_numeric_lines():
    image = np.full((95,130,3),255,np.uint8)
    cv2.putText(image,'150',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    cv2.putText(image,'2250',(20,70),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    image[0,:] = 0
    image[-1,:] = 0
    crop, info = crop_numeric_ink(image)
    assert info['changed']
    _, y = info['offset']
    assert y <= 16 and y+crop.shape[0] >= 71


def test_narrow_digit_one_at_border_is_not_discarded_as_table_rule():
    image = np.full((80,100,3),255,np.uint8)
    image[31:49,1:3] = 0  # genuine narrow leading 1 at the left rim
    cv2.putText(image,'120',(10,48),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
    image[0,5:95] = 0
    image[12:70,-1] = 0
    crop, info = crop_numeric_ink(image)
    x,y = info.get('offset',[0,0])
    assert x == 0
    assert np.all(crop[31-y:49-y,1:3] == 0)


def test_tall_mesh_cells_read_quantity_and_price_in_one_batch():
    fixtures = Path(__file__).parent / 'fixtures'
    cells = [BatchCell(cv2.imread(str(fixtures / f'fold_row5_{field}.png')),
                       5, field, (0, 0), []) for field in ('quantity', 'price')]
    results, stats = read_batches(cells)
    assert stats['batch_calls'] == 1
    assert stats['multiline_fallback'] == stats['batch_fallback'] == 0
    for field, expected in [('quantity', '1'), ('price', '3960T')]:
        result = results[5, field]
        assert result['numeric_roi']['reason'] == 'compact_number_band'
        assert result['text'] == expected


def test_compact_band_retains_tiny_decimal_mark():
    image = np.full((100, 140, 3), 224, dtype=np.uint8)
    cv2.putText(image, '1120', (20, 58), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1)
    image[59, 73] = 0
    image[5:95, 0] = 0
    crop, info = crop_numeric_ink(image)
    assert info['reason'] == 'compact_number_band'
    x, y = info['offset']
    assert 0 <= 73-x < crop.shape[1] and 0 <= 59-y < crop.shape[0]
    assert np.all(crop[59-y, 73-x] == 0)


def test_fragmented_rails_in_tall_receipt_cells_do_not_distort_numbers():
    fixtures = Path(__file__).parent / 'fixtures'
    cells = [BatchCell(cv2.imread(str(fixtures/f'receipt252161_row{row}_{field}.png')),
                       row,field,(0,0),[]) for row,field in [(15,'quantity'),(15,'price'),(16,'price')]]
    results,stats=read_batches(cells)
    from App.ocr.p1_numbers import parse_number
    assert [parse_number(results[c.row_id,c.column_id]['text']) for c in cells]==[5,2000,60]
    assert stats['numeric_roi_skipped']==0


def test_compact_crop_does_not_discard_isolated_one_at_outer_edge():
    image=np.full((120,120,3),255,np.uint8)
    image[50:68,0:2]=0
    cv2.putText(image,'200',(12,67),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
    crop,info=crop_numeric_ink(image)
    x,y=info.get('offset',[0,0])
    assert x==0
    assert np.all(crop[50-y:68-y,:2]==0)
