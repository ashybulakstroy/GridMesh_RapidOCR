"""Second OCR path: retain *original* source pixels, never remap them.

The physical grid gives a polygon; the crop of that polygon is retained as
an in-RAM numpy array until processing ends. Numeric ROI preparation and a
single recognizer pass are done lazily only for risk-prioritized cells.
"""
from __future__ import annotations

from time import perf_counter
import cv2
import numpy as np

from ..geometry.service import cell_polygon
from .numeric_roi import crop_numeric_ink
from .p1_numbers import parse_number
from .experiment_rec_trace import rec_scope


class OriginalNumericCache:
    def __init__(self, image, rect, *, max_bytes=32*1024*1024):
        self.image = image
        self.rect = rect
        self.max_bytes = max_bytes
        self.crops = {}
        self.bytes = 0
        self.prepare_ms = 0.0
        self.errors = []

    def _polygon(self, row_index, col_index):
        rect = self.rect
        if rect.method == 'original_cells':
            points = cell_polygon(rect, row_index, col_index)
            points = points + np.asarray(rect.info['source_offset'])
            return points
        # The entire rectified mesh exists; sample all four edges, not just
        # four corners, so a curved cell is not cut at its boundaries.
        xs = list(map(round, rect.xs)); ys = list(map(round, rect.ys))
        x0,x1 = xs[col_index]+1,xs[col_index+1]-1
        y0,y1 = ys[row_index]+1,ys[row_index+1]-1
        if x1 <= x0 or y1 <= y0:
            raise ValueError('empty source polygon')
        samples = 17
        xy = []
        for x,y in ([(int(x),y0) for x in np.linspace(x0,x1,samples)] +
                    [(x1,int(y)) for y in np.linspace(y0,y1,samples)] +
                    [(int(x),y1) for x in np.linspace(x1,x0,samples)] +
                    [(x0,int(y)) for y in np.linspace(y1,y0,samples)]):
            xy.append((rect.map_x[y,x], rect.map_y[y,x]))
        return np.asarray(xy)

    def store(self, row_id, field, column):
        stamp = perf_counter()
        try:
            poly = self._polygon(row_id, column)
            if not np.isfinite(poly).all():
                raise ValueError('nonfinite polygon')
            H,W = self.image.shape[:2]
            x0 = max(0, int(np.floor(poly[:,0].min()))-4)
            y0 = max(0, int(np.floor(poly[:,1].min()))-4)
            x1 = min(W, int(np.ceil(poly[:,0].max()))+5)
            y1 = min(H, int(np.ceil(poly[:,1].max()))+5)
            if x1-x0<6 or y1-y0<6 or (x1-x0)*(y1-y0)>500_000:
                raise ValueError('invalid crop bounds/size')
            crop = self.image[y0:y1, x0:x1].copy()
            points = np.round(poly-[x0,y0]).astype(np.int32)
            mask = np.zeros(crop.shape[:2],dtype=np.uint8)
            cv2.fillPoly(mask,[points],255)
            # A tiny dilation protects edge-adjacent digit ink from inaccurate
            # detected grid boundaries; keep outside white instead of black.
            mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=1)
            crop[mask==0]=255
            if self.bytes + crop.nbytes > self.max_bytes:
                raise ValueError('source_roi_cache_memory_guard')
            self.crops[(int(row_id),field)] = crop
            self.bytes += crop.nbytes
        except (ValueError, cv2.error, IndexError, TypeError, KeyError) as exc:
            self.errors.append({'row_id':row_id,'field':field,'error':f'{type(exc).__name__}: {exc}'})
        finally:
            self.prepare_ms+=(perf_counter()-stamp)*1000

    def escalate_suspects(self, book, first_pass, model_pool, *, max_calls=3, only_keys=None):
        """Optional *second independent* model only for unresolved OCR assignments.

        Never try every model/cell pair; never modify the user-visible result.
        """
        out={'recognitions':0,'attempts':[],'skipped':0}
        if model_pool is None:
            return out
        models=model_pool.models_for('numeric')
        if len(models)<2:
            return out
        second=models[1]
        only_keys=set(only_keys) if only_keys is not None else None
        for item in first_pass.get('rec_results',[]):
            if out['recognitions']>=max_calls:
                break
            key=(item['row_id'],item['field'])
            if only_keys is not None and key not in only_keys:
                continue
            raw=self.crops.get(key)
            if raw is None or not raw.size:
                out['skipped']+=1
                continue
            started=perf_counter()
            try:
                prepared,_=crop_numeric_ink(raw)
                with rec_scope([key], 'original_independent_second_pass'):
                    result=model_pool.recognize(second,prepared,'original_numeric_roi_second')
                record={'row_id':key[0], 'field':key[1], 'model_id':second,
                        'text':result.get('text'),'confidence':result.get('confidence'),
                        'error':result.get('error')}
                if result.get('text') is not None and parse_number(result['text']) is not None:
                    book.add(key[0],key[1],result['text'],result.get('confidence'),
                             'original_pixels_second_model_'+second,model=second)
            except Exception as exc:
                record={'row_id':key[0], 'field':key[1], 'model_id':second,
                        'error':f'{type(exc).__name__}: {exc}'}
            record['wall_ms']=round((perf_counter()-started)*1000,3)
            out['attempts'].append(record)
            out['recognitions']+=1
        return out

    def run_suspects(self, rows, book, *, printed_total=None, vat_rate=16,
                     max_calls=8, read_line=None, model_pool=None):
        """One REC-only candidate per flagged source cell; no silent replacement."""
        if read_line is None:
            from .engine import read_line
        first = []
        for row in rows:
            mismatch = row.get('printed_line_match') is False
            for field in ('quantity','price','printed_total'):
                key=(row['row_id'],field)
                if key not in self.crops or field not in row['cells']:
                    continue
                item=row['cells'][field]
                invalid=parse_number(item.get('text')) is None
                confidence=float(item.get('confidence') or 0)
                if invalid: priority=0
                elif mismatch: priority=1
                elif confidence<.80: priority=2
                else: priority=4
                first.append((priority,confidence,key))
        from decimal import Decimal
        current=Decimal(0)
        complete=True
        for row in rows:
            if row.get('line_total') is None: complete=False
            else: current+=Decimal(row['line_total'])
        expected=parse_number(printed_total)
        mismatch = not complete or expected is None or abs(expected-current)>Decimal('.01')
        # No extra OCR if the receipt is consistent, except low-confidence cells.
        first = [v for v in first if v[0]<3 or mismatch]
        first.sort(key=lambda x:(x[0],x[1],x[2]))
        out=[]
        # E4: an independent model reading the original pixels is preferable
        # to repeating the same v5 on another preparation of the same cell.
        for _,_,key in first[:max_calls]:
            started=perf_counter()
            info={'row_id':key[0],'field':key[1]}
            try:
                prepared,meta=crop_numeric_ink(self.crops[key])
                info['roi']=meta
                with rec_scope([key], 'source_pixel_recovery'):
                    successful=[]
                    if model_pool is not None:
                        available=model_pool.models_for('numeric')
                        # One different model first. A second model may be run
                        # in the escalation stage, not on every suspicious ROI.
                        for model_id in available[:1]:
                            candidate=model_pool.recognize(model_id,prepared,'original_numeric_roi')
                            info.setdefault('extra_models',[]).append(candidate)
                            if (candidate.get('text') is not None and parse_number(candidate['text']) is not None
                                    and float(candidate.get('confidence') or 0)>=.80):
                                book.add(key[0],key[1],candidate['text'],candidate.get('confidence'),
                                         'original_pixels_model_'+model_id,model=model_id)
                                successful.append(candidate)
                    if not successful:
                        alt=read_line(prepared)
                        if alt:
                            obs=book.add(key[0],key[1],alt['text'],alt['confidence'],
                                         'original_pixels_numeric_roi_rec_only')
                            info.update(text=obs['text'],number=obs['number'],
                                        confidence=obs['confidence'],model=obs['model'])
                        else:
                            info['error']='rec_only_returned_no_text'
                info['model_priority']='independent_first_fallback_v5'
            except Exception as exc:
                info['error']=f'{type(exc).__name__}: {exc}'
            info['wall_ms']=round((perf_counter()-started)*1000,3)
            out.append(info)
        return {'cache_cells':len(self.crops),'cache_bytes':self.bytes,
                'cache_prepare_ms':round(self.prepare_ms,3),
                'cache_errors':self.errors,'rec_calls':len(out),
                'rec_results':out,'rec_wall_ms':round(sum(x['wall_ms'] for x in out),3),
                'max_rec_calls':max_calls, 'not_scanned':max(0,len(first)-max_calls)}
