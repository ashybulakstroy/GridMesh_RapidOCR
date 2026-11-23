"""Six Cyrillic OCR engines; preserve errors, do not substitute fallbacks."""
from __future__ import annotations
import asyncio, hashlib, io, os, sys
from pathlib import Path
from PIL import Image

class RapidAdapter:
    def __init__(self, spec, root, base):
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion, ModelType
        self.spec=spec
        # Since input is already an OCR cell, detector and orientation model are unnecessary.
        params={'Global.use_det':False,'Global.use_cls':False}
        if spec.get('model_path'):
            p=base/spec['model_path']
            if not p.is_file():p=root/spec['model_path']
            if not p.is_file():raise FileNotFoundError(f'MODEL_MISSING: {p}; run prepare_models')
            expected=spec.get('model_sha256')
            if expected and hashlib.sha256(p.read_bytes()).hexdigest().lower()!=expected.lower():
                raise ValueError(f'MODEL_SHA256_MISMATCH: {p}')
            params['Rec.model_path']=str(p)
            params['Rec.lang_type']=spec.get('lang','cyrillic')
            if spec.get('rec_img_shape'):
                # Declared by model publisher, not performance tuning.
                params['Rec.rec_img_shape']=list(spec['rec_img_shape'])
            self.model_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            self.model_path=str(p)
        else:
            params['Rec.ocr_version']=getattr(OCRVersion,spec['ocr_version'])
            params['Rec.model_type']=getattr(ModelType,spec['model_type'])
            params['Rec.lang_type']=spec['lang']
            self.model_path='RapidOCR managed cache'
            self.model_sha256=None
        self.engine=RapidOCR(params=params)
    def recognize(self,img):
        import numpy as np
        arr=np.ascontiguousarray(np.asarray(img.convert('RGB'))[:,:,::-1])
        out=self.engine.recognize_txt([arr])
        txts=getattr(out,'txts',None);scores=getattr(out,'scores',None)
        text=str(txts[0]) if txts is not None and len(txts) else ''
        score=float(scores[0]) if scores is not None and len(scores) else None
        return text,score
    def close(self):pass

class TesseractAdapter:
    def __init__(self, spec, root, base):
        from tesserocr import PyTessBaseAPI, PSM
        directory=root/'models/tessdata'
        if any(not (directory/(l+'.traineddata')).exists() for l in ('rus','eng')):
            raise FileNotFoundError('tessdata_fast rus/eng missing: run prepare')
        # Input is a known text line: PSM.SINGLE_LINE is required for fair REC-only comparison.
        # No character restrictions, DPI tuning or other engine parameters.
        self.api=PyTessBaseAPI(path=str(directory),lang='rus+eng',psm=PSM.SINGLE_LINE)
    def recognize(self,img):
        self.api.SetImage(img)
        text=self.api.GetUTF8Text().strip()
        score=float(self.api.MeanTextConf())/100
        self.api.Clear()
        return text,score
    def close(self):self.api.End()

class WindowsAdapter:
    def __init__(self,spec,root,base):
        if sys.platform!='win32':raise OSError('Windows OCR is only available on Windows 10+')
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.globalization import Language
        tags=[x.language_tag for x in OcrEngine.available_recognizer_languages]
        ru=next((x for x in tags if x.lower().startswith('ru')),None)
        if not ru:raise RuntimeError(f'Windows Russian OCR language pack is not installed. Available: {tags}')
        self.engine=OcrEngine.try_create_from_language(Language(ru))
        if self.engine is None:raise RuntimeError('OcrEngine.try_create_from_language returned None')
        self.language=ru
    async def _read(self,img):
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import InMemoryRandomAccessStream, DataWriter
        file=io.BytesIO();img.convert('RGB').save(file,format='PNG')
        stream=InMemoryRandomAccessStream(); writer=DataWriter(stream.get_output_stream_at(0))
        writer.write_bytes(file.getvalue())
        await writer.store_async(); await writer.flush_async()
        stream.seek(0)
        decoder=await BitmapDecoder.create_async(stream)
        bitmap=await decoder.get_software_bitmap_async()
        result=await self.engine.recognize_async(bitmap)
        return ' '.join(line.text for line in result.lines)
    def recognize(self,img):
        return asyncio.run(self._read(img)),None
    def close(self):pass

def make_adapter(spec, root, base):
    return {'rapid':RapidAdapter,'tesseract':TesseractAdapter,'windows':WindowsAdapter}[spec['kind']](spec,root,base)
