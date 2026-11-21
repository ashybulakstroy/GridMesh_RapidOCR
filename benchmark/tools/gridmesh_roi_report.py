import json,time,html,traceback
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageOps
from benchmark.core.measure import extract_rois,is_numeric_field,gt_fields
from App.geometry.service import build_grid,cell_crop,cell_polygon
from benchmark.core.gridmesh import total_crop
out=Path('benchmark/data/output/gridmesh_roi_comparison'); out.mkdir(parents=True,exist_ok=True); E=lambda x:html.escape(str(x)); base=Path('benchmark/data'); sections=[]; summary=[]
cv2.setNumThreads(2)
for gfile in sorted((base/'ground_truth').glob('*.json')):
 g=json.loads(gfile.read_text(encoding='utf-8')); rid=g['receipt_id']; ann=json.loads((base/'annotations'/f'{rid}.json').read_text(encoding='utf-8'))
 with Image.open(base/'images'/g['source_image']) as p: img=cv2.cvtColor(np.array(ImageOps.exif_transpose(p).convert('RGB')),cv2.COLOR_RGB2BGR)
 h,w=img.shape[:2]; old=dict(extract_rois(img,ann)); fields=gt_fields(g); start=time.perf_counter()
 try:
  rect,overlay,info=build_grid(img,rectify=False); elapsed=time.perf_counter()-start; ox,oy=rect.info['source_offset']; cv2.imwrite(str(out/f'{rid}_grid.png'),overlay)
  full_overlay=img.copy(); full_overlay[oy:oy+overlay.shape[0],ox:ox+overlay.shape[1]]=overlay
  _,total_corners=total_crop(img,ann)
  cv2.polylines(full_overlay,[np.array(total_corners,dtype=np.int32)],True,(0,180,0),3,cv2.LINE_AA)
  cv2.imwrite(str(out/f'{rid}_grid_total.png'),full_overlay)
  cells=[]
  for row in range(len(rect.ys)-1):
   for col in range(len(rect.xs)-1):
    poly=cell_polygon(rect,row,col)+[ox,oy]; cells.append((row,col,poly.astype(np.float32)))
  title=f'<h2>{E(rid)}</h2><p>Поиск сетки: {elapsed:.3f} с; линий: {len(rect.xs)} вертикальных, {len(rect.ys)} горизонтальных. Метод: {E(info["method"])}. Предупреждения: {E(info["warnings"])}</p><p><a href="{rid}_grid.png">Посмотреть синюю сетку целиком</a> (область таблицы). <a href="{rid}_grid_total.png">Grid Mesh + итог: полное фото</a>; зелёная рамка — ручная ROI итога.</p>'
  section=[title,'<table><tr><th>Поле и эталон</th><th>Текущие координаты ROI (пиксели) и фактический вход OCR бенчмарка</th><th>GridMesh: ячейка, углы на исходном фото и вырезка</th></tr>']; matches=0
  for i,entry in enumerate(ann['rois']):
   field=entry['field_id']
   if not is_numeric_field(field): continue
   x0,y0,x1,y1=entry['bbox']; bbox=[round(x0*w),round(y0*h),round(x1*w),round(y1*h)]; orig=img[bbox[1]:bbox[3],bbox[0]:bbox[2]]; cv2.imwrite(str(out/f'{rid}_{i}_roi.png'),orig); cv2.imwrite(str(out/f'{rid}_{i}_old.png'),old[field]); cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
   found=[(r,c,p) for r,c,p in cells if cv2.pointPolygonTest(p,(cx,cy),False)>=0]
   if field=='printed_total':
    crop,corners=total_crop(img,ann); cv2.imwrite(str(out/f'{rid}_{i}_mesh.png'),crop); matches+=1
    mesh=f'Отдельная ячейка итога по ручной ROI; синтетические границы, без автоматического поиска. Углы: {E(corners)}<br><img src="{rid}_{i}_mesh.png" alt="GridMesh total">'
   elif len(found)==1:
    row,col,poly=found[0]; crop,offset,corners=cell_crop(rect,row,col); cv2.imwrite(str(out/f'{rid}_{i}_mesh.png'),crop); matches+=1
    mesh=f'Строка сетки {row}, столбец {col}; углы TL, TR, BR, BL: {E(corners)}<br><img src="{rid}_{i}_mesh.png" alt="GridMesh">'
   else: mesh='Нет однозначной ячейки сетки по центру текущей ROI; автоматическое сопоставление не выполнено.'
   section.append(f'<tr><td>{E(field)}<br>Эталон: {E(fields[field]["value"])}</td><td>{E(bbox)}<br>Исходный ROI: <img src="{rid}_{i}_roi.png" alt="ROI"><br>После подготовки для OCR: <img src="{rid}_{i}_old.png" alt="подготовка OCR"></td><td>{mesh}</td></tr>')
  section.append('</table>'); sections.extend(section); summary.append((rid,matches,len([a for a in ann['rois'] if is_numeric_field(a['field_id'])])))
 except Exception as exc:
  sections.append(f'<h2>{E(rid)}</h2><p class="warn">GridMesh не построил сетку: {E(exc)}</p>'); summary.append((rid,0,0)); (out/f'{rid}_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение вырезок: ROI и GridMesh</title><style>body{font:15px system-ui;max-width:1600px;margin:2em auto;padding:0 1em;color:#162536}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd6df;padding:8px;vertical-align:top;text-align:left}th{background:#e8eff7}img{max-width:450px;max-height:180px;vertical-align:middle;background:#fff} .warn{background:#fff2c4;padding:1em}</style><h1>Сравнение вырезок чисел: текущие ROI и GridMesh</h1><p>Текущая ветка: координаты аннотаций в исходных пикселях, при цене товара расширение по X, затем tighten_numeric_crop. Показаны ROI до расширения и подготовленное изображение, которое поступало на OCR в бенчмарке.</p><p>GridMesh: функция проекта App.geometry.service.build_grid и cell_crop. Координаты — четыре угла в исходном изображении; внутри ячейка выпрямляется по карте. Соответствие полю определено по положению центра текущего ROI. Это предварительное геометрическое сопоставление, которое может указать неправильную строку, особенно у границ; проверяйте по синей сетке. Вырезки GridMesh показаны до дальнейшей подготовки OCR приложения.</p><p class="warn">Значения и текущая разметка среднего чека черновые. Итог добавлен отдельной ячейкой по ручной разметке, вне товарной сетки. Прямоугольная ROI не задаёт кривизну, поэтому геометрия итога прямоугольная. OCR и качество новых вырезок здесь не измерялись. Распознавание и производственный код не менялись.</p>'''+''.join(sections)+'</html>'
(out/'report.html').write_text(page,encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False));print('REPORT',out/'report.html')
