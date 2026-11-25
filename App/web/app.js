let current=null, selected=null, pendingPaste=null, latestProgress=null, running=false, e4EarlyShown=false;
const $=id=>document.getElementById(id);
function selectPhoto(file){selected=file;if(file){$('original').src=URL.createObjectURL(file);$('photo-empty').hidden=true;$('status').textContent='Фото готово к обработке'}}
$('file').onchange=e=>selectPhoto(e.target.files[0]);
document.addEventListener('paste', e=>{const item=[...(e.clipboardData?.items||[])].find(x=>x.type.startsWith('image/'));if(!item)return;const file=item.getAsFile();if(!file)return;e.preventDefault();selectPhoto(file);if(running){pendingPaste=file;$('status').textContent='Фото из буфера в очереди на распознавание';}else{void runRecognition();}});
function profileRows(samples){const target=$('profile-rows');target.replaceChildren();for(const item of samples||[]){const row=document.createElement('tr');row.classList.toggle('timing-slow',Number(item.wall_ms)>500);for(const v of [item.name,(item.wall_ms/1000).toFixed(3),(item.process_cpu_ms/1000).toFixed(3)]){const td=document.createElement('td');td.textContent=v;row.append(td)}target.append(row)}}
function updateTextReview(){
  const textReview=$('text-review'), rows=current?.rows||[];
  const hasFlags=rows.some(row=>row.cells?.name?.text_review_required!==undefined);
  const textSuspects=hasFlags ? rows.filter(row=>row.kind==='product'&&row.cells?.name?.text_review_required).length : Number(current?.problem_text?.suspects)||0;
  textReview.hidden=textSuspects<=0;
  textReview.textContent=textSuspects>0 ? `Проверьте наименования: признаки возможной ошибки в ${textSuspects} строках.` : '';
}
function renderCorrectionLog(events){
  const entries=$('correction-log-entries');
  entries.replaceChildren();
  updateTextReview();
  $('correction-log-file').textContent=current?.processing_log_file ? `Файл журнала: ${current.processing_log_file}` : '';
  for(const event of events||[]){
    const entry=document.createElement('li');
    entry.textContent=event.message||'';
    entry.className=event.type||'';
    entries.append(entry);
  }
  $('correction-log').hidden=!entries.childElementCount;
}
function cpuGraph(history,finished=false,elapsed=0){
  const bars=$('cpu-graph-bars'),label=$('cpu-graph-label'),axis=$('cpu-graph-axis');
  const samples=(history||[]).filter(x=>Number.isFinite(Number(x.elapsed_s))&&Number.isFinite(Number(x.system_cpu_percent)));
  const duration=Math.max(0,Number(elapsed)||0,...samples.map(x=>Number(x.elapsed_s)));
  const count=Math.ceil(duration*5),buckets=Array(count).fill(null);
  for(const sample of samples){const i=Math.max(0,Math.min(count-1,Math.ceil(Number(sample.elapsed_s)*5)-1));if(count)buckets[i]=Number(sample.system_cpu_percent)}
  const levels='▁▂▃▄▅▆▇█';bars.replaceChildren();
  for(let i=count-1;i>=0;i--){
    const value=buckets[i],bar=document.createElement('span');
    bar.className='cpu-bar';bar.dataset.interval=String(i);
    if(value===null){bar.textContent='\u00a0';bar.dataset.missing='true';bar.title='Нет замера';}
    else{bar.textContent=levels[Math.min(7,Math.max(0,Math.floor(value/100*8)))];bar.dataset.cpu=String(value);bar.classList.add(value>=80?'cpu-high':value>=50?'cpu-medium':'cpu-low');bar.title=`${(i*.2).toFixed(1)}–${((i+1)*.2).toFixed(1)} с: ${value}%`;}
    bars.append(bar);
  }
  bars.dataset.intervals=String(count);
  const last=samples.at(-1);
  label.textContent=last?`Нагрузка на ЦПУ ${Math.round(last.system_cpu_percent)}%`:'Нагрузка на ЦПУ —';
  axis.textContent=`${duration.toFixed(1)} с · шаг 0,2 с`;
  bars.setAttribute('aria-label',`Загрузка ЦП: новые замеры слева, старые справа. ${duration.toFixed(2)} секунд, столбец 0,2 секунды. Пробел означает отсутствие замера. До 50% зеленый, 50–80% желтый, от 80% красный.`);
  if(bars.parentElement)bars.parentElement.scrollLeft=0;
}
function updateProgress(progress){latestProgress=progress;$('receipt-timer').textContent=`${Number(progress.elapsed_s||0).toFixed(1).replace('.',',')} с`;cpuGraph(progress.cpu_history,false,progress.elapsed_s);if($('show-details').checked)profileRows(progress.samples);$('phase').textContent=progress.phase||'Завершение…';$('elapsed').textContent=progress.elapsed_s??0;$('cpu').textContent=progress.cpu_percent??0;$('peak').textContent=progress.cpu_peak_percent??0;$('system-cpu').textContent=progress.system_cpu_percent??0;}
function updateDetails(){const on=$('show-details').checked;$('progress-details').hidden=!on;$('times').hidden=!on;$('profile').hidden=!on;if(on)profileRows(latestProgress?.samples||current?.performance?.samples||[]);else $('profile-rows').replaceChildren()}
$('show-details').onchange=updateDetails;updateDetails();
async function connection(url,options){try{return await fetch(url,options)}catch(_){throw Error('Нет связи с сервером OCR. Убедитесь, что start.bat запущен, и повторите обработку.')}}
function e4ClearEarly(){
  document.getElementById('e4-early-preview')?.remove();
}
function e4EarlyPreview(payload){
  if(!payload || document.getElementById('e4-early-preview'))return;
  const panel=document.createElement('div');panel.id='e4-early-preview';
  panel.style.cssText='border:1px solid #c9d5e4;border-radius:9px;padding:12px;margin:12px 0;background:#f6f9ff';
  const title=document.createElement('h3');title.textContent='Первый результат — математически согласован';panel.append(title);
  const hint=document.createElement('p');hint.textContent=`Готов за ${(Number(payload.elapsed_ms)/1000).toFixed(2)} с. Предварительный результат; полная проверка цифр продолжается.`;panel.append(hint);
  const table=document.createElement('table'),thead=document.createElement('thead'),tr=document.createElement('tr');
  for(const field of ['№','Наименование','Кол-во','Цена','Сумма']){const th=document.createElement('th');th.textContent=field;tr.append(th)}
  thead.append(tr);table.append(thead);const tbody=document.createElement('tbody');
  for(const row of payload.rows||[]){const cells=row.cells||{},tr=document.createElement('tr');
    for(const value of [cells.number?.text,cells.name?.text,cells.quantity?.text,cells.price?.text,row.line_total]){
      const td=document.createElement('td');td.textContent=value??'';tr.append(td)}tbody.append(tr)}
  table.append(tbody);panel.append(table);
  const sum=document.createElement('p');sum.textContent=`Итого по чеку: ${payload.total??'—'} ₸ · Напечатанный НДС: ${payload.printed_vat??'не определён'}`;panel.append(sum);
  const link=document.createElement('a');link.href=`/api/jobs/${encodeURIComponent(payload.job_id)}/early`;link.textContent='Скачать предварительный JSON';panel.append(link);
  const footer=document.createElement('p');footer.id='e4-early-status';footer.textContent='Фоновая проверка продолжается; показанные числа не изменяются автоматически.';panel.append(footer);
  const location=$('result');location.parentNode.insertBefore(panel,location);
  $('status').textContent='Первый результат готов; проверка цифр продолжается…';
}
function e4EarlyFailed(){
  const message=document.getElementById('e4-early-status');
  if(message){message.textContent='Полная обработка не завершена. Предварительный снимок не подтверждён.';message.classList.add('bad')}
}
function e4EarlyFinal(data){
  const panel=document.getElementById('e4-early-preview');if(!panel)return;
  const status=document.getElementById('e4-early-status');const diff=data?.e4_early_comparison;
  if(!diff){status.textContent='Полная проверка завершена без сравнительного отчёта.';return}
  const n=Number(diff.changed_count||0);
  status.textContent=n
    ? `Полная проверка завершена: в итоговом результате изменено полей — ${n}. Первоначальный результат сохранён отдельно.`
    : 'Полная проверка завершена: отображаемые числовые значения не изменились.';
  if(diff.final_review_required){const warn=document.createElement('p');warn.textContent='После полной проверки остались неоднозначные цифры. Проверьте итоговую таблицу.';warn.style.color='#9e2f23';panel.append(warn)}
  if(n){const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Показать изменения после полной проверки';details.append(summary);
    const list=document.createElement('ul');for(const change of diff.changed_fields||[]){const li=document.createElement('li');li.textContent=`${change.row_id==null?'Чек':`Строка ${change.row_id}`} · ${change.field}: ${change.before??'—'} → ${change.after??'—'}`;list.append(li)}details.append(list);panel.append(details)}
}

async function runRecognition(){if(running)return;if(!selected){$('status').textContent='Выберите фото';return} running=true;e4EarlyShown=false;e4ClearEarly();$('receipt-timer').textContent='0,0 с';$('correction-log').hidden=true;$('receipt-heading').classList.remove('receipt-ok','receipt-mismatch');cpuGraph([]);$('run').disabled=true;$('status').textContent='Идёт обработка…';$('job-id').textContent='';$('result').hidden=true;$('live').hidden=false;$('profile-rows').replaceChildren();latestProgress=null;$('cpu-summary').textContent='';$('rectified').removeAttribute('src');try{const f=new FormData();f.append('file',selected,selected.name||'receipt.png');f.append('rectify',$('rectify').checked);const response=await connection('/api/jobs',{method:'POST',body:f});const body=await response.json();if(!response.ok)throw Error(body.detail||'Ошибка обработки');$('job-id').textContent=`Задание: ${body.id}`;let retries=0;for(;;){let r;try{r=await connection(`/api/jobs/${body.id}`);retries=0}catch(e){if(++retries>5)throw e;$('status').textContent='Связь прервалась, повторяем запрос…';await new Promise(resolve=>setTimeout(resolve,1000));continue}const state=await r.json();if(!r.ok)throw Error(r.status===404?'Сервер перезапущен; запустите обработку ещё раз.':state.detail||'Ошибка задания');updateProgress(state.progress||{});if(state.progress?.e4_early_ready&&!e4EarlyShown){e4EarlyPreview(state.progress.e4_first_result);e4EarlyShown=true;}if(state.progress?.preview_ready&&!$('rectified').getAttribute('src')){$('rectified').src=`/api/jobs/${body.id}/preview`;$('geometry-label').textContent=state.progress.rectification==='original_cells'?'Сетка на исходном изображении':'Сетка на выпрямленной таблице'}if(state.status==='failed')throw Error(state.error||'Ошибка обработки');if(state.status==='done'){current=state.result;$('receipt-timer').textContent=`${Number(current.performance?.wall_ms/1000||0).toFixed(1).replace('.',',')} с`;cpuGraph(current.performance?.cpu_history,true,current.performance?.wall_ms/1000);render();$('job-id').textContent=`Файлы: ${current.job_directory}`;e4EarlyFinal(current);$('status').textContent='Полная обработка завершена';break}await new Promise(resolve=>setTimeout(resolve,state.progress?.e4_adaptive_enabled?150:500))}}catch(e){e4EarlyFailed();$('status').textContent=e.message}finally{$('live').hidden=true;$('run').disabled=false;running=false;if(pendingPaste){const next=pendingPaste;pendingPaste=null;selectPhoto(next);void runRecognition()}}}
$('run').onclick=runRecognition;
function render(){ e4ClearEarly();renderCorrectionLog(current.processing_log);$('result').hidden=false;$('original').src=current.original;$('rectified').src=current.rectified;$('geometry-label').textContent=current.geometry.rectification==='original_cells'?'Сетка на исходном изображении':'Сетка построена на выпрямленной таблице';$('total').value=current.total??'';$('total').title='Варианты OCR итога: '+(current.numeric_candidate_book?.footer||[]).map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');$('vat-printed').closest('p').hidden=current.numeric_matrix?.status==='disabled';$('vat-printed').value=current.printed_vat??'';$('vat-printed').title='Варианты НДС: '+(current.numeric_candidate_book?.vat||[]).map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');$('rows').replaceChildren();for(const row of current.rows){const tr=document.createElement('tr');for(const k of ['number','name','quantity','price']){const td=document.createElement('td'),el=document.createElement('input');el.value=row.cells[k].text;el.dataset.field=k;if(k==='name'&&row.cells.name?.text_review_required)el.classList.add('text-review-yellow');if(['quantity','price'].includes(k)&&Number(row.cells[k].confidence)<.8)el.classList.add('low-confidence');el.title=row.cells[k].derived?'Номер получен по порядку физической строки':`Уверенность: ${Math.round((row.cells[k].confidence??0)*100)}%`;if(['quantity','price'].includes(k)){const obs=current.numeric_candidate_book?.cells?.[`${row.row_id}:${k}`]?.observations||[];if(obs.length)el.title+=' | Кандидаты: '+obs.map(x=>`${x.text} (${x.confidence==null?'—':Math.round(x.confidence*100)+'%'}, ${x.path})`).join('; ');}el.oninput=()=>{row.cells[k].text=el.value;
if(k==='name'){row.cells[k].text_review_required=!el.value.trim();row.cells[k].text_review_reasons=el.value.trim()?[]:['empty_name'];row.cells[k].user_verified=!!el.value.trim();el.classList.toggle('text-review-yellow',!el.value.trim());updateTextReview();}
if(['quantity','price'].includes(k)){row.cells[k].user_verified=true;
el.title='Введено пользователем · приоритет над OCR';el.classList.remove('low-confidence')}
calculate()};td.append(el);tr.append(td)}const sum=document.createElement('td');sum.className='line verification-detail';if(row.printed_line_total!=null)sum.title='Напечатано: '+row.printed_line_total+' ₸';tr.append(sum);const kindCell=document.createElement('td'),kind=document.createElement('select');for(const [v,label] of [['product','Товар'],['tax','НДС']]){const opt=document.createElement('option');opt.value=v;opt.textContent=label;kind.append(opt)}kind.value=row.kind;kind.onchange=()=>{row.kind=kind.value;calculate()};kindCell.className='verification-detail';kindCell.append(kind);tr.append(kindCell);const conf=document.createElement('td');conf.className='ocr-confidence verification-detail';for(const [field,label] of [['name','Название'],['quantity','Кол-во'],['price','Цена']]){const item=document.createElement('div'),score=row.cells[field].confidence;item.dataset.field=field;item.textContent=`${label}: ${score==null?'—':Math.round(score*100)+'%'}`;if(score!=null&&score<.8)item.classList.add('low-confidence');conf.append(item)}tr.append(conf);$('rows').append(tr)}$('times').textContent=`Геометрия: ${current.timings.geometry_s} с · OCR: ${current.timings.ocr_s} с · Всего: ${current.timings.total_s} с`;const p=current.performance;$('cpu-summary').textContent=`Процесс: ${(p.process_cpu_ms/1000).toFixed(2)} с CPU · Средняя загрузка: ${(p.cpu_core_equivalents*100).toFixed(0)}% · Пик процесса: ${p.peak_process_cpu_percent}% · Пик всего CPU: ${p.peak_system_cpu_percent}%`;if($('show-details').checked)profileRows(p.samples);$('total').oninput=()=>{current.user_total_verified=true;calculate()};$('vat-printed').oninput=()=>{current.printed_vat=$('vat-printed').value;current.user_vat_verified=true;calculate()};calculate();$('receipt-editor').open=current.ocr_validation_status!=='consistent'}
function renderReceiptView(){
 const target=$('receipt-view');target.replaceChildren();
 const table=document.createElement('table'),head=document.createElement('tr');
 for(const label of ['№','Наименование','Кол-во','Цена','Сумма']){const th=document.createElement('th');th.textContent=label;head.append(th)}
 const thead=document.createElement('thead');thead.append(head);table.append(thead);
 const body=document.createElement('tbody');
 for(const row of current.rows){const tr=document.createElement('tr');
  for(const [i,text] of [row.cells.number?.text,row.cells.name?.text,row.cells.quantity?.text,row.cells.price?.text,row.line_total].entries()){const td=document.createElement('td');td.textContent=text??'—';if(i===1&&row.cells.name?.text_review_required){td.classList.add('text-review-yellow');td.title='Проверьте наименование: '+(row.cells.name.text_review_reasons||[]).join(', ')}tr.append(td)}body.append(tr)}
 table.append(body);target.append(table);
 const total=document.createElement('p');total.textContent=`Итого: ${current.total??'—'} ₸ · Сумма строк: ${current.computed_total??'—'} ₸ · Напечатанный НДС: ${current.printed_vat??'нет чтения'}`;target.append(total);
}
function num(x){
  let s=String(x??'').trim().replace(/^₸\s*/u,'').replace(/\s*(?:₸|[ТT]|тг\.?|тенге)\s*$/iu,'').trim().replace(/[ \u00a0\u202f\u2009]/g,' ');
  if(!s||!/^[0-9., ]+$/.test(s))return null;
  if(s.includes(' ')){
    if(!/^\d{1,3}(?: \d{3})+(?:[.,]\d{1,3})?$/.test(s))return null;
    s=s.replace(/ /g,'');
  }
  if(s.includes(',')&&s.includes('.')){
    const dec=s.lastIndexOf(',')>s.lastIndexOf('.')?',':'.';
    const group=dec===','?'.':',';
    const parts=s.split(dec);
    if(parts.length!==2||!/^\d{1,3}$/.test(parts[1]))return null;
    const beforeRe=group==='.'?/^\d{1,3}(?:\.\d{3})+$/:/^\d{1,3}(?:,\d{3})+$/;
    if(!beforeRe.test(parts[0]))return null;
    s=parts[0].split(group).join('')+'.'+parts[1];
  }else if(s.includes(',')||s.includes('.')){
    const sep=s.includes(',')?',':'.';const parts=s.split(sep);
    if(parts.some(p=>!/^\d+$/.test(p)))return null;
    if(parts.length>2){
      if(!(parts[0].length>=1&&parts[0].length<=3&&parts.slice(1).every(p=>p.length===3)))return null;
      s=parts.join('');
    }else if(parts.length===2){
      if(parts[1].length===3&&parts[0].length<=3)return null;
      if(parts[1].length<1||parts[1].length>3)return null;
      s=parts[0]+'.'+parts[1];
    }
  }
  return /^\d+(?:\.\d+)?$/.test(s)&&Number.isFinite(Number(s))?Number(s):null;
}
function calculate(){let sum=0,valid=true,products=0,vat=0,hasVat=false,issues=[];[...$('rows').children].forEach((tr,i)=>{let row=current.rows[i],q=num(row.cells.quantity.text),p=num(row.cells.price.text);let ok=q!==null&&p!==null;valid&&=ok;row.line_total=ok?(q*p).toFixed(2):null;const ptotal=num(row.cells.printed_total?.text);row.printed_line_match=ptotal!==null&&ok?Math.abs(q*p-ptotal)<.011:null;const line=tr.querySelector('.line');line.textContent=row.line_total??'—';line.classList.toggle('bad',row.printed_line_match===false);tr.classList.toggle('bad',!ok);for(const [key,value,label] of [['quantity',q,'количество'],['price',p,'цена']])if(String(row.cells[key].text||'').trim()&&value===null)issues.push(`Строка ${row.row_id}: проверьте ${label} «${row.cells[key].text}» — недопустимый формат числа.`);if(row.kind==='tax')hasVat=true;if(ok){sum+=q*p;if(row.kind==='tax')vat+=q*p;else products+=q*p}});current.total=$('total').value;current.computed_total=sum.toFixed(2);let total=num(current.total);current.math_match=valid&&total!==null&&Math.abs(total-sum)<.011;let vatOk=hasVat&&valid?Math.abs(vat-products*Number(current.vat.rate_percent)/100)<=1:null;current.vat.status=vatOk===null?'not_applicable':vatOk?'match':'mismatch';current.vat.amount=hasVat?vat.toFixed(2):null;const baseWarnings=current.geometry.warnings.filter(w=>!/^Строка \d+: проверьте (количество|цену) /.test(w));const rowTotalsOk=current.rows.every(row=>!row.cells.printed_total||row.printed_line_match===true);const printedVat=num(current.printed_vat);const vatPrintedMatches=printedVat===null?false:hasVat&&Math.abs(vat-printedVat)<=.011;current.vat.printed_match=printedVat===null?null:vatPrintedMatches;const e4enabled=current.numeric_matrix&&current.numeric_matrix.status!=='disabled';const verified=e4enabled?current.math_match&&rowTotalsOk&&(!hasVat||vatPrintedMatches)&&!current.numeric_matrix.review_required:current.math_match&&vatOk===true&&rowTotalsOk;current.ocr_validation_status=verified?'consistent':'review';current.vat.rate_diagnostic_only=true;const visibleWarnings=baseWarnings.filter(w=>!verified||w!=='Есть альтернативные варианты OCR чисел, требующие проверки.');$('warnings').textContent=[...visibleWarnings,...issues].join(' ');$('result').classList.toggle('verified',verified);$('receipt-heading').classList.toggle('receipt-ok',verified);$('receipt-heading').classList.toggle('receipt-mismatch',!verified&&!current.numeric_matrix?.review_required);
const e4status=$('e4-auto-status');
if(e4status){
 const matrix=current.numeric_matrix||{};
 const uncertain=!!matrix.review_required;
 const changes=current.numeric_auto_corrections?.changes?.length||0;
 e4status.textContent=matrix.status==='ambiguous'?'Выбрана комбинация с наибольшей уверенностью OCR. Несколько комбинаций совпадают по суммам — пожалуйста, проверьте цифры.':
   changes?('Система уточнила '+changes+' числовых значений по единственной арифметически согласованной комбинации OCR.'):
   matrix.status==='unique_math_match'?'Сверка по итогам завершена.':'Не удалось выбрать комбинацию чисел. Пожалуйста, проверьте цифры.';
 e4status.hidden=!e4enabled;
}
renderReceiptView();
current.status=(!verified||visibleWarnings.length||issues.length)?'review':'ok';$('sum').textContent=current.computed_total;$('math').textContent=current.math_match?'Сумма совпадает':'Сумма пока не подтверждена автоматически';$('math').className=current.math_match?'':'bad';$('vat').textContent=hasVat?`НДС: ${vat.toFixed(2)} ₸ · ${vatOk===null?'проверьте данные':vatOk?'соответствует ставке':'расхождение со ставкой (возможна ошибка оператора)'}`:'НДС: строка не обнаружена';$('vat').className=(printedVat!==null&&!vatPrintedMatches)?'bad':''}
$('save').onclick=async()=>{if(!current)return;const result={...current};delete result.original;delete result.rectified;try{const response=await connection(`/api/jobs/${current.job_id}/receipt`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(result)});if(!response.ok)throw Error('Не удалось сохранить правки в каталоге задания');const saved=await response.json();if(saved.receipt){Object.assign(result,saved.receipt);Object.assign(current,saved.receipt);render()}$('status').textContent='Правки и пользовательский эталон сохранены'}catch(e){$('status').textContent=e.message+'; JSON будет скачан'}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='receipt.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
