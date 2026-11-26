'use strict';
// Optional UI experiment: no modifications to OCR job or result data.
(()=>{
 const FRAME_MS=200, COUNT=42;
 let worker=null,frames=new Map(),names=[],generation=0,raf=0,current=-1,shownAt=null,done=false,active=false,lastFile=null,displayed=0;
 let original,host,canvas,caption,button,observer;
 function init(){
  if(host)return true;
  original=document.getElementById('original');if(!original)return false;
  host=document.createElement('div');host.className='ocr-math-host';original.before(host);host.append(original);
  canvas=document.createElement('canvas');canvas.className='ocr-math-canvas';canvas.hidden=true;canvas.setAttribute('aria-hidden','true');host.append(canvas);
  const controls=document.createElement('div');controls.className='ocr-math-controls';controls.hidden=true;
  caption=document.createElement('span');caption.setAttribute('aria-live','off');
  button=document.createElement('button');button.type='button';button.textContent='Оригинал';button.onclick=()=>stop();controls.append(caption,button);(host.closest('#original-photo-viewport')||host).after(controls);
  observer=new ResizeObserver(sync);observer.observe(original);observer.observe(host);original.addEventListener('load',sync);window.addEventListener('resize',sync);
  return true;
 }
 function sync(){if(!host||canvas.hidden)return;const a=original.getBoundingClientRect(),b=host.getBoundingClientRect();canvas.style.left=(a.left-b.left)+'px';canvas.style.top=(a.top-b.top)+'px';canvas.style.width=a.width+'px';canvas.style.height=a.height+'px';}
 function clearWorker(){if(worker){worker.terminate();worker=null;}for(const bitmap of frames.values())bitmap.close();frames.clear();}
 function stop(resetFile=false){generation++;cancelAnimationFrame(raf);active=false;clearWorker();if(resetFile)lastFile=null;if(host){canvas.hidden=true;canvas.width=1;canvas.height=1;caption.parentElement.hidden=true;host.dataset.mathState='idle';} }
 function fail(message){stop();console.warn('[OCRMathPreview]',message);}
 function start(file){
  if(!file){stop(true);return;}
  if(file===lastFile)return; // queued clipboard photo is selected again by existing app code
  stop();lastFile=file;
  if(!window.Worker||!window.OffscreenCanvas||!window.createImageBitmap||!window.ResizeObserver)return;
  try{if(!init())return;}catch(e){fail(e);return;}
  original.closest('#original-photo-viewport')?.classList.remove('zoom-pulse-active');
  const token=generation;active=true;current=-1;shownAt=null;done=false;displayed=0;names=[];caption.parentElement.hidden=false;caption.textContent='Подготовка визуализации…';
  try{worker=new Worker('/static/ocr_math_worker.js');}catch(e){fail(e);return;}
  worker.onmessage=e=>{
    const d=e.data;if(token!==generation){d.bitmap?.close();return;}
    if(d.type==='meta')names=d.names;
    else if(d.type==='frame')frames.set(d.index,d.bitmap);
    else if(d.type==='done'){done=true;worker?.terminate();worker=null;}
    else if(d.type==='error')fail(d.message);
  };
  worker.onerror=e=>{if(token===generation)fail(e.message);};
  worker.postMessage({file});
  function tick(ts){
   if(!active||token!==generation)return;
   if(shownAt===null||ts-shownAt>=FRAME_MS){
    const next=current+1;
    if(next===COUNT){active=false;caption.textContent=`42/42 · визуализация завершена`;host.dataset.mathState='complete';return;}
    const bitmap=frames.get(next);
    if(bitmap){
     canvas.width=bitmap.width;canvas.height=bitmap.height;canvas.getContext('2d').drawImage(bitmap,0,0);bitmap.close();frames.delete(next);
     canvas.hidden=false;sync();current=next;shownAt=ts;displayed++;host.dataset.mathFrame=String(displayed);host.dataset.mathState='playing';caption.textContent=`${displayed}/${COUNT} · ${names[next]||'Обработка'}`;
    }else if(done){fail('Missing frame '+next);return;}
   }
   raf=requestAnimationFrame(tick);
  }
  host.dataset.mathFrame='0';host.dataset.mathState='preparing';raf=requestAnimationFrame(tick);
 }
 window.OCRMathPreview={start,stop:()=>stop(true),isVisible:()=>active||!!(canvas&&!canvas.hidden)};
})();
