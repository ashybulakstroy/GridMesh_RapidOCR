/* Display-only subscriber to the existing /api/jobs/{id} AJAX response. */
(function(global){
  'use strict';
  function controller(doc, clock=global){
    let host=null, disc=null, label=null, search=null, amount=null, meta=null, pulse=null, timer=null;
    let lastSeq=0, previous=null, visible=false, terminal=false, jobActive=false;
    const fmt=new Intl.NumberFormat('ru-KZ',{minimumFractionDigits:2, maximumFractionDigits:2});
    function mount(){
      if(host)return;
      host=doc.createElement('div');host.id='live-total-ring';host.hidden=true;host.setAttribute('role','status');host.setAttribute('aria-live','polite');
      disc=doc.createElement('div');disc.className='ltr-disc';
      label=doc.createElement('div');label.className='ltr-label';
      search=doc.createElement('div');search.className='ltr-search';search.setAttribute('aria-hidden','true');
      amount=doc.createElement('div');amount.className='ltr-amount';
      meta=doc.createElement('div');meta.className='ltr-meta';
      disc.append(label,search,amount,meta);host.append(disc);
      for(let i=0;i<14;i++){
        const p=doc.createElement('i');p.className='ltr-particle';p.setAttribute('aria-hidden','true');
        const a=i*2*Math.PI/14,r=110+(i%3)*17;
        p.style.setProperty('--dx',`${Math.cos(a)*r}px`);p.style.setProperty('--dy',`${Math.sin(a)*r}px`);p.style.setProperty('--rot',`${i*57}deg`);
        p.style.setProperty('--pcolor',['#16a34a','#f5b83d','#24b8e0','#77db62'][i%4]);host.append(p);
      }
      doc.body.append(host);
    }
    function place(){
      if(!host)return;
      const box=doc.querySelector('.result-panel')?.getBoundingClientRect();
      if(box){const w=global.innerWidth||1024;host.style.setProperty('--ltr-x',`${Math.max(100,Math.min(w-100,(box.left+box.right)/2))}px`);}
    }
    function stopPulse(){if(pulse){pulse.cancel();pulse=null;}}
    function hide(){if(timer!==null){clock.clearTimeout(timer);timer=null;}stopPulse();if(host)host.hidden=true;visible=false;}
    function reset(){hide();lastSeq=0;previous=null;terminal=false;jobActive=true;if(host)host.dataset.state='scanning';}
    function grow(){
      if(!disc?.animate||global.matchMedia?.('(prefers-reduced-motion: reduce)').matches)return;
      stopPulse();pulse=disc.animate([{transform:'scale(1)'},{transform:'scale(1.3)',offset:.38},{transform:'scale(1)'}],{duration:300,easing:'ease-in-out'});
    }
    function show(value, stage, ready, expected){
      mount();place();host.hidden=false;visible=true;host.dataset.state=stage;
      amount.textContent=`${fmt.format(value)} ₸`;
      if(stage==='success'){label.textContent='Сумма совпала';meta.textContent='✓ Проверено';}
      else if(stage==='mismatch'){label.textContent='Сумма не совпала';meta.textContent='Проверьте цифры';}
      else if(stage==='unknown'){label.textContent='Итог не подтверждён';meta.textContent='Проверьте цифры';}
      else {label.textContent='Сумма строк';meta.textContent=Number.isInteger(ready)&&Number.isInteger(expected)?`${ready} / ${expected} строк`:'Проверка…';}
    }
    function update(progress){
      if(!jobActive||terminal)return;
      const phase=progress?.phase||'';
      const searching=phase==='Поиск физической сетки',building=phase==='Создание сетки Grid Mesh';
      if(progress?.live_total_ring_enabled&&(searching||building)){
        mount();place();stopPulse();host.hidden=false;visible=true;host.dataset.state='geometry';
        label.textContent=searching?'Поиск сетки':'Создание сетки';
        return;
      }
      if(host?.dataset.state==='geometry')hide();
      const s=progress?.live_total_ring;
      if(!s?.active||s.version!==1||!Number.isInteger(s.seq)||s.seq<=lastSeq)return;
      const value=Number(s.amount);if(!Number.isFinite(value))return;
      lastSeq=s.seq;const increase=previous!==null?value>previous:value>0;
      show(value,'scanning',s.ready_rows,s.expected_rows);if(increase)grow();previous=value;
    }
    function finish(result){
      if(!jobActive||terminal||!visible)return;
      if(previous===null){hide();jobActive=false;return;}
      terminal=true;stopPulse();
      const value=Number(result?.computed_total),known=result?.total!==null&&result?.total!==undefined&&String(result.total).trim()!=='';
      const state=result?.math_match===true?'success':known?'mismatch':'unknown';
      show(Number.isFinite(value)?value:(previous??0),state);
      timer=clock.setTimeout(()=>{timer=null;hide();jobActive=false;},1000);
    }
    function fail(){terminal=true;jobActive=false;hide();}
    global.addEventListener?.('resize',place);
    return {reset,update,finish,fail};
  }
  if(typeof module!=='undefined'&&module.exports)module.exports={controller};
  else global.LiveTotalRing=controller(document);
})(typeof window!=='undefined'?window:globalThis);
