const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
(async()=>{
 const browser = await chromium.launch({headless:true,executablePath:process.env.EDGE_EXE||path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  const root=path.join(__dirname,'../..');
  const img='data:image/png;base64,'+fs.readFileSync(path.join(__dirname,'fixtures/price_1120_border.png')).toString('base64');
  const warning='Есть альтернативные варианты OCR чисел, требующие проверки.';
  const prices=[1980,2350,8380,550,470,980,560,570,1120,2400,3180,2300,1];
  const data={original:img,rectified:img,total:'29464',job_directory:'test',problem_text:{suspects:2},
    rows:prices.map((price,i)=>({row_id:i+1,kind:i===12?'tax':'product',cells:{
      number:{text:i===12?'':String(i+1),derived:true},name:{text:i===12?'НДС 16%':`Товар ${i+1}`,confidence:.95},
      quantity:{text:String(i===12?4064:i===6?2:1),confidence:.96},price:{text:String(price),confidence:.91}}})),
    geometry:{warnings:[warning],rectification:'original_cells'},vat:{rate_percent:16},
    processing_log_file:'data/output/test/processing_log.json',
    processing_log:[{type:'budget',message:'Подготовка: превышен лимит'},
                    {type:'retry',message:'Повтор с удвоенным бюджетом: 120 → 1120'}],
    timings:{geometry_s:.4,ocr_s:1.3,total_s:1.8},performance:{wall_ms:1800,process_cpu_ms:4000,
      cpu_core_equivalents:2.2,peak_process_cpu_percent:300,peak_system_cpu_percent:40,samples:[],cpu_history:[]}};
  await page.route('**/api/jobs',r=>r.fulfill({json:{id:'display-test'}}));
  await page.route('**/api/jobs/display-test',r=>r.fulfill({json:{status:'done',result:data}}));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(__dirname,'fixtures/price_1120_border.png'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor();
  if(!await page.locator('#text-review').isVisible() || !(await page.locator('#text-review').innerText()).includes('2 строк'))throw Error('Text review notice missing');
  if(await page.locator('#text-review a').count())throw Error('Unexpected text report link');
  await page.evaluate(()=>{current.problem_text=null;render();});
  if(await page.locator('#text-review').isVisible())throw Error('Text review notice did not clear');
  if(!await page.locator('#correction-log').isVisible())throw Error('Processing log missing');
  if(!/120 → 1120/.test(await page.locator('#correction-log').innerText()))throw Error('Retry details missing');
  if(!/processing_log.json/.test(await page.locator('#correction-log-file').innerText()))throw Error('Processing log file missing');
  if(!await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-ok')))throw Error('Balanced receipt is not green');
  if(await page.locator('.verification-detail:visible').count())throw Error('Balanced receipt shows verification columns');
  if((await page.locator('#warnings').textContent()).includes(warning))throw Error('Balanced receipt shows alternative warning');
  const qty=page.locator('#rows tr').first().locator('input[data-field=quantity]');
  const old=await qty.inputValue();await qty.fill(String(Number(old)+1));
  if(!await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-mismatch')))throw Error('Mismatch is not pink');
  if(!await page.locator('.verification-detail:visible').count())throw Error('Mismatch hides verification columns');
  if(!(await page.locator('#warnings').textContent()).includes(warning))throw Error('Mismatch fails to restore warning');
  await qty.fill(old);
  await page.evaluate(()=>{current.rows.at(-1).cells.quantity.text='4164';render();document.querySelector('#total').value=current.computed_total;calculate();});
  if(!await page.evaluate(()=>current.math_match&&current.vat.status==='mismatch'))throw Error('VAT mismatch fixture invalid');
  if(!await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-mismatch')))throw Error('VAT mismatch displayed green');
  await page.evaluate(()=>{current.rows.at(-1).cells.quantity.text='4064';render();document.querySelector('#total').value='29464';calculate();current.rows[0].cells.printed_total={text:'999',confidence:1};calculate();});
  if(!await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-mismatch')))throw Error('Printed row mismatch displayed green');
  await page.evaluate(()=>{delete current.rows[0].cells.printed_total;calculate();current.geometry.warnings.push('Проверьте границы сетки.');calculate();});
  if(!(await page.locator('#warnings').textContent()).includes('границы'))throw Error('Geometry warning was suppressed');
  await page.evaluate(()=>cpuGraph([{elapsed_s:.2,system_cpu_percent:49},{elapsed_s:.4,system_cpu_percent:50},{elapsed_s:.6,system_cpu_percent:79},{elapsed_s:.8,system_cpu_percent:80},{elapsed_s:1,system_cpu_percent:100}],true,1.2));
  const bars=await page.locator('.cpu-bar').evaluateAll(es=>es.map(e=>({interval:e.dataset.interval,cpu:e.dataset.cpu,missing:e.dataset.missing,cls:e.className,text:e.textContent})));
  if(bars.length!==6||bars[0].interval!=='5'||bars[1].cpu!=='100'||bars.at(-1).cpu!=='49')throw Error('CPU timeline must be newest-left');
  for(const [index,cls]of [[1,'cpu-high'],[2,'cpu-high'],[3,'cpu-medium'],[4,'cpu-medium'],[5,'cpu-low']])if(!bars[index].cls.includes(cls))throw Error('CPU color threshold');
  if((await page.locator('#cpu-graph-bars').textContent()).includes('·'))throw Error('Graph contains dots');
  const heading=await page.locator('#receipt-heading').boundingBox(),graph=await page.locator('.cpu-graph').boundingBox();
  if(graph.x<heading.x+heading.width)throw Error('Graph must start to the right of heading');
  await page.screenshot({path:path.join(root,'data/output/display_states_browser.png'),fullPage:true});
  console.log('Browser: balanced/mismatch/VAT/row totals, warning filtering, columns, CPU direction/colors/gaps passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
