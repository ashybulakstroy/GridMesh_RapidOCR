const { chromium } = require('playwright-core');
const path = require('path');
const { pathToFileURL } = require('url');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.EDGE_EXE||path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage({viewport:{width:1500,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(__dirname,'../../docs/CPU_THREADS_EXPERIMENT.html')).href);
  const data=JSON.parse(await page.locator('#experiment-data').textContent());
  if(data.length!==9||data.reduce((n,p)=>n+p.records.filter(r=>!r.warmup).length,0)!==81)throw Error('Report data incomplete');
  if(await page.locator('.source:visible').count()!==3)throw Error('Report source panels missing');
  await page.locator('#source-filter').selectOption('long');
  if(await page.locator('.source:visible').count()!==1||!await page.locator('.source[data-source=long]').isVisible())throw Error('Report filter failed');
  await page.locator('#source-filter').selectOption('all');
  if(errors.length)throw Error(errors.join('; '));
  await page.screenshot({path:path.join(__dirname,'../../data/output/cpu_threads_experiment/report_preview.png')});
  console.log('HTML report: nine profiles, 81 measured runs, three receipts, filtering passed; no new benchmarks');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
