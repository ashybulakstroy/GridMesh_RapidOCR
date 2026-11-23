const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(root,'data/output/3a93a24ef037459a84323334ed272200/input.bin'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(r.rows.length,3);assert.equal(r.rows.filter(x=>x.kind==='product').length,2);
  assert.deepEqual(r.rows.map(x=>x.cells.quantity.text),['1','1','36064']);
  assert.deepEqual(r.rows.map(x=>x.cells.price.text),['112700','112700','1']);
  assert.equal(Number(r.total),261464);assert(r.math_match);
  assert.equal(await page.locator('#receipt-view tbody tr').count(),3);assert.deepEqual(errors,[]);
  const out=path.join(root,'data/experiment_reports/grid_speed_v3');
  fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(r,null,2));
  await page.screenshot({path:path.join(out,'browser_final.png'),fullPage:true});
  console.log('PASS web: two doors plus VAT, total 261464, math matches; job '+r.job_id);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
