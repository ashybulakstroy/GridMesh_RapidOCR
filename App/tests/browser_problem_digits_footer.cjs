const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  const old=path.join(root,'data/output/9476a64f9e8340e5b719bc7f87f703a8');
  await page.locator('#file').setInputFiles(path.join(old,'input.bin'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(Number(r.total),90665);assert(r.math_match);
  const before=JSON.parse(fs.readFileSync(path.join(old,'receipt.json'),'utf8'));
  const fields=x=>x.rows.map(row=>['name','quantity','price'].map(f=>row.cells[f].text));
  assert.deepEqual(fields(r),fields(before));
  const dir=path.join(root,r.job_directory,'problem_digits');
  const m=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
  const total=m.cells.find(x=>x.field==='receipt_total');assert(total);
  assert(total.recognition_attempts.length>=4);assert(total.footer_regions.length>0);
  for(const e of total.recognition_attempts){
   assert(e.model_id);assert(Number.isFinite(e.wall_ms));assert(e.texts);assert(e.input_image);
   assert(fs.existsSync(path.join(dir,e.input_image)));
  }
  assert(r.processing_log.some(x=>x.field==='receipt_total'&&x.message.includes('мс')));
  assert((await page.locator('#correction-log').innerText()).includes('v5_cyrillic_mobile'));
  assert.deepEqual(errors,[]);
  console.log('PASS footer conflicts, photos, models, times, trace outputs, unchanged receipt; job '+r.job_id);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
