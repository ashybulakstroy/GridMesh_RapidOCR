const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage();
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(root,'data/output/11f586984e7d4dc49693aea8114c55c4/input.bin'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert(r.math_match);assert.equal(Number(r.total),236640);
  for(const row of r.rows.slice(0,2)){
   assert.equal(row.cells.price.text,'9000');
   assert(row.cells.price.numeric_roi.currency_removed_before_ocr);
   assert.equal(row.cells.price.numeric_roi.ocr_roi_bbox[2],72);
  }
  const dir=path.join(root,r.job_directory,'problem_digits');
  const manifest=JSON.parse(fs.readFileSync(path.join(dir,'manifest.json'),'utf8'));
  const cell=manifest.cells.find(x=>x.row_id===1&&x.field==='price');
  assert(cell.images.primary_ocr_input);assert(fs.existsSync(path.join(dir,cell.images.primary_ocr_input)));
  fs.writeFileSync(path.join(root,'data/experiment_reports/currency_9000_fix/browser_result.json'),JSON.stringify(r,null,2));
  console.log('PASS actual web, currency excluded, totals match; report: '+path.join(dir,'report.html'));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
