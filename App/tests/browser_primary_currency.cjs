const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(root,'data/output/1cf81ccb658e4614ab900cecdc746e5c/original.png'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(r.math_match,true);
  assert.deepEqual(r.rows.map(x=>x.cells.price.text),['4810','1']);
  assert(r.rows.every(x=>x.cells.price.numeric_roi.currency_removed_before_ocr));
  assert.deepEqual(errors,[]);
  const out=path.join(root,'data/experiment_reports/primary_currency_crop');
  await page.screenshot({path:path.join(out,'browser_final.png'),fullPage:true});
  fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(r,null,2));
  console.log('PASS: web prices 4810 and 1; currency excluded before OCR; math matches; no JS errors.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
