const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(root,'data/output/11e98825dd84492ab54622f0e1cd6305/input.bin'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(Number(r.total),90665);assert(r.math_match);assert(r.footer_focus.accepted);
  const primary=r.rec_trace.events.filter(e=>e.cells.includes('receipt_total')&&e.stage.startsWith('printed_footer_'));
  assert.equal(primary.length,1);assert.equal(primary[0].stage,'printed_footer_focused');
  assert.equal(r.footer_ocr_attempts.length,1);
  assert(!r.geometry.warnings.some(x=>x.includes('OCR печатного итога дал разные варианты')));
  assert.deepEqual(errors,[]);
  const out=path.join(root,'data/experiment_reports/footer_line_guided');
  fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(r,null,2));
  await page.screenshot({path:path.join(out,'browser_final.png'),fullPage:true});
  console.log('PASS browser: 90665; one focused OCR call; math matches; no alternative-total warning; job '+r.job_id);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
