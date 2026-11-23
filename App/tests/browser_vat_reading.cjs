const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(root,'data/output/4bdec50f6a5e4396ac77fd18184c894a/input.bin'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(r.math_match,true);assert.equal(Number(r.printed_vat),23200);
  assert.equal(Number(r.total),168200);assert.equal(r.vat_evidence.source.field,'quantity');
  assert.equal(await page.locator('#vat-printed').inputValue(),'23200');
  assert(!r.geometry.warnings.some(x=>x.includes('Нет независимого OCR-подтверждения суммы НДС')));
  assert.deepEqual(errors,[]);
  const out=path.join(root,'data/experiment_reports/vat_reading_fix');
  await page.screenshot({path:path.join(out,'browser_final.png'),fullPage:true});
  fs.writeFileSync(path.join(out,'browser_result.json'),JSON.stringify(r,null,2));
  console.log('PASS: browser printed VAT=23200, total=168200, source=quantity, no JS errors.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
