const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'../..');
 const browser=await chromium.launch({headless:true,executablePath:path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8018/');
  await page.locator('#file').setInputFiles(path.join(__dirname,'fixtures/receipt_short_51040.jpg'));
  await page.locator('#run').click();
  await page.locator('#result:not([hidden])').waitFor({timeout:60000});
  const r=await page.evaluate(()=>current);
  assert.equal(r.math_match,true);assert.equal(Number(r.total),51040);
  assert.equal(await page.locator('#receipt-editor').evaluate(e=>e.open),false);
  assert.equal(await page.locator('#rows input').first().isVisible(),false);
  assert.equal(await page.locator('#e4-auto-status').isVisible(),false);
  assert.equal(await page.locator('#e4-early-preview').count(),0);
  assert.equal(await page.locator('#receipt-view input').count(),0);
  assert.match(await page.locator('#receipt-view').innerText(),/51040/);
  assert(r.problem_digits);
  const dir=path.join(root,r.job_directory);
  const manifest=JSON.parse(fs.readFileSync(path.join(dir,r.problem_digits.manifest),'utf8'));
  for(const change of r.numeric_auto_corrections.changes){
   const cell=manifest.cells.find(c=>c.row_id===change.row_id&&c.field===change.field);
   assert(cell);assert.equal(cell.before,change.before);assert.equal(cell.after,change.after);
   assert(Object.keys(cell.images).length>0);
  }
  for(const cell of manifest.cells)for(const image of Object.values(cell.images))assert(fs.existsSync(path.join(dir,'problem_digits',image)));
  await page.evaluate(()=>profileRows([
    {name:'Быстро',wall_ms:999,process_cpu_ms:0},
    {name:'Ровно секунда',wall_ms:1000,process_cpu_ms:0},
    {name:'Медленно',wall_ms:1001,process_cpu_ms:0}]));
  assert.equal(await page.locator('#profile-rows .timing-slow').count(),1);
  assert.match(await page.locator('#profile-rows .timing-slow').innerText(),/Медленно/);
  assert.equal(await page.locator('#profile-rows .timing-slow').evaluate(e=>getComputedStyle(e).backgroundColor),'rgb(255, 241, 168)');
  const out=path.join(root,'data/experiment_reports/receipt_editor');fs.mkdirSync(out,{recursive:true});
  await page.screenshot({path:path.join(out,'collapsed.png'),fullPage:true});
  await page.locator('#receipt-editor summary').click();
  assert.equal(await page.locator('#rows input').first().isVisible(),true);
  await page.locator('#total').fill('1');await page.locator('#total').dispatchEvent('input');
  await page.evaluate(()=>render());
  assert.equal(await page.locator('#receipt-editor').evaluate(e=>e.open),true);
  assert.equal(await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-mismatch')),true);
  await page.screenshot({path:path.join(out,'mismatch_open.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({job:r.job_id,problem_digits:r.problem_digits,changes:r.numeric_auto_corrections.changes,errors},null,2));
  console.log('PASS: verified editor collapsed, manual expand, mismatch open, >1s yellow, debug photos verified. Job '+r.job_id);
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
