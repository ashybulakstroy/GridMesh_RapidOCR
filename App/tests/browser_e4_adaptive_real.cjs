// Run against an isolated server, e.g. python -m uvicorn App.web_server:app --port 8028.
const {chromium}=require('playwright-core');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
  const root=path.resolve(__dirname,'../..');
  const browser=await chromium.launch({headless:true,executablePath:process.env.EDGE_EXE||path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1100}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const directory=path.join(root,'data/experiment_reports/e4_adaptive_browser');fs.mkdirSync(directory,{recursive:true});
    await page.goto(process.env.E4_TEST_URL||'http://127.0.0.1:8028/');
    await page.locator('#file').setInputFiles(path.join(__dirname,'fixtures/receipt_253309.jpg'));
    await page.locator('#run').click();
    await page.locator('#e4-early-preview').waitFor({timeout:60000});
    const beforeDone=await page.locator('#result').isHidden();
    await page.screenshot({path:path.join(directory,'real_early.png'),fullPage:true});
    await page.locator('#result:not([hidden])').waitFor({timeout:60000});
    const result=await page.evaluate(()=>({total:current.total,computed:current.computed_total,
      adaptive:current.e4_adaptive,comparison:current.e4_early_comparison,math:current.math_match,job:current.job_id}));
    assert.equal(beforeDone,true);assert.equal(result.adaptive.enabled,true);
    assert.equal(result.math,true);assert.equal(Number(result.total),253309);
    assert.equal(result.comparison.changed_count,0);
    assert.equal(await page.locator('#e4-early-preview input').count(),0);assert.deepEqual(errors,[]);
    await page.screenshot({path:path.join(directory,'real_final.png'),fullPage:true});
    fs.writeFileSync(path.join(directory,'real_browser_result.json'),JSON.stringify({...result,previewVisibleBeforeFinal:beforeDone,errors},null,2));
    console.log(JSON.stringify(result));
    console.log('Real browser/API: early preview before final; immutable values; total 253309; no page errors.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
