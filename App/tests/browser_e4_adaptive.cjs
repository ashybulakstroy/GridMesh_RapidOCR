const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
(async()=>{
  const root=path.join(__dirname,'../..');
  const browser=await chromium.launch({headless:true,executablePath:process.env.EDGE_EXE||path.join(process.env['ProgramFiles(x86)'],'Microsoft/Edge/Application/msedge.exe')});
  try{
    const page=await browser.newPage({viewport:{width:1500,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const rows=[{row_id:1,kind:'product',cells:{number:{text:'1'},name:{text:'Товар'},quantity:{text:'2'},price:{text:'100'}}},
                {row_id:2,kind:'tax',cells:{number:{text:''},name:{text:'НДС 16 %'},quantity:{text:'32'},price:{text:'1'}}}];
    const early={job_id:'early-test',elapsed_ms:400,rows,total:'232',printed_vat:'32',provisional:true,read_only:true};
    const finalRows=structuredClone(rows);finalRows[0].cells.price.text='101';finalRows[1].cells.quantity.text='32.32';
    for(const row of finalRows)for(const cell of Object.values(row.cells))cell.confidence=.95;
    const png='data:image/png;base64,'+fs.readFileSync(path.join(__dirname,'fixtures/price_1120_border.png')).toString('base64');
    const final={rows:finalRows,original:png,rectified:png,total:'234.32',printed_vat:'32.32',vat:{rate_percent:16},
      geometry:{rectification:'original_cells',warnings:[]},job_directory:'test',timings:{geometry_s:.2,ocr_s:.6,total_s:.8},
      performance:{wall_ms:800,process_cpu_ms:1000,cpu_core_equivalents:1.25,peak_process_cpu_percent:200,peak_system_cpu_percent:20,samples:[],cpu_history:[]},
      numeric_matrix:{status:'ambiguous',review_required:true},
      e4_early_comparison:{changed_count:4,final_review_required:true,changed_fields:[{row_id:1,field:'price',before:'100',after:'101'}]}};
    let state='running';
    await page.route('http://receipt.test/**',async route=>{
      const url=new URL(route.request().url());
      if(url.pathname==='/api/jobs')return route.fulfill({json:{id:'early-test'}});
      if(url.pathname==='/api/jobs/early-test')return route.fulfill({json:state==='done'?{status:'done',result:final}:
        state==='failed'?{status:'failed',error:'Тестовый сбой полной проверки'}:
        {status:'running',progress:{elapsed_s:.4,e4_adaptive_enabled:true,e4_early_ready:true,e4_first_result:early}}});
      const rel=url.pathname==='/'?'App/web/index.html':'App/web/'+path.basename(url.pathname);
      return route.fulfill({path:path.join(root,rel)});
    });
    await page.goto('http://receipt.test/');
    await page.locator('#file').setInputFiles(path.join(__dirname,'fixtures/price_1120_border.png'));
    await page.locator('#run').click();
    await page.locator('#e4-early-preview').waitFor();
    assert.equal(await page.locator('#result').isVisible(),false);
    assert.equal(await page.locator('#e4-early-preview input,#e4-early-preview select').count(),0);
    assert.match(await page.locator('#e4-early-preview').innerText(),/математически согласован/i);
    assert.equal(await page.locator('#receipt-heading').evaluate(e=>e.classList.contains('receipt-ok')),false);
    assert.equal(await page.locator('#e4-early-preview a').getAttribute('href'),'/api/jobs/early-test/early');
    const directory=path.join(root,'data/experiment_reports/e4_adaptive_browser');fs.mkdirSync(directory,{recursive:true});
    await page.screenshot({path:path.join(directory,'early_preview.png'),fullPage:true});
    state='done';await page.locator('#result:not([hidden])').waitFor();
    assert.match(await page.locator('#e4-early-status').innerText(),/изменено полей — 4/);
    assert.match(await page.locator('#e4-early-preview').innerText(),/100/);
    assert.equal(await page.locator('#rows tr').first().locator('input[data-field=price]').inputValue(),'101');
    assert.match(await page.locator('#e4-early-preview').innerText(),/неоднозначные цифры/);
    await page.screenshot({path:path.join(directory,'final_comparison.png'),fullPage:true});
    state='running';await page.locator('#run').click();await page.locator('#e4-early-status').filter({hasText:'Фоновая проверка продолжается'}).waitFor();
    state='failed';await page.locator('#status').filter({hasText:'Тестовый сбой'}).waitFor();
    assert.match(await page.locator('#e4-early-status').innerText(),/не подтверждён/);
    assert.equal(await page.locator('#e4-early-preview').count(),1);
    assert.deepEqual(errors,[]);
    console.log('E4 browser: read-only preview, final differences, reset, and failure status passed');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
