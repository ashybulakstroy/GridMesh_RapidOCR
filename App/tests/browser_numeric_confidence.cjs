const { chromium } = require('playwright-core');
const path = require('path');
(async () => {
  const executablePath = process.env.EDGE_EXE || path.join(process.env['ProgramFiles(x86)'], 'Microsoft/Edge/Application/msedge.exe');
  const browser = await chromium.launch({headless:true,executablePath});
  try {
    const page = await browser.newPage({viewport:{width:1600,height:1000}});
    await page.goto('http://127.0.0.1:8018/');
    await page.locator('#file').setInputFiles(path.join(__dirname,'fixtures/receipt_29464.jpg'));
    await page.locator('#run').click();
    await page.locator('#result:not([hidden])').waitFor({timeout:90000});
    const row = page.locator('#rows tr').nth(8);
    if (!/^1120/.test(await row.locator('input[data-field=price]').inputValue())) throw Error('Leading 1 lost');
    if ((await page.locator('#sum').innerText()) !== '29464.00') throw Error('Receipt sum mismatch');
    if (await row.locator('.ocr-confidence').isVisible()) throw Error('Balanced receipt shows confidence');
    await page.locator('#total').fill('28464');
    const scores = await page.evaluate(() => Object.fromEntries(['name','quantity','price'].map(k=>[k,current.rows[8].cells[k].confidence])));
    for (const [field,label] of [['name','Название'],['quantity','Кол-во'],['price','Цена']]) {
      const actual = await row.locator(`.ocr-confidence [data-field=${field}]`).innerText();
      if (actual !== `${label}: ${Math.round(scores[field]*100)}%`) throw Error(`Wrong ${field} confidence: ${actual}`);
    }
    await page.evaluate(() => {current.rows[8].cells.price.confidence=.59;render();});
    if (!(await row.locator('input[data-field=price]').getAttribute('class')||'').includes('low-confidence')) throw Error('Low price confidence not highlighted');
    if ((await row.locator('.ocr-confidence [data-field=price]').innerText()) !== 'Цена: 59%') throw Error('Price confidence uses name score');
    await page.evaluate(score => {current.rows[8].cells.price.confidence=score;render();},scores.price);
    await page.locator('#total').fill('29464');
    await page.screenshot({path:path.join(__dirname,'../../data/output/numeric_confidence_browser.png'),fullPage:true});
    console.log('Browser: 1120, 29464, independent field confidence and low numeric confidence passed');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
