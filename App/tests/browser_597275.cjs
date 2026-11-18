const { chromium } = require('playwright-core');
const path = require('path');
(async () => {
  const executablePath = process.env.EDGE_EXE || path.join(process.env['ProgramFiles(x86)'], 'Microsoft/Edge/Application/msedge.exe');
  const browser = await chromium.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage();
    await page.goto('http://127.0.0.1:8018/');
    await page.locator('#file').setInputFiles(path.join(__dirname, 'receipt_597275.png'));
    await page.locator('#run').click();
    await page.locator('#result:not([hidden])').waitFor({ timeout: 90000 });
    const rows = page.locator('#rows tr');
    if (await rows.count() !== 3 || await rows.nth(2).locator('select').inputValue() !== 'tax') throw Error('VAT row missing');
    if (!/совпадает/i.test(await page.locator('#math').innerText())) throw Error('Math mismatch');
    if ((await page.locator('#total').inputValue()) !== '23200.00') throw Error('Printed total mismatch');
    console.log('UI OK:',await page.locator('#times').innerText(),';',await page.locator('#cpu-summary').innerText());
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
