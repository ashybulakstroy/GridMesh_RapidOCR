const { chromium } = require('playwright-core');
const path = require('path');
const fs = require('fs');
(async () => {
  const executablePath = process.env.EDGE_EXE || path.join(process.env['ProgramFiles(x86)'], 'Microsoft/Edge/Application/msedge.exe');
  const browser = await chromium.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage({ acceptDownloads: true });
    await page.goto('http://127.0.0.1:8018/');
    await page.locator('#file').setInputFiles(path.join(__dirname, 'receipt_5730.png'));
    await page.locator('#run').click();
    await page.locator('#live:not([hidden])').waitFor({ timeout: 10000 });
    await page.locator('#result:not([hidden])').waitFor({ timeout: 90000 });
    if (await page.locator('#rows tr').count() !== 3) throw Error('Wrong row count');
    if (!/совпадает/i.test(await page.locator('#math').innerText())) throw Error('Math not confirmed');
    const qty = page.locator('#rows tr').first().locator('input').nth(2);
    await qty.fill('3');
    if (!/Проверьте/i.test(await page.locator('#math').innerText())) throw Error('Math not recalculated');
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#save').click();
    const download = await downloadEvent;
    const target = path.join(__dirname, '../../data/output/browser_saved_receipt.json');
    await download.saveAs(target);
    const data = JSON.parse(fs.readFileSync(target, 'utf8'));
    if (data.rows[0].cells.quantity.text !== '3' || data.math_match !== false) throw Error('Saved edits missing');
    const stored = path.join(__dirname, '../..', data.job_directory, 'receipt_edited.json');
    if (JSON.parse(fs.readFileSync(stored, 'utf8')).rows[0].cells.quantity.text !== '3') throw Error('Job folder edits missing');
    if (!data.performance?.samples?.length || !(data.performance.wall_ms > 0)) throw Error('CPU timing missing');
    if (await page.locator('#profile-rows tr').count() < 3) throw Error('Timing journal missing from UI');
    await page.screenshot({ path: path.join(__dirname, '../../data/output/browser_check.png'), fullPage: true });
    // Simulate a clipboard paste event with the same bytes (no OS clipboard mutation).
    await page.reload();
    const image = fs.readFileSync(path.join(__dirname, 'receipt_5730.png')).toString('base64');
    await page.evaluate((base64) => {
      const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
      const transfer = new DataTransfer();
      transfer.items.add(new File([bytes], 'clipboard.png', {type:'image/png'}));
      document.dispatchEvent(new ClipboardEvent('paste', {clipboardData:transfer, bubbles:true}));
    }, image);
    if (!/готово/i.test(await page.locator('#status').innerText())) throw Error('Clipboard paste was not accepted');
    await page.locator('#run').click();
    await page.locator('#result:not([hidden])').waitFor({ timeout: 90000 });
    if (await page.locator('#rows tr').count() !== 3) throw Error('Clipboard OCR row count');
    await page.route('**/api/jobs', route => route.abort());
    await page.locator('#run').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('Нет связи'));
    console.log('Browser: file, simulated clipboard paste, OCR, edit, JSON, timing, connection error passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
