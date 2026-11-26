const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('App/web/app.js', 'utf8');
const render = source.slice(source.indexOf('function renderLiveReceipt('), source.indexOf('function updateDetails('));
const numeric = source.slice(source.indexOf('function num('), source.indexOf('function calculate('));
function element() {
  return {children: [], hidden: false, textContent: '', append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; }};
}
const target = element();
const context = {document: {createElement: element}, $: () => target};
vm.createContext(context);
vm.runInContext('let liveReceiptSeq=0, liveReceiptFooter=null;\n' + numeric + render, context);

context.renderLiveReceipt({live_receipt: {seq: 1, rows: [{row_id: 1, cells: {name: '<товар>'}}]}});
assert.equal(target.hidden, false);
let cells = target.children[1].children[1].children[0].children;
assert.equal(cells[1].textContent, '<товар>');
assert.equal(cells[4].textContent, '—');

context.renderLiveReceipt({live_receipt: {seq: 2, rows: [{row_id: 1, cells: {name: 'Товар', quantity: '2', price: '100'}}]}, live_receipt_footer: '200'});
cells = target.children[1].children[1].children[0].children;
assert.equal(cells[4].textContent, '200.00');
assert.ok(target.children[2].textContent.includes('200'));
context.clearLiveReceipt();
assert.equal(target.hidden, true);
assert.equal(target.children.length, 0);
console.log('Live receipt renders partial rows, calculates and clears');
