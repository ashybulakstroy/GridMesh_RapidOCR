const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('App/web/app.js', 'utf8');
const code = source.slice(source.indexOf('function updateTextReview'), source.indexOf('function cpuGraph'));
const elements = {'correction-log-entries':{replaceChildren(){},append(){},childElementCount:0},
  'correction-log-file':{},'correction-log':{},'text-review':{}};
const context = {current:{problem_text:{suspects:5}},document:{createElement(){return {}}},
  $:id=>elements[id]};
vm.createContext(context);
vm.runInContext(code,context);
context.renderCorrectionLog([]);
if(elements['text-review'].hidden || !elements['text-review'].textContent.includes('5 строках')) throw Error('Text notice missing');
context.current.problem_text=null;
context.renderCorrectionLog([]);
if(!elements['text-review'].hidden) throw Error('Text notice not cleared');
context.current={rows:[{kind:'product',cells:{name:{text_review_required:true}}}],problem_text:null};
context.renderCorrectionLog([]);
if(elements['text-review'].hidden || !elements['text-review'].textContent.includes('1 строках')) throw Error('Notice missing when report disabled');
console.log('Text review notice shown and cleared');
