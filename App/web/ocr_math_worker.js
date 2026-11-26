'use strict';
// v9 static visual lab; offline worker, no OCR requests.
const document={createElement(tag){if(tag!=='canvas')throw Error('Unsupported element');return new OffscreenCanvas(1,1);}};
let sourceCanvas=null,processedVariants=[];
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const TAU=Math.PI*2;
const variants=[
  {name:'01 · Table ROI / Geometry', desc:'Поиск длинных линий и границ ROI таблицы.', sourceIndex:10},
  {name:'02 · Rectified Diagnostic', desc:'Линии, узлы сетки, кандидаты ячеек и диагностика.', sourceIndex:14},
  {name:'03 · Paper Locator', desc:'Кандидат области бумаги по яркой связной компоненте.', sourceIndex:15},
  {name:'04 · Adaptive Gaussian Threshold', desc:'Локальная гауссова бинаризация.', sourceIndex:16},
  {name:'05 · Morphological Opening', desc:'Горизонтальные и вертикальные штрихи после opening.', sourceIndex:17},
  {name:'06 · Ridge Response', desc:'Карта отклика тёмных линий.', sourceIndex:18},
  {name:'07 · Hough Line Search', desc:'Поиск длинных линий голосованием по углам.', sourceIndex:19},
  {name:'08 · Physical Grid Graph', desc:'Узлы и рёбра наблюдаемой структуры.', sourceIndex:20},
  {name:'09 · Polynomial Fitting', desc:'Квадратичная аппроксимация по измеренным пикселям.', sourceIndex:21},
  {name:'10 · GridRefine', desc:'Сравнение исходных и уточнённых линий.', sourceIndex:22},
  {name:'11 · Edge Recovery / Completeness', desc:'Пробелы поддержки линии и кандидаты восстановления.', sourceIndex:23},
  {name:'12 · Grid Anomaly / Safety', desc:'Проверка размеров и интервалов ячеек.', sourceIndex:24},
  {name:'13 · Mesh Mapping', desc:'Координатная карта четырёхугольных ячеек.', sourceIndex:26},
  {name:'14 · Bicubic Interpolation', desc:'Remapping по координатной карте с кубической интерполяцией.', sourceIndex:28},
  {name:'15 · Numeric ROI / Ink Crop', desc:'Рамки чернильных областей числовых колонок.', sourceIndex:29},
  {name:'16 · Footer Line Guided', desc:'Область поиска печатного итога под таблицей.', sourceIndex:30},
  {name:'17 · CLAHE', desc:'Локальное усиление контраста.', sourceIndex:31},
  {name:'18 · Gaussian Blur', desc:'Подавление мелких шумов размитием.', sourceIndex:32},
  {name:'19 · Otsu Threshold', desc:'Автоматическая бинаризация по Оцу.', sourceIndex:33},
  {name:'20 · Morphological Closing', desc:'Соединение разорванных штрихов.', sourceIndex:34},
  {name:'21 · Dilation', desc:'Расширение чернильных контуров.', sourceIndex:35},
  {name:'22 · Erosion', desc:'Утоньшение чернильных контуров.', sourceIndex:36},
  {name:'23 · Black Hat Transform', desc:'Выделение тёмных печатных элементов.', sourceIndex:37},
  {name:'24 · LSD Detector', desc:'Поиск линейных сегментов.', sourceIndex:38},
  {name:'25 · Connected Components', desc:'Связные компоненты и их цветовая маркировка.', sourceIndex:39},
  {name:'26 · Multiscale Segments', desc:'Линии, обнаруженные на нескольких масштабах.', sourceIndex:40},
  {name:'27 · Ridge Refinement', desc:'Уточнение линии по гребню отклика.', sourceIndex:41},
  {name:'28 · Projection Profiles', desc:'Гистограммы горизонтальной и вертикальной структуры.', sourceIndex:42},
  {name:'29 · Trace Printed Rules V16', desc:'Трассировка печатных линий таблицы.', sourceIndex:43},
  {name:'30 · Perspective Transform', desc:'Перспективное выпрямление внутри ROI.', sourceIndex:44},
  {name:'31 · TPS Transform', desc:'Нелинейное выпрямление сетки.', sourceIndex:45},
  {name:'32 · Fold Recovery', desc:'Поиск и восстановление линии сгиба.', sourceIndex:46},
  {name:'33 · Vertical Rail Recovery', desc:'Восстановление вертикальных границ.', sourceIndex:47},
  {name:'34 · Header Edge Recovery', desc:'Восстановление границ шапки таблицы.', sourceIndex:48},
  {name:'35 · Distance Transform', desc:'Карта расстояний до контуров.', sourceIndex:49},
  {name:'36 · Currency Separation', desc:'Отделение обозначения валюты от числа.', sourceIndex:50},
  {name:'37 · Sobel Gradient', desc:'Карта градиентов изображения.', sourceIndex:51},
  {name:'38 · HOG', desc:'Гистограмма ориентированных градиентов.', sourceIndex:52},
  {name:'39 · Local Binary Patterns', desc:'Карта локальных бинарных шаблонов.', sourceIndex:53},
  {name:'40 · Flood Topology', desc:'Топология замкнутых белых областей.', sourceIndex:54},
  {name:'41 · Shape Context', desc:'Геометрия контуров и радиальные корзины.', sourceIndex:55},
  {name:'42 · Zoning', desc:'Зональный анализ распределения пикселей.', sourceIndex:56}
];
function getGrayStats(){ const c=document.createElement('canvas'); c.width=sourceCanvas.width; c.height=sourceCanvas.height; const g=c.getContext('2d',{willReadFrequently:true}); g.drawImage(sourceCanvas,0,0); const img=g.getImageData(0,0,c.width,c.height), d=img.data; const gray=new Float32Array(c.width*c.height); let mean=0; for(let i=0,j=0;i<d.length;i+=4,j++){ const v=0.299*d[i]+0.587*d[i+1]+0.114*d[i+2]; gray[j]=v; mean+=v; } mean/=gray.length; return {gray,w:c.width,h:c.height,mean}; }
function canvasFromPixels(w,h,painter){ const c=document.createElement('canvas'); c.width=w; c.height=h; const g=c.getContext('2d',{willReadFrequently:true}); const img=g.createImageData(w,h), d=img.data; painter(d); g.putImageData(img,0,0); return c; }
function buildProcessedVariants(){
  processedVariants=[]; const {gray,w,h,mean}=getGrayStats(); const idx=(x,y)=>clamp(y,0,h-1)*w+clamp(x,0,w-1), pix=(x,y)=>gray[idx(x,y)];
  const sobelMag=new Float32Array(w*h), sobelX=new Float32Array(w*h), sobelY=new Float32Array(w*h); let maxMag=1,maxX=1,maxY=1;
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){ const gx=-pix(x-1,y-1)-2*pix(x-1,y)-pix(x-1,y+1)+pix(x+1,y-1)+2*pix(x+1,y)+pix(x+1,y+1); const gy=-pix(x-1,y-1)-2*pix(x,y-1)-pix(x+1,y-1)+pix(x-1,y+1)+2*pix(x,y+1)+pix(x+1,y+1); const m=Math.hypot(gx,gy), i=y*w+x; sobelMag[i]=m; sobelX[i]=Math.abs(gx); sobelY[i]=Math.abs(gy); if(m>maxMag)maxMag=m; if(Math.abs(gx)>maxX)maxX=Math.abs(gx); if(Math.abs(gy)>maxY)maxY=Math.abs(gy);} 
  const thr=Math.max(90,Math.min(180,mean*0.92));
  processedVariants[0]={base:canvasFromPixels(w,h,d=>{for(let i=0,j=0;i<d.length;i+=4,j++){d[i]=d[i+1]=d[i+2]=gray[j];d[i+3]=255;}})};
  processedVariants[2]={base:canvasFromPixels(w,h,d=>{for(let i=0,j=0;i<d.length;i+=4,j++){const v=255-clamp(sobelMag[j]/maxMag*255*2.1,0,255);d[i]=d[i+1]=d[i+2]=v;d[i+3]=255;}})};
  // Search parallel long dark strokes in a range of angles. In the deskewed
  // coordinate system use votes along the lines; no fixed table coordinates.
  function detect(horizontal){
    const major=horizontal?w:h,minor=horizontal?h:w;
    let best={score:0,lines:[],slope:0};
    for(let deg=-10;deg<=10;deg+=1){
      const slope=Math.tan(deg*Math.PI/180), votes=new Float32Array(minor), lo=new Int32Array(minor).fill(major),hi=new Int32Array(minor).fill(-1);
      const stride=Math.max(1,Math.floor(major/550));
      for(let u=1;u<major-1;u+=stride)for(let v=1;v<minor-1;v+=2){
        const x=horizontal?u:v,y=horizontal?v:u,j=y*w+x;
        if(gray[j]>Math.min(155,thr+10)||(horizontal?sobelY[j]:sobelX[j])<75)continue;
        const b=Math.round(v-slope*(u-major/2));if(b<0||b>=minor)continue;
        votes[b]+=stride;lo[b]=Math.min(lo[b],u);hi[b]=Math.max(hi[b],u);
      }
      const peaks=[];for(let b=3;b<minor-3;b++){
        let n=0,l=major,r=-1;for(let k=-2;k<=2;k++){n+=votes[b+k];l=Math.min(l,lo[b+k]);r=Math.max(r,hi[b+k]);}
        if(r-l>major*.24&&n>major*.22)peaks.push({b,l,r,n});
      }
      peaks.sort((a,b)=>b.n-a.n);const chosen=[];
      for(const v of peaks){if(chosen.every(q=>Math.abs(q.b-v.b)>Math.max(7,minor*.012)))chosen.push(v);if(chosen.length>=30)break;}
      const score=chosen.reduce((a,b)=>a+Math.min(major,b.n)*(b.r-b.l)/major,0);
      if(score>best.score)best={score,lines:chosen.sort((a,b)=>a.b-b.b),slope};
    }return best;
  }
  const hh=detect(true),vv=detect(false),hs=hh.lines,vs=vv.lines;
  const good=hs.length>=2&&vs.length>=2;
  function node(H,V){const a=hh.slope,b=vv.slope,A=H.b-a*w/2,B=V.b-b*h/2;const x=(B+b*A)/(1-a*b);return [x,A+a*x];}
  function path(g,pts,close=false){g.beginPath();pts.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));if(close)g.closePath();}
  function poly(g,pts,stroke,fill){if(!pts.length)return;path(g,pts,true);if(fill){g.fillStyle=fill;g.fill();}g.strokeStyle=stroke;g.stroke();}
  function cell(ri,ci){return [node(hs[ri],vs[ci]),node(hs[ri],vs[ci+1]),node(hs[ri+1],vs[ci+1]),node(hs[ri+1],vs[ci])];}
  function label(g,text,x,y,color='#5ce1ff'){g.font=`600 ${Math.max(9,w/100)}px system-ui`;const tw=g.measureText(text).width;g.fillStyle='#101a24e8';g.fillRect(x,y-w/85,tw+8,w/65);g.fillStyle=color;g.fillText(text,x+4,y);}
  function lines(g,nodes=false){g.lineWidth=Math.max(1,w/950);for(const l of hs){g.strokeStyle='#36ceff';path(g,[[l.l,l.b+hh.slope*(l.l-w/2)],[l.r,l.b+hh.slope*(l.r-w/2)]]);g.stroke();}for(const l of vs){g.strokeStyle='#ffd061';path(g,[[l.b+vv.slope*(l.l-h/2),l.l],[l.b+vv.slope*(l.r-h/2),l.r]]);g.stroke();}if(nodes&&good){g.fillStyle='#f0fdff';for(const h0 of hs)for(const v0 of vs){const [x,y]=node(h0,v0);if(x>=0&&x<w&&y>=0&&y<h){g.beginPath();g.arc(x,y,Math.max(1.8,w/550),0,TAU);g.fill();}}}}
  function roi(){return good?[node(hs[0],vs[0]),node(hs[0],vs.at(-1)),node(hs.at(-1),vs.at(-1)),node(hs.at(-1),vs[0])]:[];}
  function renderExtra(i,base,fn){const c=document.createElement('canvas');c.width=w;c.height=h;const g=c.getContext('2d');g.drawImage(base,0,0);g.lineWidth=Math.max(1,w/750);fn(g);if(frameIndices.has(i)){const bitmap=c.transferToImageBitmap();self.postMessage({type:'frame',index:frameIndices.get(i),bitmap},[bitmap]);}}
  renderExtra(10,processedVariants[2].base,g=>{if(good){const pts=roi();g.fillStyle='#07111a99';g.beginPath();g.rect(0,0,w,h);pts.slice().reverse().forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));g.closePath();g.fill('evenodd');poly(g,pts,'#5ce1ff');const [x,y]=pts[0];label(g,'ROI candidate',x,Math.max(w/65,y-5));}lines(g,true);if(!good)label(g,'ROI: insufficient lines',w*.08,h*.10);});
  renderExtra(14,processedVariants[2].base,g=>{lines(g,true);if(good){poly(g,roi(),'#5ce1ff');for(let r=0;r<Math.min(hs.length-1,24);r++)for(let c=0;c<Math.min(vs.length-1,12);c++){const p=cell(r,c);poly(g,p,'#4fe2ff66',c%2===0?'#43beef0e':null);}const p=roi();label(g,`GRID CANDIDATE · ${hs.length}H / ${vs.length}V`,p[0][0],Math.max(w/65,p[0][1]-5));}else label(g,'Grid candidate unavailable',w*.08,h*.10);});
  // Additional browser-side mathematical visualizations. No claims of OCR recognition.
  const blue='#39d9ff', amber='#ffd269', pink='#ff7394', green='#9af8c1';
  const fontSize=Math.max(10,w/85);
  function tag(g,s,x=w*.025,y=h*.055,color=blue){g.font=`600 ${fontSize}px system-ui`;const tw=g.measureText(s).width;g.fillStyle='#08121ee8';g.fillRect(x-3,y-fontSize,tw+10,fontSize*1.5);g.fillStyle=color;g.fillText(s,x+2,y);}
  function stroke(g,pts,color=blue,width=Math.max(1,w/760),dash=[]){if(!pts.length)return;g.strokeStyle=color;g.lineWidth=width;g.setLineDash(dash);path(g,pts);g.stroke();g.setLineDash([]);}
  function dot(g,x,y,color=blue,r=Math.max(2,w/470)){g.fillStyle=color;g.beginPath();g.arc(x,y,r,0,TAU);g.fill();}
  function darkBase(){const c=document.createElement('canvas');c.width=w;c.height=h;const g=c.getContext('2d');g.drawImage(sourceCanvas,0,0);g.fillStyle='#07121ed5';g.fillRect(0,0,w,h);return c;}
  const dark=darkBase(), grayBase=processedVariants[0].base;
  const makePixel=(fn)=>canvasFromPixels(w,h,d=>{for(let j=0;j<w*h;j++){const a=fn(j),i=4*j;if(typeof a==='number'){d[i]=d[i+1]=d[i+2]=a;}else{d[i]=a[0];d[i+1]=a[1];d[i+2]=a[2];}d[i+3]=255;}});
  // Separable Gaussian, 31 taps (sigma=5.0), then local threshold.
  const rad=15, weights=Array.from({length:2*rad+1},(_,k)=>Math.exp(-((k-rad)**2)/50));const sumW=weights.reduce((a,b)=>a+b,0);weights.forEach((v,i)=>weights[i]=v/sumW);
  const blurX=new Float32Array(w*h), localMean=new Float32Array(w*h), ink=new Uint8Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-rad;k<=rad;k++)v+=gray[y*w+clamp(x+k,0,w-1)]*weights[k+rad];blurX[y*w+x]=v;}
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-rad;k<=rad;k++)v+=blurX[clamp(y+k,0,h-1)*w+x]*weights[k+rad];const j=y*w+x;localMean[j]=v;ink[j]=gray[j]<v-9?1:0;}
  const adaptive=makePixel(j=>ink[j]?14:249);
  // True binary opening with axis-aligned line structuring elements: retained runs.
  const hm=new Uint8Array(w*h),vm=new Uint8Array(w*h);
  for(let y=0;y<h;y++){let s=0;for(let x=0;x<=w;x++){if(x<w&&ink[y*w+x])continue;if(x-s>=Math.max(9,Math.round(w*.025)))hm.fill(1,y*w+s,y*w+x);s=x+1;}}
  for(let x=0;x<w;x++){let s=0;for(let y=0;y<=h;y++){if(y<h&&ink[y*w+x])continue;if(y-s>=Math.max(9,Math.round(h*.025)))for(let z=s;z<y;z++)vm[z*w+x]=1;s=y+1;}}
  const morph=makePixel(j=>hm[j]&&vm[j]?[240,255,235]:hm[j]?[46,211,255]:vm[j]?[255,188,66]:[9,19,31]);
  // Bright paper component, downsampled connected component search; ROI is a candidate.
  const step=Math.max(2,Math.ceil(Math.max(w,h)/350)),sw=Math.ceil(w/step),sh=Math.ceil(h/step),seen=new Uint8Array(sw*sh),queue=new Int32Array(sw*sh);let bestPaper=null;
  const pthr=Math.max(175,Math.min(235,mean+12));
  for(let j=0;j<seen.length;j++){if(seen[j])continue;seen[j]=1;const xx=j%sw,yy=(j/sw)|0;if(gray[Math.min(h-1,yy*step)*w+Math.min(w-1,xx*step)]<pthr)continue;let head=0,tail=1;queue[0]=j;let l=xx,r=xx,t=yy,b=yy;while(head<tail){const k=queue[head++],x=k%sw,y=(k/sw)|0;l=Math.min(l,x);r=Math.max(r,x);t=Math.min(t,y);b=Math.max(b,y);for(const n of [x>0?k-1:-1,x<sw-1?k+1:-1,y>0?k-sw:-1,y<sh-1?k+sw:-1]){if(n<0||seen[n])continue;seen[n]=1;const nx=n%sw,ny=(n/sw)|0;if(gray[Math.min(h-1,ny*step)*w+Math.min(w-1,nx*step)]>=pthr)queue[tail++]=n;}}if(!bestPaper||tail>bestPaper.n)bestPaper={n:tail,l:l*step,r:Math.min(w,(r+1)*step),t:t*step,b:Math.min(h,(b+1)*step)};}
  renderExtra(15,sourceCanvas,g=>{if(bestPaper&&bestPaper.n>sw*sh*.025){const {l,r,t,b}=bestPaper;g.fillStyle='#08121ab5';g.beginPath();g.rect(0,0,w,h);g.rect(l,t,r-l,b-t);g.fill('evenodd');g.strokeStyle=blue;g.lineWidth=Math.max(2,w/500);g.strokeRect(l,t,r-l,b-t);for(const [x,y] of [[l,t],[r,t],[r,b],[l,b]])dot(g,x,y,amber);tag(g,'PAPER ROI · bright component',l+5,Math.min(h-8,Math.max(fontSize+4,t+fontSize*1.7)));}else tag(g,'PAPER ROI · insufficient evidence',w*.025,h*.055,amber);});
  renderExtra(16,adaptive,g=>tag(g,'GAUSSIAN LOCAL THRESHOLD · 31 px'));
  renderExtra(17,morph,g=>tag(g,'MORPH OPEN · H / V ink'));
  // Ridge response: positive dark center against local surrounding background.
  const ridge=makePixel(j=>{const x=j%w,y=(j/w)|0,rx=Math.max(0,(pix(x-3,y)+pix(x+3,y))*.5-gray[j]),ry=Math.max(0,(pix(x,y-3)+pix(x,y+3))*.5-gray[j]),v=clamp(Math.max(rx,ry)/95,0,1);return [12+243*v*v,23+190*v,39+180*v*(1-v)];});
  renderExtra(18,ridge,g=>tag(g,'RIDGE RESPONSE · local dark strokes'));
  renderExtra(19,dark,g=>{lines(g);for(const l of hs){const y=l.b;tag(g,`${(Math.atan(hh.slope)*180/Math.PI).toFixed(1)}°`,clamp(l.l+4,0,w*.8),clamp(y,15,h-10),blue);}tag(g,`HOUGH-LIKE VOTES · ${hs.length}H / ${vs.length}V`);});
  // Observed graph: intersections only if near observed segment ranges.
  const graph=[];
  if(good)for(let r=0;r<hs.length;r++)for(let c=0;c<vs.length;c++){const [x,y]=node(hs[r],vs[c]);if(x>=0&&x<w&&y>=0&&y<h&&x>=hs[r].l-w*.025&&x<=hs[r].r+w*.025&&y>=vs[c].l-h*.025&&y<=vs[c].r+h*.025)graph.push({r,c,x,y});}
  const graphMap=new Map(graph.map(p=>[`${p.r},${p.c}`,p]));
  renderExtra(20,dark,g=>{for(const p of graph){for(const [dr,dc] of [[1,0],[0,1]]){const q=graphMap.get(`${p.r+dr},${p.c+dc}`);if(q)stroke(g,[[p.x,p.y],[q.x,q.y]],dr?amber:blue);}}for(const p of graph)dot(g,p.x,p.y,'#edffff');tag(g,`PHYSICAL GRAPH · ${graph.length} candidate nodes`);});
  // Sample local dark-ridge points around observed lines and fit a quadratic in normalized coords.
  function fitRule(rule,horizontal){const major=horizontal?w:h,minor=horizontal?h:w,slope=horizontal?hh.slope:vv.slope;let points=[];const margin=Math.max(4,Math.round(minor*.009));for(let u=Math.ceil(rule.l);u<=rule.r;u+=Math.max(4,Math.round(major/75))){const pred=rule.b+slope*(u-major/2);let best=-1,bestV=0;for(let v=Math.max(2,Math.round(pred)-margin);v<=Math.min(minor-3,Math.round(pred)+margin);v++){const x=horizontal?u:v,y=horizontal?v:u;const val=(horizontal?(pix(x,y-2)+pix(x,y+2)):(pix(x-2,y)+pix(x+2,y)))*.5-gray[y*w+x]-.9*Math.abs(v-pred);if(val>bestV){bestV=val;best=v;}}if(best>=0&&bestV>12)points.push({u,v:best});}
    if(points.length<5)return {rule,horizontal,points,eval:u=>rule.b+slope*(u-major/2),fit:false};
    const a=Array.from({length:3},()=>[0,0,0,0]);for(const p of points){const t=(p.u-major/2)/major,v=[1,t,t*t];for(let i=0;i<3;i++){for(let j=0;j<3;j++)a[i][j]+=v[i]*v[j];a[i][3]+=v[i]*p.v;}}
    for(let i=0;i<3;i++){let k=i;for(let j=i+1;j<3;j++)if(Math.abs(a[j][i])>Math.abs(a[k][i]))k=j;[a[i],a[k]]=[a[k],a[i]];const div=a[i][i];if(Math.abs(div)<1e-8)return {rule,horizontal,points,eval:u=>rule.b+slope*(u-major/2),fit:false};for(let j=i;j<4;j++)a[i][j]/=div;for(let k=0;k<3;k++)if(k!==i){const f=a[k][i];for(let j=i;j<4;j++)a[k][j]-=f*a[i][j];}}
    const cf=a.map(r=>r[3]);return {rule,horizontal,points,fit:true,eval:u=>{const t=(u-major/2)/major;return cf[0]+cf[1]*t+cf[2]*t*t;}};}
  const fittedH=hs.map(r=>fitRule(r,true)), fittedV=vs.map(r=>fitRule(r,false)), fitted=[...fittedH,...fittedV];
  function fitPath(f){return Array.from({length:41},(_,i)=>{const u=lerp(f.rule.l,f.rule.r,i/40),v=f.eval(u);return f.horizontal?[u,v]:[v,u];});}
  renderExtra(21,dark,g=>{for(const f of fitted){for(const p of f.points)dot(g,f.horizontal?p.u:p.v,f.horizontal?p.v:p.u,amber,Math.max(1,w/900));if(f.fit)stroke(g,fitPath(f),blue,Math.max(2,w/650));}tag(g,'POLYNOMIAL FIT · measured ridge samples');});
  renderExtra(22,sourceCanvas,g=>{g.fillStyle='#0b172660';g.fillRect(0,0,w,h);for(const f of fitted){const s=f.horizontal?hh.slope:vv.slope,major=f.horizontal?w:h,old=u=>f.rule.b+s*(u-major/2);stroke(g,[f.horizontal?[f.rule.l,old(f.rule.l)]:[old(f.rule.l),f.rule.l],f.horizontal?[f.rule.r,old(f.rule.r)]:[old(f.rule.r),f.rule.r]],amber,1.5,[6,5]);if(f.fit)stroke(g,fitPath(f),blue,Math.max(2,w/650));}tag(g,'GRID REFINE · initial / fitted');});
  // Missing ink segments / edge recovery candidates, not claimed recovered.
  renderExtra(23,dark,g=>{for(const f of fitted){stroke(g,fitPath(f),'#6b9bbb',1,[5,5]);const major=f.horizontal?w:h;const pts=f.points;for(let i=1;i<pts.length;i++){const a=pts[i-1],b=pts[i],gap=b.u-a.u;if(gap>major*.045){const p=f.horizontal?[a.u,a.v]:[a.v,a.u],q=f.horizontal?[b.u,b.v]:[b.v,b.u];stroke(g,[p,q],pink,3,[7,4]);}else stroke(g,[f.horizontal?[a.u,a.v]:[a.v,a.u],f.horizontal?[b.u,b.v]:[b.v,b.u]],green,2);}}tag(g,'EDGE SUPPORT · gaps = review',w*.025,h*.055,pink);});
  // Cell safety proxy: spacing/area, no fabricated labels.
  renderExtra(24,dark,g=>{if(good){const dif=(a)=>a.slice(1).map((r,i)=>r.b-a[i].b),dx=dif(vs),dy=dif(hs),med=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];const mx=med(dx),my=med(dy);for(let r=0;r<dy.length;r++)for(let c=0;c<dx.length;c++){const bad=dx[c]<mx*.35||dy[r]<my*.35||dy[r]>my*3.5;poly(g,cell(r,c),bad?pink:blue,bad?'#ff73944b':'#39d9ff12');}}tag(g,'GRID SAFETY · spacing / shape candidate',w*.025,h*.055,amber);});
  // Multiscale contrast response calculated at 1, 4 and 12 pixels.
  // Mesh / rectification within the detected grid only. Same canvas dimensions for overlay alignment.
  const meshNodes=good?hs.map(H=>vs.map(V=>node(H,V))):[];
  renderExtra(26,dark,g=>{if(good){for(let r=0;r<hs.length-1;r++)for(let c=0;c<vs.length-1;c++){const p=cell(r,c);poly(g,p,blue,((r+c)%2)?'#298bba24':'#39d9ff10');for(const q of p)dot(g,q[0],q[1],amber,1.8);}}tag(g,'GRID MESH · coordinate cell candidates');});
  let bicubic=null;
  if(good){const src=sourceCanvas.getContext('2d',{willReadFrequently:true}).getImageData(0,0,w,h),sd=src.data;const left=clamp(Math.round(vs[0].b),0,w-1),right=clamp(Math.round(vs.at(-1).b),left+1,w),top=clamp(Math.round(hs[0].b),0,h-1),bottom=clamp(Math.round(hs.at(-1).b),top+1,h);
    const xs=vs.map(v=>v.b),ys=hs.map(v=>v.b);function interval(a,x){let i=0;while(i<a.length-2&&x>a[i+1])i++;return i;}function uv(x,y){const c=interval(xs,x),r=interval(ys,y),u=clamp((x-xs[c])/(xs[c+1]-xs[c]),0,1),v=clamp((y-ys[r])/(ys[r+1]-ys[r]),0,1),p=meshNodes[r][c],q=meshNodes[r][c+1],s=meshNodes[r+1][c],t=meshNodes[r+1][c+1];return [lerp(lerp(p[0],q[0],u),lerp(s[0],t[0],u),v),lerp(lerp(p[1],q[1],u),lerp(s[1],t[1],u),v)];}
    function cubic(x){x=Math.abs(x);return x<=1?1.5*x*x*x-2.5*x*x+1:x<2?-.5*x*x*x+2.5*x*x-4*x+2:0;}
    function remap(useCubic){const c=document.createElement('canvas');c.width=w;c.height=h;const g=c.getContext('2d');const dst=g.createImageData(w,h);dst.data.set(sd);const d=dst.data;for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){const [sx0,sy0]=uv(x,y),sx=clamp(sx0,0,w-1),sy=clamp(sy0,0,h-1),ix=Math.floor(sx),iy=Math.floor(sy),j=4*(y*w+x);if(!useCubic){const k=4*(Math.round(sy)*w+Math.round(sx));for(let z=0;z<3;z++)d[j+z]=sd[k+z];}else{let rr=0,gg=0,bb=0;for(let yy=-1;yy<=2;yy++)for(let xx=-1;xx<=2;xx++){const weight=cubic(sx-ix-xx)*cubic(sy-iy-yy),k=4*(clamp(iy+yy,0,h-1)*w+clamp(ix+xx,0,w-1));rr+=weight*sd[k];gg+=weight*sd[k+1];bb+=weight*sd[k+2];}d[j]=rr;d[j+1]=gg;d[j+2]=bb;}}g.putImageData(dst,0,0);return c;}
    bicubic=remap(true);
  }
  renderExtra(28,bicubic||sourceCanvas,g=>{if(good){g.strokeStyle=green;g.strokeRect(vs[0].b,hs[0].b,vs.at(-1).b-vs[0].b,hs.at(-1).b-hs[0].b);}tag(g,good?'BICUBIC REMAP · 4×4 samples':'BICUBIC · no reliable grid');});
  // Numeric ink bounds in candidate cells: last three columns are hypotheses.
  const inkBoxes=[];if(good)for(let r=0;r<Math.min(hs.length-1,30);r++)for(let c=Math.max(0,vs.length-4);c<vs.length-1;c++){const p=cell(r,c),l=clamp(Math.ceil(Math.min(...p.map(q=>q[0])))+3,0,w),rt=clamp(Math.floor(Math.max(...p.map(q=>q[0])))-3,0,w),t=clamp(Math.ceil(Math.min(...p.map(q=>q[1])))+3,0,h),b=clamp(Math.floor(Math.max(...p.map(q=>q[1])))-3,0,h);let x0=rt,x1=l,y0=b,y1=t,n=0;for(let y=t;y<b;y++)for(let x=l;x<rt;x++)if(ink[y*w+x]&&!hm[y*w+x]&&!vm[y*w+x]){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);n++;}if(n>4&&x1>x0&&y1>y0)inkBoxes.push({p,x:x0,y:y0,w:x1-x0+1,h:y1-y0+1,n,r,c});}
  renderExtra(29,grayBase,g=>{for(const z of inkBoxes){poly(g,z.p,'#4b94bc');g.strokeStyle=amber;g.strokeRect(z.x,z.y,z.w,z.h);}tag(g,`NUMERIC ROI · ${inkBoxes.length} candidate ink boxes`);});
  renderExtra(30,sourceCanvas,g=>{g.fillStyle='#06111bb5';g.fillRect(0,0,w,h);if(good){lines(g);const p=roi(),y=Math.max(...p.slice(2).map(z=>z[1])),l=clamp(Math.min(p[2][0],p[3][0]),0,w),r=clamp(Math.max(p[2][0],p[3][0]),0,w),end=clamp(y+(h-y)*.75,0,h);if(end>y){g.drawImage(sourceCanvas,l,y,r-l,end-y,l,y,r-l,end-y);g.strokeStyle=amber;g.strokeRect(l,y,r-l,end-y);tag(g,'FOOTER SEARCH ROI · not recognized',l+3,Math.min(h-8,y+fontSize*1.6),amber);}}else tag(g,'FOOTER · no reliable grid');});

  // ===== Additional source-inspired visualizations =====
  // Histogram equalization (used as CLAHE-style contrast enhancement demo).
  const hist=new Uint32Array(256); for(let j=0;j<gray.length;j++) hist[Math.max(0,Math.min(255,Math.round(gray[j])))]++;
  const cdf=new Uint32Array(256); let accum=0, cdfMin=0; for(let i=0;i<256;i++){ accum+=hist[i]; cdf[i]=accum; if(!cdfMin && accum) cdfMin=accum; }
  const grayCount=gray.length;
  const eqMap=new Uint8Array(256); for(let i=0;i<256;i++) eqMap[i]=Math.round((cdf[i]-cdfMin)/Math.max(1,grayCount-cdfMin)*255);
  const clahe=makePixel(j=>{const v=eqMap[Math.max(0,Math.min(255,Math.round(gray[j])))]; return [v,v,v];});
  renderExtra(31,clahe,g=>tag(g,'CLAHE PREVIEW · global histogram approximation'));
  const gauss=makePixel(j=>{const v=Math.round(localMean[j]); return [v,v,v];});
  renderExtra(32,gauss,g=>tag(g,'GAUSSIAN BLUR · sigma≈5.0'));
  // Otsu threshold
  let total=gray.length,sum=0; for(let i=0;i<256;i++) sum += i*hist[i]; let sumB=0,wB=0,maxVar=-1,otsu=127;
  for(let t=0;t<256;t++){ wB += hist[t]; if(!wB) continue; const wF=total-wB; if(!wF) break; sumB += t*hist[t]; const mB=sumB/wB,mF=(sum-sumB)/wF; const between=wB*wF*(mB-mF)*(mB-mF); if(between>maxVar){ maxVar=between; otsu=t; } }
  const otsuBin=makePixel(j=>gray[j]<otsu?24:248);
  renderExtra(33,otsuBin,g=>tag(g,`OTSU THRESHOLD · T=${otsu}`));
  // Simple 3x3 morphology on ink mask
  function morphOp(src, mode){ const out=new Uint8Array(w*h); for(let y=0;y<h;y++) for(let x=0;x<w;x++){ let on = mode==='erode'?1:0; for(let yy=-1;yy<=1;yy++) for(let xx=-1;xx<=1;xx++){ const v=src[clamp(y+yy,0,h-1)*w+clamp(x+xx,0,w-1)]; if(mode==='dilate') on = on||v; else on = on&&v; } out[y*w+x]=on?1:0; } return out; }
  const dil = morphOp(ink,'dilate'), ero = morphOp(ink,'erode'), close1 = morphOp(dil,'erode');
  const dilCanvas=makePixel(j=>dil[j]?18:250), eroCanvas=makePixel(j=>ero[j]?18:250), closeCanvas=makePixel(j=>close1[j]?18:250);
  renderExtra(34,closeCanvas,g=>tag(g,'MORPH CLOSE · bridge gaps'));
  renderExtra(35,dilCanvas,g=>tag(g,'DILATION · expand strokes'));
  renderExtra(36,eroCanvas,g=>tag(g,'EROSION · shrink strokes'));
  const blackHat=makePixel(j=>{const v=clamp(localMean[j]-gray[j]+128,0,255); return [v,v*.92,v*.78];});
  renderExtra(37,blackHat,g=>tag(g,'BLACK HAT PREVIEW · local contrast approximation'));
  renderExtra(38,dark,g=>{lines(g); tag(g,`LSD-LIKE SEGMENTS · ${hs.length+vs.length}`);});
  // Connected components on ink
  const compColor=document.createElement('canvas'); compColor.width=w; compColor.height=h; {
    const g=compColor.getContext('2d',{willReadFrequently:true}); const img=g.createImageData(w,h), d=img.data; const lab=new Int32Array(w*h); let cid=0; const qx=new Int32Array(w*h), qy=new Int32Array(w*h);
    for(let y=0;y<h;y++) for(let x=0;x<w;x++){ const s=y*w+x; if(!ink[s]||lab[s]) continue; cid++; let head=0,tail=0; lab[s]=cid; qx[tail]=x; qy[tail++]=y; while(head<tail){ const cx=qx[head], cy=qy[head++]; for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){ const nx=cx+dx, ny=cy+dy; if(nx<0||ny<0||nx>=w||ny>=h) continue; const ni=ny*w+nx; if(!ink[ni]||lab[ni]) continue; lab[ni]=cid; qx[tail]=nx; qy[tail++]=ny; } }
    }
    for(let i=0;i<w*h;i++){ const j=4*i; if(!lab[i]){ d[j]=246; d[j+1]=246; d[j+2]=246; d[j+3]=255; continue; } const hue=(lab[i]*47)%360; const a=0.82; const c=0.62; const x=c*(1-Math.abs((hue/60)%2-1)); let r1=0,g1=0,b1=0; if(hue<60){r1=c;g1=x;} else if(hue<120){r1=x;g1=c;} else if(hue<180){g1=c;b1=x;} else if(hue<240){g1=x;b1=c;} else if(hue<300){r1=x;b1=c;} else {r1=c;b1=x;} const m=0.15; d[j]=(r1+m)*255; d[j+1]=(g1+m)*255; d[j+2]=(b1+m)*255; d[j+3]=255; }
    g.putImageData(img,0,0);
  }
  renderExtra(39,compColor,g=>tag(g,'CONNECTED COMPONENTS · color labels'));
  renderExtra(40,dark,g=>{ if(good){ for(let k=0;k<3;k++){ g.globalAlpha=.85-k*.2; g.lineWidth=Math.max(1,w/950)+k; lines(g); } g.globalAlpha=1; } tag(g,'MULTISCALE PREVIEW · same detected segments'); });
  renderExtra(41,dark,g=>{ for(const f of fitted){ if(f.fit) stroke(g,fitPath(f),green,Math.max(2,w/650)); for(const p of f.points) dot(g,f.horizontal?p.u:p.v,f.horizontal?p.v:p.u,amber,1.6);} tag(g,'RIDGE REFINEMENT · sampled support',w*.025,h*.055,green);});
  renderExtra(42,grayBase,g=>{ const margin=Math.round(Math.min(w,h)*.08); g.fillStyle='#0a1522cc'; g.fillRect(0,0,w,margin); g.fillRect(0,0,margin,h); let rowMax=1,colMax=1; const rowS=new Float32Array(h), colS=new Float32Array(w); for(let y=0;y<h;y++){ let s=0; for(let x=0;x<w;x++) s+=ink[y*w+x]; rowS[y]=s; rowMax=Math.max(rowMax,s);} for(let x=0;x<w;x++){ let s=0; for(let y=0;y<h;y++) s+=ink[y*w+x]; colS[x]=s; colMax=Math.max(colMax,s);} g.strokeStyle=blue; g.beginPath(); for(let x=0;x<w;x++){ const yy=margin-(colS[x]/colMax)*(margin-6); x?g.lineTo(x,yy):g.moveTo(x,yy);} g.stroke(); g.strokeStyle=amber; g.beginPath(); for(let y=0;y<h;y++){ const xx=margin-(rowS[y]/rowMax)*(margin-6); y?g.lineTo(xx,y):g.moveTo(xx,y);} g.stroke(); tag(g,'PROJECTION PROFILES · H/V histograms', margin+6, Math.max(fontSize+4, margin-4)); });
  renderExtra(43,dark,g=>{ if(good){ for(let i=0;i<hs.length;i++){ const f=fittedH[i]; if(f&&f.fit) stroke(g,fitPath(f),green,Math.max(2,w/700)); } for(let i=0;i<vs.length;i++){ const f=fittedV[i]; if(f&&f.fit) stroke(g,fitPath(f),amber,Math.max(2,w/760)); } } tag(g,'TRACE PRINTED RULES V16'); });
  let linearMap=null; 
  if(good){ const src=sourceCanvas.getContext('2d',{willReadFrequently:true}).getImageData(0,0,w,h), sd=src.data; const left=clamp(Math.round(vs[0].b),0,w-1), right=clamp(Math.round(vs.at(-1).b),left+1,w), top=clamp(Math.round(hs[0].b),0,h-1), bottom=clamp(Math.round(hs.at(-1).b),top+1,h); const xs=vs.map(v=>v.b), ys=hs.map(v=>v.b); function interval2(a,x){let i=0;while(i<a.length-2&&x>a[i+1])i++;return i;} function uv2(x,y){const c=interval2(xs,x),r=interval2(ys,y),u=clamp((x-xs[c])/(xs[c+1]-xs[c]),0,1),v=clamp((y-ys[r])/(ys[r+1]-ys[r]),0,1),p=meshNodes[r][c],q=meshNodes[r][c+1],s=meshNodes[r+1][c],t=meshNodes[r+1][c+1];return [lerp(lerp(p[0],q[0],u),lerp(s[0],t[0],u),v),lerp(lerp(p[1],q[1],u),lerp(s[1],t[1],u),v)];} const c=document.createElement('canvas'); c.width=w; c.height=h; const g=c.getContext('2d'); const dst=g.createImageData(w,h); dst.data.set(sd); const d=dst.data; for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){ const [sx0,sy0]=uv2(x,y), sx=clamp(Math.round(sx0),0,w-1), sy=clamp(Math.round(sy0),0,h-1), j=4*(y*w+x), k=4*(sy*w+sx); d[j]=sd[k]; d[j+1]=sd[k+1]; d[j+2]=sd[k+2]; } g.putImageData(dst,0,0); linearMap=c; }
  renderExtra(44,linearMap||sourceCanvas,g=>tag(g,good?'PERSPECTIVE PREVIEW · mesh remap':'PERSPECTIVE · no reliable grid'));
  // TPS-like smooth dewarp visualization: same image plus smooth control grid.
  renderExtra(45,bicubic||sourceCanvas,g=>{ if(good){ g.strokeStyle='rgba(57,217,255,.8)'; g.lineWidth=Math.max(1,w/950); for(let r=0;r<hs.length;r++){ const pts=[]; for(let c=0;c<vs.length;c++) pts.push(node(hs[r],vs[c])); stroke(g,pts,blue,1.5); } for(let c=0;c<vs.length;c++){ const pts=[]; for(let r=0;r<hs.length;r++) pts.push(node(hs[r],vs[c])); stroke(g,pts,amber,1.5); } } tag(g,'TPS PREVIEW · mesh approximation'); });
  // Fold recovery approximation: strongest central vertical ridge / fold candidate.
  let foldX=Math.round(w/2), foldScore=0; for(let x=Math.round(w*.15);x<Math.round(w*.85);x++){ let s=0; for(let y=0;y<h;y++) s += Math.max(0,(pix(Math.max(0,x-2),y)+pix(Math.min(w-1,x+2),y))*.5 - gray[y*w+x]); if(s>foldScore){ foldScore=s; foldX=x; }}
  renderExtra(46,sourceCanvas,g=>{ g.fillStyle='#0a111880'; g.fillRect(0,0,w,h); g.drawImage(sourceCanvas,0,0); g.strokeStyle=pink; g.lineWidth=Math.max(2,w/550); g.setLineDash([10,6]); g.beginPath(); g.moveTo(foldX,0); g.lineTo(foldX,h); g.stroke(); g.setLineDash([]); g.strokeStyle=green; g.beginPath(); g.moveTo(foldX-12,0); g.lineTo(foldX-12,h); g.stroke(); g.beginPath(); g.moveTo(foldX+12,0); g.lineTo(foldX+12,h); g.stroke(); tag(g,'FOLD PREVIEW · candidate crease', clamp(foldX+8,8,w*.7), Math.max(fontSize+4,h*.08), pink); });
  renderExtra(47,dark,g=>{ lines(g); if(vs.length){ const left=vs[0], right=vs.at(-1); stroke(g,[[left.b+vv.slope*(left.l-h/2),left.l],[left.b+vv.slope*(left.r-h/2),left.r]],green,3); stroke(g,[[right.b+vv.slope*(right.l-h/2),right.l],[right.b+vv.slope*(right.r-h/2),right.r]],green,3); } tag(g,'VERTICAL RAIL RECOVERY · outer rails',w*.025,h*.055,green); });
  renderExtra(48,dark,g=>{ lines(g); if(hs.length){ const topL=hs[0]; stroke(g,[[topL.l,topL.b+hh.slope*(topL.l-w/2)],[topL.r,topL.b+hh.slope*(topL.r-w/2)]],green,3); if(hs.length>1) stroke(g,[[hs[1].l,hs[1].b+hh.slope*(hs[1].l-w/2)],[hs[1].r,hs[1].b+hh.slope*(hs[1].r-w/2)]],amber,2); } tag(g,'HEADER EDGE RECOVERY · top borders',w*.025,h*.055,green); });
  // Chamfer-like distance transform on inverse ink (distance to nearest ink stroke)
  const dist=new Float32Array(w*h); const INF=1e9; for(let i=0;i<w*h;i++) dist[i]=ink[i]?0:INF; for(let y=0;y<h;y++) for(let x=0;x<w;x++){ const i=y*w+x; let v=dist[i]; if(x>0) v=Math.min(v,dist[i-1]+1); if(y>0) v=Math.min(v,dist[i-w]+1); if(x>0&&y>0) v=Math.min(v,dist[i-w-1]+1.414); dist[i]=v; } for(let y=h-1;y>=0;y--) for(let x=w-1;x>=0;x--){ const i=y*w+x; let v=dist[i]; if(x<w-1) v=Math.min(v,dist[i+1]+1); if(y<h-1) v=Math.min(v,dist[i+w]+1); if(x<w-1&&y<h-1) v=Math.min(v,dist[i+w+1]+1.414); dist[i]=v; } let maxDist=1; for(let i=0;i<dist.length;i++) if(dist[i]<INF) maxDist=Math.max(maxDist,dist[i]);
  const dtCanvas=makePixel(j=>{ const t=1-Math.min(1,dist[j]/maxDist); return [30+210*t, 30+150*(1-t), 70+185*(1-t*t)]; });
  renderExtra(49,dtCanvas,g=>tag(g,'DISTANCE TRANSFORM · to nearest stroke'));
  renderExtra(50,grayBase,g=>{ for(const z of inkBoxes){ poly(g,z.p,'#4b94bc'); g.strokeStyle=amber; g.strokeRect(z.x,z.y,z.w,z.h); const cw=Math.max(6,Math.round(z.w*.18)); g.fillStyle='rgba(255,208,77,.28)'; g.fillRect(z.x+z.w-cw,z.y,cw,z.h); g.strokeStyle=pink; g.strokeRect(z.x+z.w-cw,z.y,cw,z.h); } tag(g,'CURRENCY PREVIEW · suffix candidate',w*.025,h*.055,amber); });
  const sobelCanvas=makePixel(j=>{ const ang=Math.atan2((j>=w?sobelY[j]:0),(j>=w?sobelX[j]:0)); const mag=Math.min(1,sobelMag[j]/maxMag*2.4); const hue=(ang+Math.PI)/(2*Math.PI); const h6=hue*6, c=mag, x=c*(1-Math.abs(h6%2-1)); let r=0,g=0,b=0; if(h6<1){r=c;g=x;} else if(h6<2){r=x;g=c;} else if(h6<3){g=c;b=x;} else if(h6<4){g=x;b=c;} else if(h6<5){r=x;b=c;} else {r=c;b=x;} return [255*(r*.9+.1),255*(g*.9+.1),255*(b*.9+.1)]; });
  renderExtra(51,sobelCanvas,g=>tag(g,'SOBEL GRADIENT · orientation + magnitude'));
  renderExtra(52,grayBase,g=>{ const step=Math.max(18,Math.round(Math.min(w,h)/22)); g.strokeStyle=blue; g.lineWidth=1.4; for(let y=Math.floor(step/2);y<h;y+=step) for(let x=Math.floor(step/2);x<w;x+=step){ let gx=0,gy=0; for(let yy=-Math.floor(step/2);yy<Math.floor(step/2);yy++) for(let xx=-Math.floor(step/2);xx<Math.floor(step/2);xx++){ const px=clamp(x+xx,0,w-1), py=clamp(y+yy,0,h-1), i=py*w+px; gx+=sobelX[i]; gy+=sobelY[i]; } const ang=Math.atan2(gy,gx), len=Math.min(step*.38, Math.hypot(gx,gy)/(step*step*8)); g.beginPath(); g.moveTo(x-len*Math.cos(ang), y-len*Math.sin(ang)); g.lineTo(x+len*Math.cos(ang), y+len*Math.sin(ang)); g.stroke(); } tag(g,'HOG PREVIEW · cell orientation field'); });
  // LBP map
  const lbpCanvas=makePixel(j=>{ const x=j%w,y=(j/w)|0; if(!x||!y||x===w-1||y===h-1) return [128,128,128]; const c=gray[j]; let code=0,bit=0; for(const [dx,dy] of [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]){ if(gray[(y+dy)*w+(x+dx)]>=c) code |= (1<<bit); bit++; } return [code,255-code,(code^85)]; });
  renderExtra(53,lbpCanvas,g=>tag(g,'LOCAL BINARY PATTERNS'));
  // Flood topology: holes in white background not touching border
  const free=new Uint8Array(w*h); for(let i=0;i<w*h;i++) free[i]=ink[i]?0:1; const seen2=new Uint8Array(w*h); const topo=document.createElement('canvas'); topo.width=w; topo.height=h; { const g=topo.getContext('2d',{willReadFrequently:true}); const img=g.createImageData(w,h), d=img.data; for(let i=0;i<w*h;i++){ d[4*i]=240; d[4*i+1]=240; d[4*i+2]=240; d[4*i+3]=255; if(ink[i]) d[4*i]=d[4*i+1]=d[4*i+2]=25; }
    const q=new Int32Array(w*h); for(let s=0;s<w*h;s++){ if(!free[s]||seen2[s]) continue; let head=0,tail=0,border=false; seen2[s]=1; q[tail++]=s; const cells=[]; while(head<tail){ const i=q[head++], x=i%w,y=(i/w)|0; cells.push(i); if(x===0||y===0||x===w-1||y===h-1) border=true; for(const n of [i-1,i+1,i-w,i+w]){ const nx=n%w, ny=(n/w)|0; if(n<0||nx<0||ny<0||nx>=w||ny>=h||!free[n]||seen2[n]) continue; seen2[n]=1; q[tail++]=n; } }
      if(!border){ const hue=(cells.length*13)%360; for(const i of cells){ const j=4*i; d[j]=255; d[j+1]=155+(hue%100); d[j+2]=120; } }
    }
    g.putImageData(img,0,0); }
  renderExtra(54,topo,g=>tag(g,'FLOOD TOPOLOGY · enclosed white regions'));
  renderExtra(55,grayBase,g=>{ // sample a few contour anchor points and radial bins
    const pts=[]; for(let y=8;y<h-8;y+=Math.max(12,Math.round(h/18))) for(let x=8;x<w-8;x+=Math.max(12,Math.round(w/18))) if(sobelMag[y*w+x]>maxMag*.35) pts.push([x,y]); const sample=pts.slice(0,Math.min(6,pts.length)); for(const [x,y] of sample){ dot(g,x,y,pink,2.2); for(const r of [18,36,54]){ g.strokeStyle='rgba(255,115,148,.45)'; g.beginPath(); g.arc(x,y,r,0,TAU); g.stroke(); } for(let a=0;a<8;a++){ const ang=a*TAU/8; g.strokeStyle='rgba(57,217,255,.35)'; g.beginPath(); g.moveTo(x,y); g.lineTo(x+60*Math.cos(ang),y+60*Math.sin(ang)); g.stroke(); } } tag(g,'SHAPE CONTEXT · radial bins around edge samples',w*.025,h*.055,pink); });
  renderExtra(56,grayBase,g=>{ const zonesX=4,zonesY=4,xs=Math.floor(w/zonesX),ys=Math.floor(h/zonesY); for(let yy=0;yy<zonesY;yy++) for(let xx=0;xx<zonesX;xx++){ let n=0,area=0; for(let y=yy*ys;y<Math.min(h,(yy+1)*ys);y++) for(let x=xx*xs;x<Math.min(w,(xx+1)*xs);x++){ area++; n+=ink[y*w+x]; } const occ=n/Math.max(1,area); g.fillStyle=`rgba(57,217,255,${0.08+occ*0.45})`; g.fillRect(xx*xs,yy*ys,Math.min(xs,w-xx*xs),Math.min(ys,h-yy*ys)); g.strokeStyle='rgba(255,255,255,.38)'; g.strokeRect(xx*xs,yy*ys,Math.min(xs,w-xx*xs),Math.min(ys,h-yy*ys)); g.fillStyle='#eaf7ff'; g.font=`600 ${Math.max(10,w/120)}px system-ui`; g.fillText((occ*100).toFixed(0)+'%',xx*xs+6,yy*ys+18); } tag(g,'ZONING · occupancy by zone'); });


}

const frameIndices=new Map(variants.map((v,i)=>[v.sourceIndex,i]));
self.onmessage=async e=>{try{
 const im=await createImageBitmap(e.data.file);const k=Math.min(1,850/Math.max(im.width,im.height));
 sourceCanvas=new OffscreenCanvas(Math.max(1,Math.round(im.width*k)),Math.max(1,Math.round(im.height*k)));
 sourceCanvas.getContext('2d').drawImage(im,0,0,sourceCanvas.width,sourceCanvas.height);im.close();
 self.postMessage({type:'meta',names:variants.map(v=>v.name)});
 buildProcessedVariants();self.postMessage({type:'done'});
}catch(e){self.postMessage({type:'error',message:String(e?.message||e)});}};
