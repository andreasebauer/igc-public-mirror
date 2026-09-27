(()=>{
'use strict';
const $=id=>document.getElementById(id), api=globalThis.RMTest;
if(!api?.pack())return;
const checked=new Set();let lastLevel='',lastGraph=null,sharpness=100,frame=0;
const questions=[
 {id:'zero',label:'Label L equals zero',test:o=>o.row[0]===0},
 {id:'score',label:'Score s equals two',test:o=>o.row[2]===2},
 {id:'stay',label:'Stay equals one',test:o=>o.row[4]===1}
];
const left=document.createElement('div');left.id='rm-left';left.innerHTML='<div><span class="rm-kicker">Reverse microscope</span><h1>Infinity Grid</h1><div class="rm-level" id="rm-level">L0</div></div><section class="rm-group"><h2>Zoom · scientific level</h2><div class="rm-controls"><button id="rm-in" title="One recorded level inward">In</button><button id="rm-out" title="One recorded level outward">Out</button></div><p class="rm-small">One step changes the data level.</p></section><section class="rm-group"><h2>Focus · visibility</h2><input class="rm-focus" id="rm-focus" type="range" min="0" max="100" value="100" aria-label="Focus: hide records that do not match checked questions"><p class="rm-small">Fade to hide records outside the checked answers.</p><div class="rm-controls"><button id="rm-camera-out" title="Magnify picture out">−</button><button id="rm-fit" title="Fit picture">Fit</button><button id="rm-camera-in" title="Magnify picture in">+</button></div></section>';
const right=document.createElement('aside');right.id='rm-observers';right.innerHTML='<span class="rm-kicker">Recorded answers</span><h2>Observers</h2><div id="rm-status" class="rm-status"></div><div id="rm-questions"></div><div><button id="rm-all" class="rm-action">Ask bundle</button><button id="rm-clear" class="rm-action">Clear answers</button><button id="rm-collection" class="rm-action" hidden>Open collection</button></div><p id="rm-provenance" class="rm-meta"></p>';
const screen=document.createElement('div');screen.className='rm-screen';screen.onclick=closeInspector;
document.body.append(screen);$('catalog').prepend(left);document.querySelector('.workspace').append(right);
function closeInspector(){$('inspector').classList.remove('open');screen.classList.remove('open')}
function openInspector(){$('inspector').classList.add('open');screen.classList.add('open')}
$('close-inspector').addEventListener('click',closeInspector);
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeInspector()});
// Native tap handlers fill the inspector first; the bubble listener opens the sheet.
$('graph-area').addEventListener('click',e=>{if(e.target.closest?.('.upper-node'))openInspector()});
$('rm-in').onclick=()=>$('level-in').click();$('rm-out').onclick=()=>$('level-out').click();
$('rm-camera-out').onclick=()=>$('camera-out').click();$('rm-camera-in').onclick=()=>$('camera-in').click();$('rm-fit').onclick=()=>$('fit-button').click();
$('rm-focus').oninput=e=>{sharpness=Number(e.target.value);apply()};
$('rm-all').onclick=()=>{questions.forEach(q=>checked.add(q.id));renderQuestions();apply()};
$('rm-clear').onclick=()=>{checked.clear();renderQuestions();apply()};
$('rm-collection').onclick=()=>{const level=$('current-level').textContent;api.level(level);closeInspector();schedule()};
function level(){return $('current-level')?.textContent||api.state().level}
function foundation(){return /^L[012]$/.test(level())}
function objects(){return api.pack().foundation.objects.filter(o=>o.level===Number(level()[1]))}
function accepted(o){return questions.every(q=>!checked.has(q.id)||q.test(o))}
function renderQuestions(){
 const box=$('rm-questions');box.replaceChildren();
 if(!foundation()){
  $('rm-focus').disabled=true;
  box.textContent='No per-object observer answers are bundled at this level.';
  $('rm-status').textContent='Questions unavailable';$('rm-provenance').textContent='The displayed evidence stays inspectable. An answer table is needed before checkboxes can select its objects.';
  $('rm-all').hidden=$('rm-clear').hidden=true;$('rm-collection').hidden=true;return;
 }
 $('rm-focus').disabled=false;
 const all=objects();$('rm-all').hidden=$('rm-clear').hidden=false;$('rm-collection').hidden=!api.state().object;
 for(const q of questions){const label=document.createElement('label');label.className='rm-question';const input=document.createElement('input');input.type='checkbox';input.checked=checked.has(q.id);input.onchange=()=>{input.checked?checked.add(q.id):checked.delete(q.id);apply()};const span=document.createElement('span');span.textContent=q.label+' · '+all.filter(q.test).length+'/'+all.length;label.append(input,span);box.append(label)}
 $('rm-provenance').textContent='Exact saved row [L, ports, score, destinations, stay]. Checked answers combine with AND. Unchecked questions impose no constraint. This filters the view; it does not select which possibility is actual.';
 apply();
}
function apply(){
 if(!foundation())return;
 const all=objects(),matches=all.filter(accepted);$('rm-status').textContent=matches.length+' / '+all.length+' records match';
 const g=api.graph();if(!g||g.destroyed())return;
 const atlas=!api.state().object;
 if(!atlas){$('rm-collection').hidden=false;return}
 $('rm-collection').hidden=true;
 g.nodes().forEach(n=>{
  const id=n.id();if(!id.startsWith('object:'))return;
  const ok=matches.some(o=>id==='object:'+o.object_sha256);
  n.style('display',!ok&&sharpness===100?'none':'element');n.style('opacity',ok?1:Math.max(.07,1-sharpness/100));
 });
}
function schedule(){cancelAnimationFrame(frame);frame=requestAnimationFrame(sync)}
function sync(){const l=level(),g=api.graph();$('rm-level').textContent=l;$('rm-in').disabled=$('level-in').disabled;$('rm-out').disabled=$('level-out').disabled;if(l!==lastLevel){checked.clear();lastLevel=l;renderQuestions();closeInspector()}else if(g!==lastGraph){renderQuestions()}if(g&&g!==lastGraph)g.on('tap','node,edge',()=>openInspector());lastGraph=g;apply()}
const obs=new MutationObserver(schedule);obs.observe($('current-level'),{childList:true,characterData:true,subtree:true});obs.observe($('graph-area'),{childList:true,subtree:true});
// The source viewer recreates its graph on level and object changes.
document.addEventListener('click',()=>setTimeout(schedule,0));window.addEventListener('resize',schedule);sync();
})();
