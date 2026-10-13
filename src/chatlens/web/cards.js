(() => {
'use strict';
const $=id=>document.getElementById(id),TOKEN=window.CARD_TOKEN,preview=window.CARD_PREVIEW||null;
// Bootstrap names are only a preview/legacy fallback. Live identity always comes
// from the local service's canonical contact IDs, including same-name contacts.
let contacts=(Array.isArray(window.CARD_CONTACTS)?window.CARD_CONTACTS:[]).map(x=>typeof x==='string'?{id:x,name:x,analysis_enabled:true,available:true}:x);
if(preview&&!contacts.length)contacts=Object.keys(preview.contacts||{}).map(name=>({id:name,name,analysis_enabled:true,available:true}));
let target=contacts[0]?.id||null;
const nameCollator=new Intl.Collator('zh-CN-u-co-pinyin',{sensitivity:'base',numeric:true});
const compareContacts=(a,b)=>nameCollator.compare(a.sort_name||a.name,b.sort_name||b.name)||nameCollator.compare(a.name,b.name)||String(a.id).localeCompare(String(b.id));
const contactInitial=c=>String(c.sort_name||c.name).trim().match(/^[a-z]/i)?.[0].toUpperCase()||'#';
const contactLabel=c=>c.name.toLocaleLowerCase('en');
const currentContact=()=>contacts.find(x=>x.id===target);
const contactName=()=>state.displayName||currentContact()?.name||'请选择聊天';
const previewData=()=>preview?.contacts?.[target]||preview;
const state={ready:false,busy:false,polling:false,signature:'',selected:null,selectionNotice:'',revision:'',messages:[],result:null,proposal:null,job:null,jobIdentity:null,dirty:false,cursor:null,history:[],historyEpoch:0,historyRequest:0,historyLoading:false,historyMode:'recent',selectionChange:0,selecting:false,contactRevision:'',contactChange:0,switching:false,displayName:'',enabled:!!target,canAnalyze:!!target,requiresConsent:false,contactStatus:'',enabling:false,deferredRevision:'',catalogRevision:'',errorKind:''};
// The backend's active contact is shared with native clicks and other windows.
// Send dropdown changes in order so a slow earlier POST cannot restore an old choice.
let contactSwitchQueue=Promise.resolve();
// Serialize only the selection POSTs, never the much longer model work. The
// server also owns this selection, so ignoring an old response alone is not enough.
let selectionQueue=Promise.resolve();
const rawFile=!preview&&(!TOKEN||TOKEN==='__SESSION_TOKEN__');
if(window.CARD_COMPACT)document.body.classList.add('compact');
function node(tag,text,parent,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;if(parent)parent.append(e);return e}
function fail(e,kind='operation'){state.errorKind=kind;$('error').textContent=e.message||String(e);$('error').hidden=false}
function clearError(){state.errorKind='';$('error').hidden=true;$('error').textContent=''}
function native(action){if(window.webkit?.messageHandlers?.native)window.webkit.messageHandlers.native.postMessage({action,id:action+'-'+Date.now()});else if(preview)window.chatlensLayout({mode:action==='layout-dock'?'docked':'floating',hint:action==='layout-dock'?'贴靠微信右侧 · 拖动标题栏可自由放置':'自由窗口 · 位置已保留'})}
window.chatlensLayout=value=>{const dock=$('layout-dock');dock.classList.toggle('active',value.mode==='docked');dock.title=value.mode==='docked'?'已贴靠微信右侧':'贴靠微信右侧'};
$('layout-dock').onclick=()=>native('layout-dock');
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'X-ChatLens-Token':TOKEN,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(['/api/contact/enable','/api/history/sync'].includes(path)?30000:10000)});const v=await r.json();if(!r.ok){const e=Error(v.error||'本地服务不可用');e.status=r.status;throw e}return v}
function who(role){return role==='self'?'我':role==='other'?contactName():'发言方待核对'}
function when(timestamp){return new Date(timestamp*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})}
function score(value){return (Math.max(0,Math.min(1,Number.isFinite(value)?value:0))*10).toFixed(1)}
function contrastScores(rows){
 const values=rows.map(row=>Math.max(0,Math.min(1,Number.isFinite(row.score)?row.score:0)));
 if(values.length<2)return values;
 const hi=Math.max(...values),lo=Math.min(...values),span=hi-lo;
 // Preserve ties and already separated choices. Contrast changes presentation,
 // never the cached head scores, JEV order, winner, or writing instructions.
 const close=values.some((v,i)=>i>0&&values[i-1]>0&&(values[i-1]-v)/values[i-1]<.10);
 if(span<1e-12||!close)return values;
 const u=values.map(v=>(v-lo)/span),second=[...u].sort((a,b)=>b-a)[1];
 const target=second>.65?.65:second<.35?.35:second;
 const power=second>0&&second<1?Math.log(target)/Math.log(second):1;
 return u.map(v=>.4+.6/(1+Math.exp(-6*(Math.pow(v,power)-.5))));
}
const rankedByScore=rows=>[...(rows||[])].sort((a,b)=>(Number.isFinite(b.score)?b.score:-Infinity)-(Number.isFinite(a.score)?a.score:-Infinity));
function heading(label,model,parent){const h=node('h2',label,parent,'section-heading');if(model)node('span',model,h,'model');return h}
function card(value,parent,partial=false){
 const c=node('article',undefined,parent,'decision-card'+(partial?' partial':'')),p=value.proposal;
 const analysis=node('section',undefined,c,'card-section analysis-section');heading('分析','MiniCPM',analysis);node('p',p.analysis||(partial?(p.render_failed?'文案未完成':'正在组织分析…'):''),analysis,'analysis');
 const emotion=value.emotion||(partial&&p.jev_scored?p.emotion:null),hasScores=!partial||p.jev_scored===true;
 if(emotion?.available){const h=node('div','对方情感',analysis,'emotion-head');node('span','JEV类',h,'model');const grid=node('div',undefined,analysis,'emotion-grid');const ranked=rankedByScore(emotion.ranked).slice(0,3),display=contrastScores(ranked);ranked.forEach((e,i)=>{const chip=node('div',undefined,grid,'emotion-chip'+(i===0?' leading':''));node('span',e.label,chip);node('span',score(display[i])+'/10',chip,'emotion-score')})}
 const ranked=partial?(p.options||[]).slice(0,3):rankedByScore(value.decision?.ranked||p.options).slice(0,3);
 if(!partial||ranked.length){const actions=node('section',undefined,c,'card-section actions-section');heading('动作排名','JEV类',actions);const list=node('ol',undefined,actions,'action-list');
 const display=contrastScores(ranked);
 ranked.forEach((a,i)=>{const tone=!hasScores?'':i===0?' recommend-best':i===ranked.length-1?' recommend-low':' recommend-middle';const row=node('li',undefined,list,'action'+tone);node('span',String(i+1),row,'number');node('span',a.action,row,'label');const s=node('span',hasScores?score(display[i]):'—',row,'score');if(hasScores){node('span','/10',s,'score-max');const track=node('span',undefined,row,'score-track');const bar=node('i',undefined,track);bar.style.width=Math.max(0,Math.min(100,display[i]*100))+'%'}})}
 const next=node('section',undefined,c,'card-section next-section');heading('下一步','',next);node('p',partial?(p.render_failed?'文案未完成':'正在选择…'):ranked[0]?.detail||value.card?.next_step||'',next,'next-step');return c
}
function mayAnalyze(){return !!target&&state.canAnalyze&&state.enabled&&!state.requiresConsent&&!state.switching&&!state.enabling}
function updateSelectedControl(){$('analyze-selected').disabled=!mayAnalyze()||state.selected?.role!=='other'||state.busy||(!preview&&(!state.ready||!state.signature));$('refresh').disabled=!mayAnalyze()||state.selected?.role!=='other'||state.busy}
function renderGate(){const gated=!state.enabled||state.requiresConsent||!state.canAnalyze;$('consent').hidden=!gated||state.deferredRevision===state.contactRevision||rawFile;$('consent-enable').disabled=!target||!state.canAnalyze||state.enabling||state.switching||['ambiguous','unresolved','unsupported','unavailable'].includes(state.contactStatus);$('consent-enable').textContent=state.enabling?'读取中…':'分析';$('consent-defer').disabled=state.enabling||state.switching;$('open-history').disabled=gated;$('history-pane').hidden=gated;$('workspace').style.gridTemplateColumns=gated?'minmax(0,1fr)':'';if(gated){$('selected-message').hidden=true;$('candidates').hidden=true;$('status').textContent=''}updateSelectedControl()}
function render(){const feed=$('feed');feed.replaceChildren();renderGate();if(!state.enabled||state.requiresConsent||!state.canAnalyze)return;if(!state.result&&!state.proposal){const waiting=!!state.selected&&(state.busy||state.dirty||!state.signature);const empty=node('div',state.selectionNotice||(state.selected?'分析中…':'点击对方消息，查看对话建议'),feed,'empty-state'+(waiting?' pending':''));empty.id='empty';return}card(state.result||{proposal:state.proposal},feed,!state.result)}
function result(value){clearError();state.result=value;state.proposal=null;state.dirty=false;render();$('status').textContent=when(state.selected.timestamp)}
function highlight(){for(const b of document.querySelectorAll('.history-message'))b.classList.toggle('selected',b.dataset.messageId===state.selected?.id)}
function renderHistoryState(){const empty=$('history-empty');empty.hidden=state.history.length>0;empty.textContent=state.historyLoading?'正在读取聊天记录…':state.historyMode==='search'?'没有找到相关消息':'暂无聊天记录';$('older').disabled=state.historyLoading}
function renderHistory(){const list=$('history-list');list.replaceChildren();for(const m of state.history){const b=node('button',undefined,list,'history-message '+m.role);b.dataset.messageId=m.id;node('small',(who(m.role))+' · '+when(m.timestamp),b);node('span',m.text.slice(0,2000)+(m.text.length>2000?'…':''),b);b.onclick=()=>selectMessage(m.id)}highlight();renderHistoryState()}
function clearSelection(){state.revision='';state.selected=null;state.selectionNotice='';state.result=null;state.proposal=null;state.signature='';state.dirty=false;state.messages=[];$('candidates').hidden=true;showSelected();render()}
function rejectSelected(selection){clearSelection();state.revision=selection.revision||'';state.selectionNotice='请选择对方消息';clearError();render();$('status').textContent=state.selectionNotice}
function updateCatalog(rows){if(!Array.isArray(rows))return;contacts=rows.map(x=>({...x,id:x.id||x.username||x.name})).filter(x=>x.id&&x.name);renderContacts()}
function renderContacts(){const select=$('contact-select');select.replaceChildren();if(!target){const option=node('option',contactName().toLocaleLowerCase('en'),select);option.value=''}const groups=new Map();for(const c of contacts.filter(x=>x.available!==false||x.analysis_enabled||x.id===target).sort(compareContacts)){const key=contactInitial(c);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(c)}for(const key of [...groups.keys()].sort((a,b)=>a==='#'?1:b==='#'?-1:a.localeCompare(b))){const group=node('optgroup',undefined,select);group.label=key;for(const c of groups.get(key)){const option=node('option',contactLabel(c),group);option.value=c.id}}if(target&&!contacts.some(c=>c.id===target)){const option=node('option',contactName().toLocaleLowerCase('en'),select);option.value=target}select.value=target||''}
function applyContact(value,revision){if(typeof value!=='object'||value===null)value={active_target:value,contact_revision:revision};if(Array.isArray(value.contacts))updateCatalog(value.contacts);const next=value.active_target===undefined?target:value.active_target;const nextRevision=String(value.contact_revision??value.contactRevision??state.contactRevision);const entry=contacts.find(x=>x.id===next);const enabled=next?value.analysis_enabled??entry?.analysis_enabled??(!('requires_consent' in value)&&!('analysis_enabled' in value)):false;const requiresConsent=value.requires_consent??!enabled;const canAnalyze=!!next&&(value.can_analyze??(entry?.supported!==false&&entry?.available!==false));const nextName=value.display_name||entry?.name||(next===target?state.displayName:'')||'请选择聊天';const changed=target!==next||nextRevision!==state.contactRevision||state.enabled!==!!enabled||state.requiresConsent!==!!requiresConsent||state.canAnalyze!==canAnalyze||(!next&&nextName!==state.displayName);target=next;state.contactRevision=nextRevision;state.enabled=!!enabled;state.canAnalyze=canAnalyze;state.requiresConsent=!!requiresConsent;state.displayName=nextName;state.contactStatus=value.contact_status||value.status||'';if(entry&&'analysis_enabled' in value)entry.analysis_enabled=!!value.analysis_enabled;if(value.contact_catalog_revision)state.catalogRevision=String(value.contact_catalog_revision);renderContacts();if(changed){state.enabling=false;state.selecting=false;state.selectionChange++;state.historyRequest++;state.historyLoading=false;state.historyMode='recent';state.historyEpoch++;state.cursor=null;state.history=[];state.deferredRevision='';clearError();clearSelection();renderHistory();if(!preview&&!rawFile&&mayAnalyze())history()}else renderGate();return changed}
function identity(){return {target,contactRevision:state.contactRevision,revision:state.revision,signature:state.signature,messageId:state.selected?.id,selectionChange:state.selectionChange}}
function sameAnchor(value){return value&&value.target===target&&value.contactRevision===state.contactRevision&&value.revision===state.revision&&value.messageId===state.selected?.id&&value.selectionChange===state.selectionChange&&mayAnalyze()}
function isCurrent(value){return sameAnchor(value)&&value.signature===state.signature}
function resultMatchesAnchor(value,anchor){return value?.selected_message?.id===anchor.messageId&&(!value.target||value.target===anchor.target)&&(!value.contact_revision||String(value.contact_revision)===anchor.contactRevision)}
async function completeResult(value,request,jobId){
 // History sync can change the source signature after accept() reads context
 // but before the server constructs the analysis request. A completed result
 // must be checked against fresh context rather than silently discarded.
 const active=()=>sameAnchor(request)&&state.job===jobId&&state.jobIdentity===request;
 if(!active())return;
 if(!resultMatchesAnchor(value,request)){state.result=null;state.proposal=null;state.dirty=true;render();return}
 if(value.context_signature===state.signature){result(value);return}
 state.result=null;state.proposal=null;render();$('status').textContent='核对上下文…';
 try{
  const context=await api('/api/history/context',{target:request.target,through_id:request.messageId,contact_revision:request.contactRevision});
  if(!active())return;
  state.messages=context.messages;state.signature=context.context_signature;
  if(value.context_signature===state.signature)result(value);
  else{state.dirty=true;render();$('status').textContent='上下文已更新，重新分析…'}
 }catch(e){if(active()){state.dirty=false;render();fail(e)}}
}
function beginHistory(mode){state.historyMode=mode;state.historyLoading=true;renderHistoryState();return ++state.historyRequest}
function endHistory(request){if(request===state.historyRequest){state.historyLoading=false;renderHistoryState()}}
async function history(older=false){
 if(preview||rawFile||!mayAnalyze()||(state.historyLoading&&(older||state.historyMode==='recent')))return;
 const epoch=state.historyEpoch,requestedTarget=target,request=beginHistory('recent');
 const active=()=>request===state.historyRequest&&epoch===state.historyEpoch&&requestedTarget===target&&mayAnalyze();
 try{
  if(!older){try{await api('/api/history/sync',{target:requestedTarget})}catch(e){if(e.message!=='历史同步正在进行')throw e}if(!active())return;state.cursor=null;$('history-query').value=''}
  const previous=$('history-scroll').scrollHeight;
  const page=await api('/api/history/recent',{target:requestedTarget,limit:50,before:older?state.cursor:null});
  if(!active())return;
  state.cursor=page.before;
  // New messages may arrive between pages. Repeated anchors must remain one button.
  const seen=new Set();state.history=(older?[...page.messages,...state.history]:page.messages).filter(m=>!seen.has(m.id)&&seen.add(m.id));
  $('older').hidden=!page.has_more;renderHistory();$('history-scroll').scrollTop=older?$('history-scroll').scrollHeight-previous:$('history-scroll').scrollHeight;clearError();
 }catch(e){if(active())fail(e)}finally{endHistory(request)}
}
async function selectMessage(id){
 if(!mayAnalyze())return;
 if(preview){const data=previewData(),i=data.messages.findIndex(x=>x.id===id);if(i<0)return;if(data.messages[i].role!=='other'){rejectSelected({revision:id});return}state.selected=data.messages[i];state.selectionNotice='';state.revision=id;state.result=null;state.proposal=null;showSelected();result(data.results[i]);return}
 const requestedTarget=target,revision=state.contactRevision,change=++state.selectionChange;
 const active=()=>change===state.selectionChange&&requestedTarget===target&&revision===state.contactRevision&&mayAnalyze();
 state.selecting=true;
 try{
  const pending=selectionQueue.then(()=>active()?api('/api/history/select',{target:requestedTarget,message_id:id,contact_revision:revision}):null);
  selectionQueue=pending.catch(()=>{});
  const selection=await pending;
  if(active()){state.selecting=false;await accept(selection)}
 }catch(e){if(active()&&e.status!==409)fail(e)}finally{if(change===state.selectionChange)state.selecting=false;updateSelectedControl()}
}
function showSelected(){const selected=state.selected?.role==='other'&&state.enabled&&!state.requiresConsent&&state.canAnalyze;$('selected-message').hidden=!selected;$('selected-quote').textContent=selected?state.selected.text.slice(0,180):'';updateSelectedControl();highlight()}
async function accept(selection){if(!selection||selection.revision===state.revision)return;if(selection.target&&selection.target!==target)return;if(selection.contact_revision!==undefined&&String(selection.contact_revision)!==state.contactRevision)return;if(selection.status==='requires_consent'){state.requiresConsent=true;state.enabled=false;clearSelection();render();return}if(!mayAnalyze())return;if(['self_message','unknown_role','unknown_sender'].includes(selection.status)||(selection.status==='matched'&&selection.message?.role!=='other')){rejectSelected(selection);return}state.revision=selection.revision;state.selected=selection.status==='matched'?selection.message:null;state.selectionNotice='';state.result=null;state.proposal=null;state.signature='';state.dirty=false;render();showSelected();clearError();const box=$('candidates');box.replaceChildren();box.hidden=selection.status==='matched';if(selection.status!=='matched'){$('status').textContent=selection.status==='ambiguous'?'点击文本重复，请确认是哪一条。':'未能定位这条消息，可从本地历史准确选择。';for(const x of selection.candidates||[]){const b=node('button',when(x.message.timestamp)+' · '+x.message.text.slice(0,70),box);b.onclick=()=>selectMessage(x.message.id)}return}if(!state.selected?.id)return;const request=identity();$('status').textContent='读取中…';try{const context=await api('/api/history/context',{target:request.target,through_id:request.messageId,contact_revision:request.contactRevision});if(!isCurrent(request))return;state.messages=context.messages;state.signature=context.context_signature;state.dirty=true;if(state.busy)$('status').textContent='切换中…';else await analyze()}catch(e){if(isCurrent(request)){state.dirty=false;fail(e)}}}
async function analyze(){if(preview||rawFile||!mayAnalyze()||!state.ready||state.busy||state.selected?.role!=='other'||!state.signature)return;const request=identity();state.busy=true;state.jobIdentity=request;updateSelectedControl();try{const job=await api('/api/analyze/history',{target:request.target,through_id:request.messageId,contact_revision:request.contactRevision,workflow:'llm_then_jev',relationship:'未指定',goal:'只根据所选消息及之前的内容，说明对方这句话的心思。站在我方视角给出能直接选择的具体互动做法，沿用前文的事项和最新约定；不使用所选句之后的消息。'});state.job=job.id;if(isCurrent(request)){state.dirty=false;clearError()}}catch(e){state.busy=false;state.jobIdentity=null;if(isCurrent(request)){state.dirty=e.status===409;if(e.status!==409)fail(e)}}finally{updateSelectedControl()}}
async function poll(){
 if(preview||rawFile||state.polling||state.switching||state.selecting)return;
 state.polling=true;
 const change=state.contactChange,contactRevision=state.contactRevision,selectionChange=state.selectionChange;
 try{
  const s=await api('/api/status');
  // A status snapshot requested before a click must not clear that newer choice.
  if(state.switching||state.selecting||change!==state.contactChange||contactRevision!==state.contactRevision||selectionChange!==state.selectionChange)return;
  if(state.errorKind==='connection'){clearError();$('status').textContent=state.selected?when(state.selected.timestamp):'点击微信消息，或从本地历史选择一条。'}
  state.ready=s.ready;
  if('active_target' in s)applyContact(s);
  if(!mayAnalyze()){if(state.job&&s.job?.id===state.job&&s.job.status!=='running'){state.job=null;state.busy=false;state.jobIdentity=null}return}
  if(!s.selection&&state.revision){clearSelection();$('status').textContent='点击微信消息，或从本地历史选择一条。'}
  if(s.selection&&s.selection.revision!==state.revision){await accept(s.selection);return}
  if(!s.ready){$('status').textContent=s.phase;return}
  if(state.job&&s.job?.id===state.job){
   if(s.job.status==='running'){
    if(isCurrent(state.jobIdentity)){$('status').textContent=s.phase;if(s.job.proposal){state.proposal=s.job.proposal;render()}}
   }else{
    const jobId=state.job,jobIdentity=state.jobIdentity;
    if(s.job.status==='done')await completeResult(s.job.result,jobIdentity,jobId);
    if(state.job===jobId&&state.jobIdentity===jobIdentity){state.busy=false;state.job=null;state.jobIdentity=null}
    if(s.job.status==='error'&&isCurrent(jobIdentity)){if(s.job.proposal?.jev_scored){state.proposal={...s.job.proposal,render_failed:true};render()}fail(Error(s.job.error));state.dirty=false}
   }
  }else if(state.job){state.job=null;state.busy=false;state.jobIdentity=null;state.dirty=!!state.selected}
  if(!state.busy&&state.dirty&&!s.worker_busy&&s.job?.status!=='running')await analyze();
  if(!s.selection&&s.click_observer&&!s.click_observer.ax_locator&&(!s.click_observer.screen_capture||!s.click_observer.mouse_monitor))$('status').textContent='原生点击定位暂不可用，请从本地历史选择消息。';
 }catch(e){if(change===state.contactChange&&selectionChange===state.selectionChange){fail(e,'connection');$('status').textContent='本地服务连接中…'}}finally{state.polling=false;updateSelectedControl()}
}
$('older').onclick=()=>history(true);
$('history-reload').onclick=()=>history();
$('history-search').onclick=async()=>{
 if(preview||!mayAnalyze())return;
 const requestedTarget=target,epoch=state.historyEpoch,request=beginHistory('search');
 const active=()=>request===state.historyRequest&&epoch===state.historyEpoch&&requestedTarget===target&&mayAnalyze();
 try{
  const v=await api('/api/history/search',{target:requestedTarget,query:$('history-query').value.trim()});
  if(!active())return;
  state.history=[...v.messages].reverse();state.cursor=null;$('older').hidden=true;renderHistory();clearError();
 }catch(e){if(active())fail(e)}finally{endHistory(request)}
};
$('history-query').onkeydown=event=>{if(event.key==='Enter'&&!event.isComposing){event.preventDefault();$('history-search').onclick()}};
$('dismiss').onclick=()=>{if(window.webkit?.messageHandlers?.native)window.webkit.messageHandlers.native.postMessage({action:'hide-card',id:'hide-'+Date.now()})};
$('open-history').onclick=()=>{if(!mayAnalyze())return;if(window.webkit?.messageHandlers?.native)window.webkit.messageHandlers.native.postMessage({action:'open-history',id:'open-'+Date.now()});else document.getElementById('history-pane').scrollIntoView()};
$('refresh').onclick=async()=>{if(!mayAnalyze()||state.busy||state.selected?.role!=='other')return;if(preview){selectMessage(state.selected.id);return}const request=identity();try{await api('/api/clear',{});if(!isCurrent(request))return;state.dirty=true;await analyze()}catch(e){if(isCurrent(request))fail(e)}};
$('analyze-selected').onclick=()=>{if($('analyze-selected').disabled)return;return $('refresh').onclick()};
renderContacts();
$('contact-select').onchange=async()=>{const chosen=$('contact-select').value;if(!contacts.some(c=>c.id===chosen))return;const change=++state.contactChange;state.switching=true;state.enabling=false;clearSelection();state.historyEpoch++;state.history=[];renderHistory();$('status').textContent='正在切换联系人…';clearError();try{if(preview){applyContact(chosen,'preview-'+chosen);state.history=previewData().messages;renderHistory();$('status').textContent='点击消息，查看该位置的建议。';return}const pending=contactSwitchQueue.then(()=>api('/api/contact/select',{target:chosen}));contactSwitchQueue=pending.catch(()=>{});const v=await pending;if(change!==state.contactChange)return;applyContact(v);if(state.enabled&&!state.requiresConsent)$('status').textContent='点击微信消息，或从本地历史选择一条。';if(v.selection&&state.enabled&&!state.requiresConsent){state.switching=false;await accept(v.selection)}}catch(e){if(change===state.contactChange){renderContacts();fail(e)}}finally{if(change===state.contactChange){state.switching=false;renderGate();if(!preview&&mayAnalyze()&&!state.history.length)history()}}};
$('consent-enable').onclick=async()=>{if(!target||state.enabling||$('consent-enable').disabled)return;const requestedTarget=target,revision=state.contactRevision,change=++state.contactChange;state.enabling=true;clearError();render();try{const value=await api('/api/contact/enable',{target:requestedTarget,contact_revision:revision});if(change!==state.contactChange||target!==requestedTarget||state.contactRevision!==revision)return;applyContact(value);state.enabling=false;render();if(value.selection&&mayAnalyze())await accept(value.selection);else if(mayAnalyze())await poll()}catch(e){if(change===state.contactChange&&target===requestedTarget&&state.contactRevision===revision)fail(e)}finally{if(change===state.contactChange){state.enabling=false;renderGate();if(mayAnalyze()&&!state.history.length)history()}}};
$('consent-defer').onclick=async()=>{if(state.enabling||state.switching)return;const requestedTarget=target,revision=state.contactRevision,change=++state.contactChange;state.deferredRevision=revision;clearSelection();render();if(!requestedTarget)return;try{const value=await api('/api/contact/defer',{target:requestedTarget,contact_revision:revision});if(change!==state.contactChange||target!==requestedTarget||state.contactRevision!==revision)return;applyContact(value);state.deferredRevision=state.contactRevision;render()}catch(e){if(e.status!==404&&change===state.contactChange&&target===requestedTarget&&state.contactRevision===revision)fail(e)}};
async function initializeContacts(){const change=state.contactChange,revision=state.contactRevision;try{const v=await api('/api/contacts');updateCatalog(v.contacts);if(change!==state.contactChange||revision!==state.contactRevision)return;applyContact(v);if(mayAnalyze()&&!state.history.length)history();await poll()}catch(e){if(change===state.contactChange)fail(e)}}
window.ChatLensDebug=()=>({selected_id:state.selected?.id,revision:state.revision,signature:state.signature,result_selected_id:state.result?.selected_message?.id,job:state.job,busy:state.busy,target,contact_revision:state.contactRevision,analysis_enabled:state.enabled,requires_consent:state.requiresConsent,contacts_count:contacts.length});window.receiveNative=()=>{};window.chatlensSetVisible=()=>{};render();
if(preview){state.history=previewData().messages;renderHistory();$('older').hidden=true;$('status').textContent='点击左侧不同消息，查看该位置的建议。'}else if(rawFile){$('status').textContent='请打开「微信对话助手」使用真实记录；此页面需要本机服务。';$('older').hidden=true}else{initializeContacts();setInterval(poll,700)}
})();
