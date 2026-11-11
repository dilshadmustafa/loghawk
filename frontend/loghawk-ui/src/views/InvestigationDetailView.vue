<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { getInvestigation, investigationAction, previewInvestigationSource, uploadInvestigationFiles, listInvestigationS3, getBuckets, resolveInvestigationS3 } from "../services/api";
import type { InvestigationS3File } from "../services/api";
import type { Investigation, InvestigationPreview } from "../types/api";
const route=useRoute();
const record=ref<Investigation|null>(null), preview=ref<InvestigationPreview|null>(null);
const busy=ref(false), error=ref(""), notice=ref(""), sourceId=ref(""), sharedSource=ref(""), sharedIncident=ref("");
const evidenceType=ref("other"), recordPath=ref(""), files=ref<File[]>([]), uri=ref(""), nextToken=ref<string|null>(null);
const s3files=ref<InvestigationS3File[]>([]), s3selected=ref<string[]>([]);
const buckets=ref<string[]>([]), bucket=ref(""), s3folders=ref<{uri:string}[]>([]);
const reviewedFiles=ref<InvestigationS3File[]>([]), reviewed=ref(false), s3Message=ref("");
const browseRoot=computed(()=>bucket.value ? `s3://${bucket.value}/` : "");
const breadcrumbs=computed(()=>{
  const parts=uri.value.slice(browseRoot.value.length).split("/").filter(Boolean);
  return parts.map((name,index)=>({name,uri:browseRoot.value+parts.slice(0,index+1).join("/")+"/"}));
});
const parentFolder=computed(()=>breadcrumbs.value.length>1 ? breadcrumbs.value[breadcrumbs.value.length-2]!.uri : browseRoot.value);
const formatSize=(size:number)=>size<1024 ? `${size} B` : size<1048576 ? `${(size/1024).toFixed(1)} KB` : `${(size/1048576).toFixed(1)} MB`;
const entryName=(value:string)=>value.slice(uri.value.length) || value;
watch(s3selected,()=>{reviewed.value=false;reviewedFiles.value=[];},{deep:true});
function resetS3(){bucket.value="";uri.value="";s3files.value=[];s3folders.value=[];s3selected.value=[];nextToken.value=null;reviewed.value=false;reviewedFiles.value=[];s3Message.value="";}

const selected=ref<number[]>([]), allRecords=ref(false), mapping=ref("{}"), parseFormat=ref("json");
const finding=ref(""), supersedes=ref(""), aiRequest=ref(""), callId=ref("");
const pretty=(value:unknown)=>JSON.stringify(value,null,2);

type Tab = "overview" | "evidence" | "incidents" | "findings" | "history";
type ImportResult = {id:string; name:string; status:string; error?:string};
type RemovedEvidence = {source_id:string; incident_id:string|null; records:number[]|null};
type UndoItem = {id:number; label:string; kind:"evidence"; evidence:RemovedEvidence} | {id:number; label:string; kind:"incident"; incident_id:string} | {id:number; label:string; kind:"source"; source_id:string};
const undoItems=ref<UndoItem[]>([]), editTypeId=ref(""), editTypeValue=ref("other");
let nextUndoId=0;
type FindingReference = {source_id:string; relationship:"supports"|"contradicts"; rows:string};
const tab=ref<Tab>("overview"), addOpen=ref(false), closeOpen=ref(false);
const importMode=ref("s3"), importResults=ref<ImportResult[]>([]), ignoredFiles=ref(0);
const loading=ref(false), working=ref(""), previewOffset=ref(0), pageStarts=ref<number[]>([]);
const previewPanel=ref<HTMLElement|null>(null), findingReferences=ref<FindingReference[]>([]);
const copyNotice=ref(""), selectionDirty=ref(false);
let generation=0;
const evidenceLabels:Record<string,string>={"stage-a":"Stage A · Features","stage-b":"Stage B · Anomalies","stage-c":"Stage C · Incidents","raw-logs":"Raw logs",other:"Other"};
const tabs=computed(()=>[
  {id:"overview" as Tab,label:"Overview",count:null},
  {id:"evidence" as Tab,label:"Evidence",count:record.value?.sources.length||0},
  {id:"incidents" as Tab,label:"Incidents",count:record.value?.incidents.length||0},
  {id:"findings" as Tab,label:"Findings",count:record.value?.findings.length||0},
  {id:"history" as Tab,label:"RCA History",count:record.value?.requests.length||0},
]);
const asObject=(value:unknown):Record<string,unknown>=>value!==null && typeof value==="object" && !Array.isArray(value) ? value as Record<string,unknown> : {};
const display=(value:unknown):string=>value==null || value==="" ? "Not provided" : typeof value==="object" ? JSON.stringify(value) : String(value);
const date=(value:unknown)=>{if(!value)return "Not provided";const d=new Date(String(value));return Number.isNaN(d.getTime())?String(value):d.toLocaleString();};
const timeRange=computed(()=>asObject(record.value?.reported_context.time_range));
const currentFindings=computed(()=>record.value?.findings.filter(f=>!record.value?.findings.some(newer=>newer.supersedes_id===f.id))||[]);
const findingLinks=computed(()=>currentFindings.value.flatMap(f=>f.evidence.map(asObject)));
const allLinks=computed(()=>[...(record.value?.evidence_links||[]),...(record.value?.incident_evidence||[]),...findingLinks.value]);
const attachedCount=computed(()=>new Set(allLinks.value.map(link=>String(link.source_id))).size);
const failedSources=computed(()=>record.value?.sources.filter(source=>source.status==="FAILED" && asObject(source).is_directly_linked===1)||[]);
const readySources=computed(()=>record.value?.sources.filter(s=>s.status==="READY")||[]);
const previewColumns=computed(()=>[...new Set(preview.value?.records.flatMap(row=>Object.keys(row.original_record))||[])].slice(0,6));
const sourceName=(id:unknown)=>record.value?.sources.find(s=>s.id===id)?.name||String(id);
const incidentName=(id:unknown)=>{const incident=record.value?.incidents.find(i=>i.id===id);return incident?.source_incident_id||String(id);};
const selectionText=(selection:unknown)=>selection===null?"All records":Array.isArray(selection)?selection.length+" selected record(s)":"Record selection";
function attachmentRows(id:string){
  return [
    ...(record.value?.evidence_links.filter(l=>l.source_id===id).map(l=>({id:String(l.id),incident_id:null as string|null,label:"Investigation · "+selectionText(l.selection),removable:true}))||[]),
    ...(record.value?.incident_evidence.filter(l=>l.source_id===id).map(l=>({id:String(l.id),incident_id:String(l.incident_id),label:incidentName(l.incident_id)+" · "+selectionText(l.selection),removable:true}))||[]),
    ...findingLinks.value.filter(l=>l.source_id===id).map(l=>({id:String(l.id),incident_id:null as string|null,label:"Finding · "+String(l.relationship)+" · "+selectionText(l.selection),removable:false})),
  ];
}
function canRemoveDirectSource(id:string){
  const source=record.value?.sources.find(item=>item.id===id);
  return !!source && source.status==="READY" && asObject(source).is_directly_linked===1
    && !record.value?.evidence_links.some(link=>String(link.source_id)===id)
    && !findingLinks.value.some(link=>String(link.source_id)===id);
}
function editType(id:string){const source=record.value?.sources.find(s=>s.id===id);if(source){editTypeId.value=id;editTypeValue.value=source.evidence_type;}}
async function saveType(){
  const id=editTypeId.value,value=editTypeValue.value,owner=record.value?.id;
  if(await action(key=>investigationAction(key,`sources/${id}/type`,{evidence_type:value},"PUT"),"Evidence type updated. Previous RCA snapshots are unchanged.")){
    if(record.value?.id!==owner)return;
    editTypeId.value="";
    if(preview.value?.id===id)preview.value.evidence_type=value;
  }
}
function syncRemovedPreview(){
  if(preview.value&&!record.value?.sources.some(s=>s.id===preview.value?.id)){
    preview.value=null;sourceId.value="";selected.value=[];allRecords.value=false;selectionDirty.value=false;
  }else if(preview.value&&!selectionDirty.value)restoreSelection();
}
async function removeEvidence(id:string,incidentId:string|null=null){
  if(busy.value||!record.value?.can_modify)return;
  const owner=record.value.id,name=sourceName(id),destination=incidentId?"incident "+incidentName(incidentId):"this investigation";
  const shared=incidentId?" If this incident is shared, its evidence changes in every investigation linking to it.":"";
  if(!window.confirm(`Remove ${name} from ${destination}? The original file and saved RCA snapshots are retained. Other attachments may still include this evidence.${shared} You can Undo while this page is open.`))return;
  let result:RemovedEvidence|undefined;
  const saved=await action(async key=>{result=await investigationAction<RemovedEvidence>(key,"evidence/unlink",{source_id:id,incident_id:incidentId});return result;},"Evidence attachment removed. The original file is retained; other references may still include it.");
  if(saved&&result&&record.value?.id===owner){undoItems.value.push({id:++nextUndoId,label:`Removed ${name} from ${destination}`,kind:"evidence",evidence:result});syncRemovedPreview();}
}
async function unlinkIncident(id:string){
  if(busy.value||!record.value?.can_modify)return;
  const owner=record.value.id,name=incidentName(id);
  if(!window.confirm(`Unlink incident ${name} from this investigation? Its evidence remains with the incident, and other investigations are unchanged. Saved RCA snapshots are retained. You can Undo while this page is open.`))return;
  if(await action(key=>investigationAction(key,"incident-links/unlink",{incident_id:id}),"Incident unlinked from this investigation.")){
    if(record.value?.id===owner){undoItems.value.push({id:++nextUndoId,label:`Unlinked incident ${name}`,kind:"incident",incident_id:id});syncRemovedPreview();}
  }
}
async function removeSource(id:string){
  if(busy.value||!record.value?.can_modify)return;
  const owner=record.value.id,name=sourceName(id);
  if(!window.confirm(`Remove ${name} from the Evidence list? Stored files, records and previous RCA history are retained. You can Undo while this page is open.`))return;
  if(await action(key=>investigationAction(key,`sources/${id}/remove`,{}),"Imported source removed from the list. Stored evidence is retained.")){
    if(record.value?.id!==owner)return;
    undoItems.value.push({id:++nextUndoId,label:`Removed imported source ${name}`,kind:"source",source_id:id});
    importResults.value=importResults.value.filter(item=>item.id!==id);
    if(editTypeId.value===id)editTypeId.value="";
    syncRemovedPreview();
  }
}
async function undoRemoval(item:UndoItem){
  if(busy.value||!record.value?.can_modify)return;
  const owner=record.value.id;
  const saved=await action(key=>item.kind==="evidence"
    ?investigationAction(key,"evidence/restore",item.evidence)
    :item.kind==="source" ? investigationAction(key,`sources/${item.source_id}/restore`,{})
    :investigationAction(key,"incident-links",{incident_id:item.incident_id}),"Removal undone. Existing attachments are not overwritten.");
  if(saved&&record.value?.id===owner){undoItems.value=undoItems.value.filter(entry=>entry.id!==item.id);syncRemovedPreview();}
}
function restoreSelection(){
  const link=record.value?.evidence_links.find(l=>l.source_id===sourceId.value);
  selected.value=Array.isArray(link?.selection)?link.selection.filter((n):n is number=>typeof n==="number"):[];
  allRecords.value=!!link && link.selection===null;
  selectionDirty.value=false;
}
function openEvidence(){tab.value="evidence";addOpen.value=true;}
async function copy(value:string){try{await navigator.clipboard.writeText(value);copyNotice.value="Copied to clipboard.";}catch{copyNotice.value="Clipboard unavailable. Select and copy the ID from Technical details."}}
function moveTab(event:KeyboardEvent){
  if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
  event.preventDefault();const index=tabs.value.findIndex(t=>t.id===tab.value);
  const next=event.key==="Home"?0:event.key==="End"?tabs.value.length-1:(index+(event.key==="ArrowRight"?1:-1)+tabs.value.length)%tabs.value.length;
  tab.value=tabs.value[next]!.id;void nextTick(()=>document.getElementById("tab-"+tab.value)?.focus());
}
async function load(){
  const g=++generation,id=String(route.params.id);loading.value=true;
  try{const value=await getInvestigation(id);if(g===generation)record.value=value;}
  catch(e){if(g===generation)error.value=String(e);}
  finally{if(g===generation)loading.value=false;}
}
watch(()=>route.params.id,()=>{
  undoItems.value=[];editTypeId.value="";
  generation++;record.value=null;preview.value=null;sourceId.value="";selected.value=[];allRecords.value=false;
  error.value="";notice.value="";copyNotice.value="";tab.value="overview";addOpen.value=false;closeOpen.value=false;
  files.value=[];ignoredFiles.value=0;importResults.value=[];finding.value="";findingReferences.value=[];supersedes.value="";aiRequest.value="";
  sharedSource.value="";sharedIncident.value="";callId.value="";recordPath.value="";evidenceType.value="other";
  pageStarts.value=[];previewOffset.value=0;selectionDirty.value=false;resetS3();void load();
},{immediate:true});
async function action(fn:(id:string)=>Promise<unknown>, message="Saved."){
  if(busy.value||!record.value)return false;
  const id=record.value.id;busy.value=true;working.value="Saving changes…";error.value="";notice.value="";
  try{
    const result=await fn(id);
    if(String(route.params.id)!==id)return false;
    if(result && typeof result==="object" && "sources" in result && Array.isArray(result.sources) && !("problem" in result)){
      importResults.value=result.sources.map(asObject).map(r=>({id:String(r.id),name:String(r.name),status:String(r.status),error:r.error?String(r.error):undefined}));
      const failed=importResults.value.filter(r=>r.status==="FAILED").length;
      notice.value=`Import finished: ${importResults.value.length-failed} ready/existing, ${failed} failed. Preview a source to attach records.`;
    }else notice.value=message;
    await load();return true;
  }catch(e){if(String(route.params.id)===id){error.value=String(e);await load();}return false;}
  finally{busy.value=false;working.value="";}
}
function chooseFiles(event:Event){
  const candidates=Array.from((event.target as HTMLInputElement).files||[]);
  files.value=candidates.filter(f=>/\.(csv|parquet|json|jsonl|log|txt)$/i.test(f.name));
  ignoredFiles.value=candidates.length-files.value.length;
}
async function uploadLocal(){
  if(await action(id=>uploadInvestigationFiles(id,files.value,evidenceType.value,recordPath.value),"Files imported."))files.value=[];
}
async function showSource(id:string,offset=0){
  if(!record.value||busy.value)return;
  if(sourceId.value!==id && selectionDirty.value && !window.confirm("Discard the unsaved record selection and open another source?"))return;
  busy.value=true;working.value="Loading source preview…";error.value="";const owner=record.value.id;
  try{
    const value=await previewInvestigationSource(owner,id,offset);
    if(String(route.params.id)!==owner)return;
    if(sourceId.value!==id){pageStarts.value=[];sourceId.value=id;restoreSelection();}
    preview.value=value;previewOffset.value=offset;mapping.value=pretty(value.mapping);parseFormat.value=value.format;
    tab.value="evidence";await nextTick();previewPanel.value?.focus({preventScroll:true});previewPanel.value?.scrollIntoView({block:"nearest"});
  }catch(e){if(String(route.params.id)===owner)error.value=String(e);}
  finally{busy.value=false;working.value="";}
}
async function nextPreview(){if(preview.value?.next_offset!=null){const offset=previewOffset.value;await showSource(sourceId.value,preview.value.next_offset);if(previewOffset.value!==offset)pageStarts.value.push(offset);}}
async function previousPreview(){const offset=pageStarts.value.at(-1);if(offset!=null){await showSource(sourceId.value,offset);if(previewOffset.value===offset)pageStarts.value.pop();}}
function selectPage(){selected.value=[...new Set([...selected.value,...(preview.value?.records.map(r=>r.record_number)||[])])];selectionDirty.value=true;}
function closePreview(){if(selectionDirty.value&&!window.confirm("Discard the unsaved record selection?"))return;preview.value=null;sourceId.value="";selectionDirty.value=false;}
async function loadBuckets(){
  if(busy.value)return;busy.value=true;error.value="";
  const owner=String(route.params.id);
  try{const result=await getBuckets();if(String(route.params.id)===owner){buckets.value=result;s3Message.value=result.length?"Choose an input bucket.":"No internal S3 buckets are available.";}}
  catch(e){if(String(route.params.id)===owner)error.value=String(e);}finally{busy.value=false;}
}
async function browseS3(location:string,more=false){
  if(!record.value||busy.value||!location)return;
  const owner=record.value.id;busy.value=true;error.value="";s3Message.value="";
  if(!more){uri.value=location;s3files.value=[];s3folders.value=[];nextToken.value=null;}
  try{
    const result=await listInvestigationS3(owner,location,more?nextToken.value||"":"",true);
    if(String(route.params.id)!==owner)return;
    s3files.value=more?[...s3files.value,...result.files]:result.files;
    s3folders.value=more?[...s3folders.value,...(result.folders||[])]:result.folders||[];
    nextToken.value=result.next_token;
    if(!s3files.value.length&&!s3folders.value.length)s3Message.value=nextToken.value?"No supported files on this page. Load more entries.":"No supported files or subfolders here.";
  }catch(e){if(String(route.params.id)===owner)error.value=String(e);}finally{busy.value=false;}
}
function changeBucket(){
  s3selected.value=[];reviewed.value=false;reviewedFiles.value=[];
  if(bucket.value)void browseS3(browseRoot.value);
  else resetS3();
}
async function reviewS3(){
  if(!record.value||busy.value||!s3selected.value.length)return;
  const owner=record.value.id;busy.value=true;error.value="";reviewed.value=false;reviewedFiles.value=[];
  try{
    const result=await resolveInvestigationS3(owner,[...s3selected.value]);
    if(String(route.params.id)!==owner)return;
    reviewedFiles.value=result.files;reviewed.value=true;
    s3Message.value=result.files.length?"Review the files below before importing.":"The selection contains no supported files.";
  }catch(e){if(String(route.params.id)===owner)error.value=String(e);}finally{busy.value=false;}
}
function importReviewed(){
  if(!reviewed.value||!reviewedFiles.value.length)return;
  const objects=reviewedFiles.value.map(({uri,etag})=>({uri,etag}));
  return action(async id=>{
    const result=await investigationAction(id,"sources/s3",{objects,evidence_type:evidenceType.value,record_path:recordPath.value||null});
    if(String(route.params.id)===id){s3selected.value=[];reviewedFiles.value=[];reviewed.value=false;}
    return result;
  },"Import results — preview sources below and attach relevant records:");
}

watch([addOpen,importMode],()=>{if(addOpen.value && importMode.value==="s3" && !buckets.value.length)void loadBuckets();});
async function attach(){
  if(await action(id=>investigationAction(id,"evidence",{source_id:sourceId.value,records:allRecords.value?null:selected.value}),"Evidence selection saved for future RCA requests."))selectionDirty.value=false;
}
async function reparse(){
  if(await action(id=>investigationAction(id,`sources/${sourceId.value}/reparse`,{format:parseFormat.value,record_path:recordPath.value||null}),"Parsing finished. Review the source status."))await showSource(sourceId.value,0);
}
async function saveMapping(){
  if(await action(id=>investigationAction(id,`sources/${sourceId.value}/mapping`,{mapping:JSON.parse(mapping.value)},"PUT"),"Column mapping saved."))await showSource(sourceId.value,previewOffset.value);
}
function addReference(){findingReferences.value.push({source_id:"",relationship:"supports",rows:""});}
async function addFinding(){
  const saved=await action(id=>{
    const evidence=findingReferences.value.map(ref=>{
      if(!ref.source_id)throw new Error("Select a source for each evidence reference, or remove the unused reference.");
      const parts=ref.rows.trim()?ref.rows.split(",").map(n=>n.trim()):[];
      if(parts.some(n=>!/^\d+$/.test(n)||Number(n)<1||!Number.isSafeInteger(Number(n))))throw new Error("Enter record numbers separated by commas, for example 1, 2, 5. Leave blank for all records.");
      return {source_id:ref.source_id,relationship:ref.relationship,records:parts.length?[...new Set(parts.map(Number))]:null};
    });
    return investigationAction(id,"findings",{content:finding.value,evidence,supersedes_id:supersedes.value||null,request_id:aiRequest.value||null});
  },"Finding saved. Previous revisions remain available.");
  if(saved){finding.value="";findingReferences.value=[];supersedes.value="";aiRequest.value="";}
}
async function closeInvestigation(){
  if(await action(id=>investigationAction(id,"close",{}),"Investigation closed. Saved evidence and history remain available."))closeOpen.value=false;
}
async function clearFailedSources(selectedSourceId?:string){
  if(busy.value||!record.value?.can_modify)return;
  const count=selectedSourceId?1:failedSources.value.length;
  if(!count)return;
  const name=selectedSourceId?sourceName(selectedSourceId):count+" failed imports";
  if(!window.confirm("Clear "+name+" from this investigation's Evidence list? Only unreferenced failed imports will be cleared. Original files and failure details will be retained for troubleshooting."))return;
  const owner=record.value.id;
  let result:{cleared:string[];skipped:string[]}|undefined;
  const saved=await action(async id=>{
    result=await investigationAction<{cleared:string[];skipped:string[]}>(id,"sources/clear-failed",{source_id:selectedSourceId||null});
    return result;
  },"Failed imports cleared.");
  if(!saved||!result||String(route.params.id)!==owner)return;
  const cleared=new Set(result.cleared);
  importResults.value=importResults.value.filter(item=>!cleared.has(item.id));
  if(cleared.has(sourceId.value)){
    preview.value=null;sourceId.value="";selected.value=[];allRecords.value=false;
    selectionDirty.value=false;pageStarts.value=[];previewOffset.value=0;
  }
  notice.value=result.cleared.length+" failed import(s) cleared. "+result.skipped.length+" referenced source(s) skipped. Original files and failure details retained.";
}
function findingEvidence(value:unknown){const ref=asObject(value);return `${sourceName(ref.source_id)} · ${display(ref.relationship)} · ${selectionText(ref.selection)}`;}
</script>
<template>
<section class="view investigation">
  <RouterLink to="/rca/investigations">← Investigations</RouterLink>
  <p v-if="error" class="alert failure" role="alert">{{error}}</p>
  <p v-if="notice" class="alert" role="status">{{notice}}</p>
  <section v-if="undoItems.length" class="card undo-panel" aria-label="Undo removals">
    <h2>Recent removals</h2><p class="muted">Undo is available while this investigation page is open. Retained sources and incidents can also be linked again later. If both were removed, restore the incident before its evidence.</p>
    <ul><li v-for="item in undoItems" :key="item.id">{{item.label}} <button :disabled="busy||!record?.can_modify" @click="undoRemoval(item)">Undo</button></li></ul>
  </section>
  <p v-if="copyNotice" class="muted" role="status">{{copyNotice}}</p>
  <p v-if="loading || busy" class="muted" role="status">{{busy ? working || 'Working…' : 'Loading investigation…'}}</p>
  <template v-if="record">
    <header class="view-header">
      <div><div class="eyebrow">Investigation <span class="pill">{{record.status}}</span></div><h1>{{record.problem}}</h1><p>{{record.userid}} · Created {{date(record.created_at)}}</p></div>
      <div class="actions">
        <RouterLink v-if="record.can_modify" class="btn primary" :to="{path:'/rca/chat',query:{investigation:record.id}}">Open RCA Chat</RouterLink>
        <button :disabled="busy||loading" @click="load">Refresh</button>
        <details v-if="record.can_modify" class="menu"><summary>Actions</summary><button :disabled="busy" @click="closeOpen=true">Close Investigation…</button></details>
      </div>
    </header>
    <div v-if="closeOpen && record.can_modify" class="card close-confirm" role="region" aria-label="Confirm close investigation">
      <h2>Close this investigation?</h2><p>This ends editing and new RCA requests for this investigation. Saved incidents, files and request history remain available.</p>
      <div class="actions"><button :disabled="busy" @click="closeInvestigation">Yes, Close Investigation</button><button :disabled="busy" @click="closeOpen=false">Cancel</button></div>
    </div>
    <p v-if="!record.can_modify" class="alert">Read-only investigation. You can view sources, incidents, findings and saved RCA history.</p>
    <nav class="tabs" role="tablist" aria-label="Investigation sections" @keydown="moveTab">
      <button v-for="item in tabs" :id="'tab-'+item.id" :key="item.id" role="tab" :aria-selected="tab===item.id" :aria-controls="'panel-'+item.id" :tabindex="tab===item.id?0:-1" :class="{active:tab===item.id}" @click="tab=item.id">{{item.label}} <span v-if="item.count!==null" class="pill">{{item.count}}</span></button>
    </nav>

    <section v-show="tab==='overview'" id="panel-overview" role="tabpanel" aria-labelledby="tab-overview" class="stack">
      <div class="stats">
        <button @click="tab='evidence'"><strong>{{record.sources.length}}</strong>Imported sources</button>
        <button @click="tab='evidence'"><strong>{{attachedCount}}</strong>Sources attached to context</button>
        <button @click="tab='incidents'"><strong>{{record.incidents.length}}</strong>Linked incidents</button>
        <button @click="tab='findings'"><strong>{{currentFindings.length}}</strong>Current findings</button>
      </div>
      <div v-if="record.can_modify" class="card next-step">
        <h2>{{!record.sources.length?'Start by adding evidence':!attachedCount?'Choose the evidence for AI context':'Ready to investigate'}}</h2>
        <p>{{!record.sources.length?'Import incident files, anomalies, features or logs from your computer or internal S3.':!attachedCount?'Imported files are preserved but are not automatically included in AI context. Preview a source and attach relevant records.':'Open RCA Chat to ask questions using the linked incidents, selected evidence and current findings.'}}</p>
        <button v-if="!record.sources.length" class="primary" @click="openEvidence">Add Evidence</button>
        <button v-else-if="!attachedCount" class="primary" @click="tab='evidence'">Select Records</button>
        <RouterLink v-else class="btn primary" :to="{path:'/rca/chat',query:{investigation:record.id}}">Open RCA Chat</RouterLink>
      </div>
      <div class="card"><h2>Reported context</h2><p class="muted">These are reported details, not independently verified facts. Dates are shown in your browser's local timezone.</p>
        <dl class="context-grid"><div><dt>Entity</dt><dd>{{display(record.reported_context.entity_id)}}</dd></div><div><dt>Environment</dt><dd>{{display(record.reported_context.environment)}}</dd></div><div><dt>Start</dt><dd>{{date(timeRange.start)}}</dd></div><div><dt>End</dt><dd>{{date(timeRange.end)}}</dd></div></dl>
        <details><summary>All reported details</summary><pre>{{pretty(record.reported_context)}}</pre></details>
      </div>
      <div class="card"><h2>How evidence becomes context</h2><ol class="steps"><li><strong>Import</strong> — preserve an original file and parse its records.</li><li><strong>Attach</strong> — select records for the investigation, an incident or a finding.</li><li><strong>Investigate</strong> — each RCA request saves its own context snapshot. Model inputs are redacted and may be compacted.</li></ol><p>Changing evidence affects future requests. Previous RCA requests retain the context they originally used.</p></div>
      <details class="card"><summary>Technical details</summary><p>Investigation ID: <code>{{record.id}}</code> <button @click="copy(record.id)">Copy ID</button></p><p>Last updated: {{date(record.updated_at)}}</p></details>
    </section>

    <section v-show="tab==='evidence'" id="panel-evidence" role="tabpanel" aria-labelledby="tab-evidence" class="stack">
      <div class="section-heading"><div><h2>Evidence</h2><p class="muted">Imported does not mean attached. Select records to include them in this investigation's AI context.</p></div><div v-if="record.can_modify" class="actions"><button v-if="failedSources.length" :disabled="busy" @click="clearFailedSources()">Clear Failed Imports ({{failedSources.length}})</button><button class="primary" :disabled="busy" @click="addOpen=!addOpen">{{addOpen?'Cancel Add Evidence':'Add Evidence'}}</button></div></div>
      <section v-if="addOpen && record.can_modify" class="card">
        <h3>Add Evidence</h3>
        <fieldset :disabled="busy">
          <label>Source<select v-model="importMode"><option value="s3">Internal S3</option><option value="local">Local files / Folder</option><option value="collector">Saved collector evidence</option><option value="existing">Existing source</option></select></label>
          <template v-if="importMode==='local'||importMode==='s3'">
            <label>Evidence type<select v-model="evidenceType"><option v-for="(label,key) in evidenceLabels" :key="key" :value="key">{{label}}</option></select></label>
            <details><summary>Advanced import options</summary><label>Nested JSON record path (JSON Pointer)<input v-model="recordPath" placeholder="/incidents" /></label><p class="muted">Leave blank for flat files. Use this only when records are inside a nested JSON object.</p></details>
          </template>
          <section v-if="importMode==='local'">
            <div class="form-grid"><label>Choose files<input type="file" multiple accept=".csv,.parquet,.json,.jsonl,.log,.txt" @change="chooseFiles" /></label><label>Choose a folder<input type="file" multiple webkitdirectory @change="chooseFiles" /></label></div>
            <p>{{files.length}} supported file(s) selected. <span v-if="ignoredFiles">{{ignoredFiles}} unsupported file(s) excluded.</span></p>
            <ul class="file-list"><li v-for="(file,index) in files" :key="index">{{file.webkitRelativePath||file.name}} · {{formatSize(file.size)}}</li></ul>
            <button class="primary" :disabled="!files.length" @click="uploadLocal">Upload Selected Files</button>
            <p class="muted">CSV, Parquet, JSON, JSONL, LOG and TXT. Selecting files or a folder replaces the current local selection.</p>
          </section>
          <template v-if="importMode==='s3'"><section class="s3-picker" aria-labelledby="s3-title">
<h3 id="s3-title">Internal S3</h3>
<div class="s3-toolbar">
<label>Input bucket<select v-model="bucket" @change="changeBucket"><option value="">Select an input bucket</option><option v-for="name in buckets" :key="name" :value="name">{{name}}</option></select></label>
<button type="button" @click="loadBuckets">Refresh Buckets</button>
</div>
<template v-if="bucket">
<nav class="s3-breadcrumbs" aria-label="S3 folder location">
<button type="button" @click="browseS3(browseRoot)">{{bucket}}</button>
<span v-for="crumb in breadcrumbs" :key="crumb.uri"> / <button type="button" @click="browseS3(crumb.uri)">{{crumb.name}}</button></span>
</nav>
<div class="s3-toolbar">
<button type="button" :disabled="uri===browseRoot" @click="browseS3(parentFolder)">Up one folder</button>
<button type="button" @click="browseS3(uri)">Refresh Folder</button>
<button type="button" :disabled="s3selected.includes(uri)" @click="s3selected.push(uri)">Select This Folder</button>
</div>
<p class="s3-location">{{uri}}</p>
<div class="s3-table-wrap"><table class="s3-table">
<thead><tr><th>Select</th><th>Name</th><th>Type</th><th>Size</th></tr></thead>
<tbody>
<tr v-for="folder in s3folders" :key="folder.uri"><td><input v-model="s3selected" type="checkbox" :value="folder.uri" :aria-label="'Select folder '+entryName(folder.uri)" /></td><td><button type="button" @click="browseS3(folder.uri)">{{entryName(folder.uri)}}</button></td><td>Folder</td><td>—</td></tr>
<tr v-for="file in s3files" :key="file.uri"><td><input v-model="s3selected" type="checkbox" :value="file.uri" :aria-label="'Select file '+entryName(file.uri)" /></td><td>{{entryName(file.uri)}}</td><td>{{file.uri.split(".").pop()?.toUpperCase()}}</td><td>{{formatSize(file.size)}}</td></tr>
</tbody></table></div>
<button v-if="nextToken" type="button" @click="browseS3(uri,true)">Load More Entries</button>
<p>Supported files: CSV, Parquet, JSON, JSONL, LOG and TXT. Selecting a folder includes supported files in all subfolders.</p>
<details v-if="s3selected.length"><summary>{{s3selected.length}} file/folder selection(s)</summary><ul><li v-for="value in s3selected" :key="value">{{value}} <button type="button" @click="s3selected=s3selected.filter(item=>item!==value)">Remove</button></li></ul></details>
<div class="s3-toolbar"><button type="button" :disabled="!s3selected.length" @click="reviewS3">Review Selected Files</button><button type="button" :disabled="!s3selected.length" @click="s3selected=[]">Clear Selection</button></div>
</template>
<p v-if="s3Message" role="status">{{s3Message}}</p>
<section v-if="reviewed" class="s3-review" aria-labelledby="s3-review-title">
<h4 id="s3-review-title">Review import · {{reviewedFiles.length}} file(s) · {{formatSize(reviewedFiles.reduce((sum,file)=>sum+file.size,0))}}</h4>
<p>Overlapping selections are counted once. Remove any files you do not want to import.</p>
<ul><li v-for="file in reviewedFiles" :key="file.uri">{{file.uri}} · {{formatSize(file.size)}} <button type="button" @click="reviewedFiles=reviewedFiles.filter(item=>item.uri!==file.uri)">Remove</button></li></ul>
<button type="button" :disabled="!reviewedFiles.length" @click="importReviewed">Import {{reviewedFiles.length}} Selected File(s)</button>
<p>Import preserves the files. Preview and attach records below to include them in AI context.</p>
</section>
</section>
</template>
          <section v-if="importMode==='existing'"><label>Existing owned source ID<input v-model="sharedSource" /></label><p>Attaches all records from a ready source owned by this investigation's owner. Copy its ID from a source's Technical details.</p><button :disabled="!sharedSource.trim()" @click="action(id=>investigationAction(id,'evidence',{source_id:sharedSource.trim(),records:null}),'Existing source attached to investigation.')">Attach Existing Source</button></section>
          <section v-if="importMode==='collector'"><label>Collector tool-call event ID<input v-model="callId" /></label><p>Use a saved tool-call event ID from RCA History. This references saved collector output; it does not execute a collector.</p><button :disabled="!callId.trim()" @click="action(id=>investigationAction(id,'sources/collector',{tool_call_id:callId.trim()}),'Collector source registered. Preview it below to attach its evidence.')">Reference Saved Collector Evidence</button></section>
        </fieldset>
      </section>
      <div v-if="importResults.length" class="card"><h3>Latest import</h3><ul class="result-list"><li v-for="item in importResults" :key="item.id"><span class="pill" :class="{failed:item.status==='FAILED'}">{{item.status}}</span> {{item.name}} <button :disabled="busy" @click="showSource(item.id)">{{item.status==='FAILED'?'Inspect / Retry':'Preview and Attach'}}</button><p v-if="item.error" class="error">{{item.error}}</p></li></ul></div>
      <div v-if="!record.sources.length" class="card empty"><h3>No evidence imported yet</h3><p>Add files or saved collector evidence to start building context.</p></div>
      <div v-else class="card table-wrap"><table><caption class="sr-only">Evidence sources and their attachments</caption><thead><tr><th>File / source</th><th>Type</th><th>Import status</th><th>Records</th><th>Attached to</th><th>Action</th></tr></thead><tbody>
        <tr v-for="source in record.sources" :key="source.id"><td>{{source.name}}<p v-if="source.error" class="error">{{source.error}}</p></td><td>
          <template v-if="editTypeId===source.id && record.can_modify"><label>Evidence type<select v-model="editTypeValue" :disabled="busy"><option v-for="(label,key) in evidenceLabels" :key="key" :value="key">{{label}}</option></select></label><div class="actions"><button :disabled="busy" @click="saveType">Save</button><button :disabled="busy" @click="editTypeId=''">Cancel</button></div><p class="muted">Updates this source wherever it is referenced. Saved RCA snapshots stay unchanged.</p></template>
          <template v-else>{{evidenceLabels[source.evidence_type]||source.evidence_type}}<button v-if="record.can_modify && ['READY','FAILED'].includes(source.status)" :disabled="busy" @click="editType(source.id)">Change Type</button></template>
        </td><td><span class="pill" :class="{failed:source.status==='FAILED'}">{{source.status}}</span></td><td>{{source.record_count}}</td><td><ul v-if="attachmentRows(source.id).length" class="compact-list"><li v-for="link in attachmentRows(source.id)" :key="link.id">{{link.label}}<button v-if="record.can_modify && link.removable" :disabled="busy" @click="removeEvidence(source.id,link.incident_id)">Remove Attachment</button><button v-else-if="record.can_modify" @click="tab='findings'">Revise Finding</button></li></ul><span v-else class="muted">Not attached</span></td><td><button :disabled="busy" @click="showSource(source.id)">{{source.status==='FAILED'?'Inspect error':'Preview / Select'}}</button><button v-if="record.can_modify && canRemoveDirectSource(source.id)" :disabled="busy" :aria-label="'Remove imported source '+source.name" @click="removeSource(source.id)">Remove</button><button v-if="record.can_modify && failedSources.some(item=>item.id===source.id)" :disabled="busy" :aria-label="'Remove failed import '+source.name" @click="clearFailedSources(source.id)">Remove</button></td></tr>
      </tbody></table></div>

      <section v-if="preview" ref="previewPanel" tabindex="-1" class="card preview">
        <div class="section-heading"><div><span class="eyebrow">Source preview</span><h3>{{preview.name}}</h3><span class="pill">{{preview.status}}</span> · {{preview.record_count}} records</div><button :disabled="busy" @click="closePreview">Close Preview</button></div>
        <details><summary>Technical details and source location</summary><p>{{preview.original_uri || (preview.tool_call_id?'Saved collector evidence':'Local upload')}}</p><p>Source ID: <code>{{preview.id}}</code> <button @click="copy(preview.id)">Copy ID</button></p><p>SHA-256: <code>{{preview.sha256||'Not applicable'}}</code></p><p v-if="preview.tool_call_id">Tool call: <code>{{preview.tool_call_id}}</code></p></details>
        <div v-if="preview.status==='FAILED'" class="alert failure"><h4>Import failed</h4><p>{{preview.error||'The source could not be parsed.'}}</p><fieldset v-if="record.can_modify" :disabled="busy"><label>Parse format<select v-model="parseFormat"><option v-for="format in ['csv','parquet','json','jsonl','log','txt']" :key="format">{{format}}</option></select></label><label>Nested JSON path (optional)<input v-model="recordPath" placeholder="/incidents" /></label><button @click="reparse">Retry Parsing Preserved Source</button></fieldset></div>
        <details v-if="preview.status==='READY' && record.can_modify"><summary>Advanced: column mapping</summary><fieldset :disabled="busy"><p>Map standard incident fields to source columns, for example {"incident_id":"case_id"}. Mapping becomes fixed after incident evidence is linked.</p><label>Mapping JSON<textarea v-model="mapping" rows="4" /></label><button @click="saveMapping">Save Column Mapping</button></fieldset></details>
        <fieldset v-if="record.can_modify && preview.status==='READY'" :disabled="busy" class="selection-controls">
          <label class="check"><input v-model="allRecords" type="checkbox" @change="selectionDirty=true" />All records in this source</label>
          <button :disabled="allRecords||!preview.records.length" @click="selectPage">Select This Page</button><button :disabled="allRecords||!selected.length" @click="selected=[];selectionDirty=true">Clear Row Selection</button>
        </fieldset>
        <p v-if="!preview.records.length" class="muted">No parsed records to display.</p>
        <div v-else class="table-wrap record-table"><table><caption class="sr-only">Source records, showing up to six columns; expand JSON for all fields</caption><thead><tr><th v-if="record.can_modify">Select</th><th>Record</th><th v-for="column in previewColumns" :key="column">{{column}}</th><th>All fields</th></tr></thead><tbody>
          <tr v-for="row in preview.records" :key="row.id"><td v-if="record.can_modify"><input v-model="selected" type="checkbox" :value="row.record_number" :disabled="busy||allRecords||preview.status!=='READY'" :aria-label="'Select record '+row.record_number" @change="selectionDirty=true" /></td><td>{{row.record_number}}<small v-if="row.source_line_number">Line {{row.source_line_number}}</small></td><td v-for="column in previewColumns" :key="column"><span class="cell-value" :title="display(row.original_record[column])">{{display(row.original_record[column])}}</span></td><td><details><summary>JSON</summary><pre>{{pretty(row.original_record)}}</pre></details></td></tr>
        </tbody></table></div>
        <div class="actions pagination"><button :disabled="busy||!pageStarts.length" @click="previousPreview">Previous</button><span>{{preview.records.length?previewOffset+1:0}}–{{previewOffset+preview.records.length}} of {{preview.record_count}} records</span><button :disabled="busy||preview.next_offset===null" @click="nextPreview">Next</button></div>
        <div v-if="record.can_modify && preview.status==='READY'" class="attach-bar"><p>{{allRecords?'All '+preview.record_count+' records':selected.length+' record(s) selected across pages'}} <strong v-if="selectionDirty">· Unsaved selection</strong></p><button class="primary" :disabled="busy||(!allRecords&&!selected.length)" @click="attach">Save Evidence Selection</button><p class="muted">Save the selected records for the target chosen above. The investigation and each linked incident keep separate selections for this file. Changes apply to future RCA chats; earlier requests keep the evidence they already used.</p></div>
      </section>
    </section>

    <section v-show="tab==='incidents'" id="panel-incidents" role="tabpanel" aria-labelledby="tab-incidents" class="stack">
      <div class="section-heading"><div><h2>Incidents</h2><p class="muted">Link existing incidents to this investigation, or unlink them. Manage incident evidence from the Evidence tab or the incident detail page.</p></div></div>
      <div v-if="!record.incidents.length" class="card empty"><h3>No incidents linked</h3><p>Create an incident first, then link it here using its internal ID. <RouterLink to="/rca/incidents/new">Create Incident</RouterLink></p></div>
      <article v-for="incident in record.incidents" :key="incident.id" class="card">
        <div class="section-heading"><h3>{{incident.source_incident_id||'User-selected incident'}}</h3><span class="pill">{{display(incident.primary_details.max_severity)}}</span></div>
        <dl class="context-grid"><div><dt>Start</dt><dd>{{date(incident.primary_details.start_time)}}</dd></div><div><dt>End</dt><dd>{{date(incident.primary_details.end_time)}}</dd></div><div><dt>Primary service</dt><dd>{{display(incident.primary_details.primary_service)}}</dd></div><div><dt>Services</dt><dd>{{display(incident.primary_details.services)}}</dd></div></dl>
        <h4>Attached evidence</h4><ul v-if="record.incident_evidence.some(link=>link.incident_id===incident.id)" class="compact-list"><li v-for="link in record.incident_evidence.filter(l=>l.incident_id===incident.id)" :key="String(link.id)"><button :disabled="busy" @click="showSource(String(link.source_id))">{{sourceName(link.source_id)}}</button> · {{selectionText(link.selection)}}</li></ul><p v-else class="muted">No evidence attached to this incident.</p>
        <div v-if="record.can_modify" class="actions"><button :disabled="busy" @click="unlinkIncident(incident.id)">Unlink Incident</button></div>
        <details><summary>All incident details</summary><p>Internal ID: <code>{{incident.id}}</code> <button @click="copy(incident.id)">Copy ID</button></p><pre>{{pretty(incident.primary_details)}}</pre></details>
      </article>
      <details v-if="record.can_modify" class="card"><summary>Link an existing incident</summary><fieldset :disabled="busy"><label>Existing owned incident ID<input v-model="sharedIncident" /></label><p>Use the internal ID from the incident's details. The incident must belong to this investigation's owner.</p><button :disabled="!sharedIncident.trim()" @click="action(id=>investigationAction(id,'incident-links',{incident_id:sharedIncident.trim()}),'Incident linked.')">Link Incident</button></fieldset></details>
    </section>

    <section v-show="tab==='findings'" id="panel-findings" role="tabpanel" aria-labelledby="tab-findings" class="stack">
      <h2>Findings</h2><p class="muted">Record observations and saved AI assessments. New revisions preserve earlier findings; only current revisions enter new RCA context.</p>
      <div v-if="!record.findings.length" class="card empty">No findings saved yet.</div>
      <article v-for="item in record.findings" :key="item.id" class="card">
        <div class="section-heading"><strong>{{item.origin==='AI'?'AI assessment':'User observation'}}</strong><span class="pill">{{currentFindings.some(f=>f.id===item.id)?'Current':'Superseded'}}</span></div>
        <p class="muted">{{item.userid}} · {{date(item.created_at)}}</p><p class="finding-content">{{item.content}}</p>
        <ul v-if="item.evidence.length" class="compact-list"><li v-for="(ref,index) in item.evidence" :key="index">{{findingEvidence(ref)}}</li></ul>
        <RouterLink v-if="item.request_id" :to="'/rca/history/'+item.request_id">View RCA request</RouterLink>
        <details><summary>Revision details</summary><p>ID: <code>{{item.id}}</code> <button @click="copy(item.id)">Copy ID</button></p><p v-if="item.supersedes_id">Revision of {{item.supersedes_id}}</p></details>
      </article>
      <form v-if="record.can_modify" class="card" @submit.prevent="addFinding"><h3>Add Finding</h3><fieldset :disabled="busy">
        <label>Finding source<select v-model="aiRequest"><option value="">My observation</option><option v-for="item in record.requests.filter(r=>!r.evidence_removed)" :key="item.request_id" :value="item.request_id">Saved AI response · {{date(item.created_at)}} · {{item.request_id}}</option></select></label>
        <p v-if="aiRequest">The server will copy the saved response for this request. Review it in RCA History first; it may include a failure message.</p>
        <label v-else>Observation<textarea v-model="finding" rows="4" maxlength="20000" placeholder="What did you observe? What supports or contradicts it?" required /></label>
        <label>Revision<select v-model="supersedes"><option value="">New finding</option><option v-for="item in record.findings" :key="item.id" :value="item.id">{{date(item.created_at)}} · {{item.content.slice(0,90)}}</option></select></label>
        <h4>Evidence references (optional)</h4><p class="muted">Referenced records become part of the investigation context. Blank record numbers mean all records in that source.</p>
        <div v-for="(ref,index) in findingReferences" :key="index" class="reference-row">
          <label>Source<select v-model="ref.source_id" required><option value="">Choose source</option><option v-for="source in readySources" :key="source.id" :value="source.id">{{source.name}} · {{source.original_uri || (source.tool_call_id ? 'Saved collector evidence' : 'Local upload')}} · {{source.id.slice(0,8)}}</option></select></label>
          <label>Relationship<select v-model="ref.relationship"><option value="supports">Supports</option><option value="contradicts">Contradicts</option></select></label>
          <label>Record numbers<input v-model="ref.rows" placeholder="Blank for all; e.g. 1, 2, 5" /></label><button type="button" @click="findingReferences.splice(index,1)">Remove</button>
        </div>
        <div class="actions"><button type="button" :disabled="!readySources.length||findingReferences.length>=100" @click="addReference">Add Evidence Reference</button><button type="submit" class="primary" :disabled="!finding.trim()&&!aiRequest">Save Finding</button></div>
      </fieldset></form>
    </section>

    <section v-show="tab==='history'" id="panel-history" role="tabpanel" aria-labelledby="tab-history" class="stack">
      <h2>RCA History</h2><p class="muted">Each request retains the context it originally used. Open its details for evidence, input arguments, execution results, explanation attempts and LLM usage.</p>
      <div v-if="!record.requests.length" class="card empty">No RCA requests yet. Open RCA Chat with this investigation to start.</div>
      <div v-else class="card table-wrap"><table><thead><tr><th>Created</th><th>Status</th><th>Client</th><th>Request</th></tr></thead><tbody><tr v-for="item in record.requests" :key="item.request_id"><td>{{date(item.created_at)}}</td><td>{{item.evidence_removed?'Evidence removed by retention':item.status}}</td><td>{{item.origin_software||'—'}}</td><td><code v-if="item.evidence_removed">{{item.request_id}}</code><RouterLink v-else :to="'/rca/history/'+item.request_id">{{item.request_id}}</RouterLink><button @click="copy(item.request_id)">Copy ID</button></td></tr></tbody></table></div>
    </section>
  </template>
  <p v-else-if="!loading && !error" class="muted">No investigation loaded.</p>
</section>
</template>
<style scoped>
.undo-panel li{margin:.6rem 0;overflow-wrap:anywhere}.undo-panel button{margin-left:.5rem}
.investigation{max-width:1400px}.stack{display:grid;gap:1rem;min-width:0}
h1{overflow-wrap:anywhere;white-space:pre-wrap}h2,h3,h4{margin:.3rem 0 .7rem}
.eyebrow{font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;color:#aebbc7;margin-bottom:.5rem}
.section-heading{display:flex;justify-content:space-between;align-items:start;gap:1rem;flex-wrap:wrap}
.section-heading p{margin:.3rem 0}.view-header{flex-wrap:wrap}.view-header>div:first-child{flex:1;min-width:200px}
button{padding:.55rem .75rem;color:#e8edf2;background:#243442;border:1px solid #405665;border-radius:.45rem;cursor:pointer;font:inherit}
button:hover:not(:disabled){border-color:#6bd7bb;background:#2b414c}
button:disabled{opacity:.45;cursor:not-allowed}button.primary{background:#6bd7bb;color:#08131b;font-weight:700}
button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}
fieldset{border:0;padding:0;margin:0;min-width:0}label{display:grid;gap:.4rem;margin:.65rem 0;color:#bbc9d2;font-size:.9rem}
input:not([type=checkbox]),select,textarea{width:100%;min-width:0;padding:.65rem;color:#e8edf2;background:#17232d;border:1px solid #405665;border-radius:.4rem;font:inherit}
textarea{resize:vertical}input[type=checkbox]{width:1rem;height:1rem;accent-color:#6bd7bb}.check{display:flex;align-items:center}
details{margin:.8rem 0}summary{cursor:pointer;color:#b8d9d3;padding:.25rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:22rem;overflow:auto;background:#0d151c;padding:.8rem;border-radius:.4rem}
code{overflow-wrap:anywhere}.alert{padding:.85rem 1rem;background:#172d30;border:1px solid #355c59;border-radius:.5rem;overflow-wrap:anywhere}
.failure{background:#34242a;border-color:#89515c}.failed{background:#572e38;color:#ffc4cc}
.tabs{display:flex;gap:.3rem;overflow-x:auto;border-bottom:1px solid #344553;padding-bottom:.5rem}
.tabs button{white-space:nowrap;border-color:transparent;background:transparent}.tabs button.active{background:#203d3b;color:#8fe5ce;border-color:#47766b}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem}.stats button{text-align:left;padding:1rem;background:#131f28}.stats strong{display:block;font-size:1.7rem;margin-bottom:.3rem;color:#8fe5ce}
.context-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.context-grid dt{color:#93a7b6;font-size:.85rem}.context-grid dd{margin:.4rem 0;overflow-wrap:anywhere}
.steps{padding-left:1.25rem;line-height:1.9}.empty{padding:2rem;text-align:center;color:#aebbc7}
.file-list,.result-list,.s3-review ul{max-height:18rem;overflow:auto;padding-left:1.3rem}.result-list li{padding:.5rem 0}.compact-list{padding-left:1.1rem;margin:.25rem 0}.compact-list li{padding:.2rem 0}
.table-wrap td{vertical-align:top;overflow-wrap:anywhere}.record-table{max-height:28rem;overflow:auto}.record-table th{position:sticky;top:0;background:#17232d}
.cell-value{display:block;min-width:7rem;max-width:20rem;max-height:4.5rem;overflow:auto;white-space:pre-wrap}.record-table pre{min-width:18rem;max-width:40rem}
small{display:block;color:#93a7b6}.pagination{margin:1rem 0;justify-content:space-between}
.attach-bar{padding:1rem;background:#172d30;border-radius:.5rem;margin:.8rem 0}.attach-bar p{margin:.4rem 0 .7rem}
.reference-row{display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:.6rem;align-items:end}.reference-row button{margin-bottom:.65rem}
.finding-content{white-space:pre-wrap;overflow-wrap:anywhere;max-height:24rem;overflow:auto}
.s3-toolbar{display:flex;align-items:end;gap:.5rem;flex-wrap:wrap}.s3-toolbar label{flex:1;min-width:12rem}
.s3-breadcrumbs{margin:.5rem 0;overflow-wrap:anywhere}.s3-location{overflow-wrap:anywhere;font-family:monospace}
.s3-table-wrap{max-height:22rem;overflow:auto}.s3-table th,.s3-table td{padding:.6rem;border-bottom:1px solid #344553}
.s3-review{border-top:1px solid #344553;margin-top:1rem;padding-top:.6rem}.s3-picker li{overflow-wrap:anywhere}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.reference-row{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.context-grid,.form-grid,.reference-row{grid-template-columns:1fr}.card{padding:.85rem}.stats button{font-size:.8rem}}
</style>
