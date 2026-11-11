<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import {
  createIncident, createIncidentsFromImport, getBuckets, importIncidentS3Files,
  listIncidentImportS3, listIncidentImportSources, previewIncidentImportSource,
  resolveIncidentImportS3, uploadIncidentImportFiles,
} from "../services/api";
import type { IncidentImportPreview, InvestigationRecord } from "../types/api";
import type { IncidentImportFile, IncidentImportListing, IncidentImportSource } from "../services/api";

const router=useRouter();
const incidentId=ref(""), summary=ref(""), start=ref(""), end=ref(""), severity=ref(""), service=ref(""), services=ref("");
const busy=ref(false), error=ref(""), notice=ref("");
const mode=ref<"manual"|"import">("manual");
const sourceMode=ref<"s3"|"local">("s3"),recordPath=ref(""),localFiles=ref<File[]>([]);
const buckets=ref<string[]>([]),bucket=ref(""),browseUri=ref(""),filesAtLocation=ref<IncidentImportFile[]>([]),folders=ref<{uri:string}[]>([]),nextToken=ref<string|null>(null),selectedUris=ref<string[]>([]),reviewedFiles=ref<IncidentImportFile[]>([]);
const importedSources=ref<IncidentImportSource[]>([]),selectedSourceId=ref(""),preview=ref<IncidentImportPreview|null>(null),previewOffset=ref(0),selectedRecords=ref<number[]>([]),recordIssues=ref<Record<number,string[]>>({}),createdIncidentIds=ref<string[]>([]);
const selectedRows=computed(()=>selectedRecords.value);
const selectedImportSource=computed(()=>importedSources.value.find(source=>source.id===selectedSourceId.value)||null);
const invalidSelected=computed(()=>selectedRows.value.filter(number=>!recordIssues.value[number]||recordIssues.value[number].length>0));
const canCreate=computed(()=>selectedRows.value.length>0&&invalidSelected.value.length===0);
const parentUri=computed(()=>{const root="s3://"+bucket.value,clean=browseUri.value.replace(/\/$/,"");if(clean===root)return root+"/";const index=clean.lastIndexOf("/");return index>=root.length?clean.slice(0,index+1):root+"/";});

async function submit(){
  if(busy.value)return;
  error.value="";notice.value="";
  if(!incidentId.value.trim()||!start.value||!end.value){error.value="Incident ID, start time, and end time are required.";return;}
  if(Date.parse(end.value)<Date.parse(start.value)){error.value="Incident end must be at or after its start.";return;}
  busy.value=true;
  try{
    const item=await createIncident({incident_id:incidentId.value.trim(),summary:summary.value.trim(),start_time:new Date(start.value).toISOString(),end_time:new Date(end.value).toISOString(),severity:severity.value.trim(),primary_service:service.value.trim(),services:services.value.split(",").map(v=>v.trim()).filter(Boolean)});
    await router.push("/rca/incidents/"+item.id);
  }catch(e){error.value=String(e);}finally{busy.value=false;}
}
function clearPreview(){preview.value=null;previewOffset.value=0;selectedRecords.value=[];recordIssues.value={};createdIncidentIds.value=[];}
async function loadImportSources(preferred=""){
  const rows=await listIncidentImportSources();importedSources.value=rows;
  const choice=preferred||selectedSourceId.value;
  selectedSourceId.value=rows.some(row=>row.id===choice)?choice:(rows.find(row=>row.status==="READY")?.id||"");
  if(selectedSourceId.value){
    preview.value=await previewIncidentImportSource(selectedSourceId.value,0);
    previewOffset.value=0;selectedRecords.value=[];recordIssues.value={};
    for(const row of preview.value.records)recordIssues.value[row.record_number]=row.missing_incident_fields;
  }else clearPreview();
}
async function changeImportedSource(){
  selectedRecords.value=[];recordIssues.value={};createdIncidentIds.value=[];
  if(!selectedSourceId.value){clearPreview();return;}
  await openSource(selectedSourceId.value);
}
async function openSource(id:string,offset=0){
  if(!id||busy.value)return;
  const sameSource=selectedSourceId.value===id;
  busy.value=true;error.value="";notice.value="";
  try{preview.value=await previewIncidentImportSource(id,offset);selectedSourceId.value=id;previewOffset.value=offset;if(!sameSource){selectedRecords.value=[];recordIssues.value={};createdIncidentIds.value=[];}for(const row of preview.value.records)recordIssues.value[row.record_number]=row.missing_incident_fields;}
  catch(e){error.value=String(e);preview.value=null;}
  finally{busy.value=false;}
}
async function refreshImported(){busy.value=true;error.value="";try{await loadImportSources();}catch(e){error.value=String(e);}finally{busy.value=false;}}
function onFiles(event:Event){localFiles.value=Array.from((event.target as HTMLInputElement).files||[]).filter(file=>/\.(csv|parquet|json|jsonl|log|txt)$/i.test(file.name));}
async function showImportResults(result:{sources:{id:string;status:string;name:string;error?:string}[]}){
  const failed=result.sources.filter(source=>source.status==="FAILED").length;
  notice.value=`Import finished: ${result.sources.length-failed} ready/existing, ${failed} failed.`;
  await loadImportSources(result.sources.find(source=>source.status==="READY"||source.status==="DUPLICATE")?.id||"");
}
async function importLocal(){
  if(!localFiles.value.length||busy.value)return;
  busy.value=true;error.value="";notice.value="";
  try{const result=await uploadIncidentImportFiles(localFiles.value,recordPath.value);await showImportResults(result);localFiles.value=[];}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
async function loadBuckets(){busy.value=true;error.value="";try{buckets.value=await getBuckets();if(!buckets.value.length)notice.value="No internal S3 buckets are available.";}
  catch(e){error.value=String(e);}finally{busy.value=false;}}
async function browse(uri:string,more=false){
  if(!uri||busy.value)return;
  busy.value=true;error.value="";notice.value="";
  if(!more){browseUri.value=uri;filesAtLocation.value=[];folders.value=[];nextToken.value=null;selectedUris.value=[];reviewedFiles.value=[];}
  try{const result:IncidentImportListing=await listIncidentImportS3(uri,more?nextToken.value||"":"",true);filesAtLocation.value=more?[...filesAtLocation.value,...result.files]:result.files;folders.value=more?[...folders.value,...(result.folders||[])]:result.folders||[];nextToken.value=result.next_token;}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
function chooseBucket(){selectedUris.value=[];reviewedFiles.value=[];if(bucket.value)void browse("s3://"+bucket.value+"/");else{browseUri.value="";filesAtLocation.value=[];folders.value=[];}}
async function reviewS3(){
  if(!selectedUris.value.length||busy.value)return;
  busy.value=true;error.value="";notice.value="";
  try{const result=await resolveIncidentImportS3([...selectedUris.value]);reviewedFiles.value=result.files;notice.value=`${result.files.length} supported file(s), ${result.total_bytes.toLocaleString()} bytes. Review before importing.`;}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
async function importS3(){
  if(!reviewedFiles.value.length||busy.value)return;
  busy.value=true;error.value="";notice.value="";
  try{const result=await importIncidentS3Files(reviewedFiles.value,recordPath.value);await showImportResults(result);selectedUris.value=[];reviewedFiles.value=[];}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
function toggleRecord(row:InvestigationRecord,event:Event){const checked=(event.target as HTMLInputElement).checked;selectedRecords.value=checked?[...new Set([...selectedRecords.value,row.record_number])]:selectedRecords.value.filter(number=>number!==row.record_number);}
async function createSelectedIncidents(){
  if(!preview.value||!canCreate.value||busy.value)return;
  busy.value=true;error.value="";notice.value="";createdIncidentIds.value=[];
  try{const result=await createIncidentsFromImport(preview.value.id,selectedRows.value);createdIncidentIds.value=result.incident_ids;notice.value=`Created or linked ${result.created_or_linked} incident(s). Each incident retains a link to its Stage C source record.`;selectedRecords.value=[];await loadImportSources(preview.value.id);}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
watch(selectedUris,()=>{reviewedFiles.value=[];});
watch(mode,async value=>{if(value!=="import")return;error.value="";notice.value="";try{await loadImportSources();if(sourceMode.value==="s3"&&!buckets.value.length)await loadBuckets();}catch(e){error.value=String(e);}});
watch(sourceMode,async value=>{if(mode.value==="import"&&value==="s3"&&!buckets.value.length)await loadBuckets();});
</script>
<template>
  <section class="view incident-create">
    <header class="view-header"><div><h1>Create Incident</h1><p>Create an incident manually or import Stage C incident records from a file.</p></div><RouterLink class="btn" to="/rca/incidents">List Incidents</RouterLink></header>
    <nav class="mode-tabs" aria-label="Incident creation method"><button :class="{active:mode==='manual'}" @click="mode='manual'">Manual Entry</button><button :class="{active:mode==='import'}" @click="mode='import'">Import Incident File</button></nav>
    <p v-if="error" class="error" role="alert">{{error}}</p><p v-if="notice" class="notice" role="status">{{notice}}</p>

    <form v-if="mode==='manual'" class="card" @submit.prevent="submit">
      <div class="grid">
        <label>Incident ID <span class="required">Required</span><input v-model="incidentId" maxlength="300" required placeholder="INC-20261109-001" /></label>
        <label>Summary<input v-model="summary" maxlength="10000" placeholder="Briefly describe the incident" /></label>
        <label>Start time <span class="required">Required</span><input v-model="start" type="datetime-local" required /></label>
        <label>End time <span class="required">Required</span><input v-model="end" type="datetime-local" required /></label>
        <label>Maximum severity<input v-model="severity" maxlength="100" placeholder="critical, high, medium" /></label>
        <label>Primary service<input v-model="service" maxlength="300" placeholder="payment-service" /></label>
        <label class="wide">Related services<input v-model="services" placeholder="payment-service, order-service" /><small>Separate service names with commas.</small></label>
      </div>
      <div class="actions"><button class="btn primary" :disabled="busy">{{busy?'Creating...':'Create Incident'}}</button><RouterLink class="btn" to="/rca/incidents">Cancel</RouterLink></div>
    </form>

    <section v-else class="card stack">
      <div><h2>Import Incident</h2><p>Import a Stage C file, preview its incident records, select the rows to create, and review any required fields before saving.</p></div>
      <label>Source<select v-model="sourceMode"><option value="s3">Internal S3</option><option value="local">Local files / Folder</option></select></label>
      <p><strong>Evidence type:</strong> Stage C · Incidents</p>
      <details><summary>Advanced import options</summary><label>Nested JSON record path (JSON Pointer)<input v-model="recordPath" placeholder="/incidents" /><small>Leave blank for flat files. Use this only when records are inside a nested JSON object.</small></label></details>

      <section v-if="sourceMode==='local'" class="subcard stack">
        <h3>Local files / Folder</h3>
        <label>Choose incident files<input type="file" multiple accept=".csv,.parquet,.json,.jsonl,.log,.txt" @change="onFiles" /></label>
        <label>Choose a folder<input type="file" multiple webkitdirectory accept=".csv,.parquet,.json,.jsonl,.log,.txt" @change="onFiles" /></label>
        <p>{{localFiles.length}} supported file(s) selected. Supported formats: CSV, Parquet, JSON, and JSONL; LOG/TXT can be imported but only records containing incident fields can create incidents.</p>
        <button class="btn primary" :disabled="busy||!localFiles.length" @click="importLocal">{{busy?'Importing...':'Import Files'}}</button>
      </section>

      <section v-else class="subcard stack">
        <h3>Internal S3</h3>
        <div class="bucket-row"><label>Input bucket<select v-model="bucket" @change="chooseBucket"><option value="">Select an input bucket</option><option v-for="name in buckets" :key="name" :value="name">{{name}}</option></select></label><button :disabled="busy" @click="loadBuckets">Refresh Buckets</button></div>
        <p v-if="!browseUri" class="muted">Choose an input bucket to browse for an incident file or folder.</p>
        <template v-else>
          <p class="location">Location: <code>{{browseUri}}</code></p>
          <div class="actions"><button :disabled="busy" @click="browse(parentUri)">Up one folder</button><button :disabled="busy" @click="browse(browseUri)">Refresh</button></div>
          <div class="table-wrap"><table><thead><tr><th>Select</th><th>Name</th><th>Type</th><th>Size</th></tr></thead><tbody>
            <tr v-for="folder in folders" :key="folder.uri"><td><input v-model="selectedUris" type="checkbox" :value="folder.uri" /></td><td><button class="link-button" @click="browse(folder.uri)">{{folder.uri.split('/').filter(Boolean).at(-1)}}</button></td><td>Folder</td><td>—</td></tr>
            <tr v-for="file in filesAtLocation" :key="file.uri"><td><input v-model="selectedUris" type="checkbox" :value="file.uri" /></td><td>{{file.uri.split('/').at(-1)}}</td><td>{{file.uri.split('.').at(-1)?.toUpperCase()}}</td><td>{{file.size.toLocaleString()}} bytes</td></tr>
          </tbody></table></div>
          <button v-if="nextToken" :disabled="busy" @click="browse(browseUri,true)">Load More</button>
          <div class="actions"><button :disabled="busy||!selectedUris.length" @click="reviewS3">Review Selected Files</button></div>
          <div v-if="reviewedFiles.length" class="review-box"><h4>Files to import</h4><ul><li v-for="file in reviewedFiles" :key="file.uri">{{file.uri}}</li></ul><button class="btn primary" :disabled="busy" @click="importS3">Import Files</button></div>
        </template>
      </section>

      <section class="subcard stack">
        <div class="section-heading"><div><h3>Imported Files</h3><p>Select an imported Stage C file to preview its rows and create incidents.</p></div><button :disabled="busy" @click="refreshImported">Refresh Files</button></div>
        <label>Choose an imported file<select v-model="selectedSourceId" @change="changeImportedSource"><option value="">Select an imported file</option><option v-for="source in importedSources" :key="source.id" :value="source.id">{{source.name}} · {{source.status}} · {{source.record_count}} records</option></select></label>
        <p v-if="!importedSources.length" class="muted">No imported Stage C files yet.</p>
        <p v-else-if="selectedSourceId&&preview?.status!=='READY'" class="error">This import is not ready for preview. {{selectedImportSource?.error||'Check its import status or select a ready file.'}}</p>
        <template v-if="preview?.status==='READY'">
          <p>Source: <code>{{preview.original_uri||preview.name}}</code> · {{preview.record_count}} records</p>
          <div class="table-wrap"><table><thead><tr><th>Select</th><th>Record</th><th>Incident fields</th><th>Record data</th></tr></thead><tbody>
            <tr v-for="row in preview.records" :key="row.record_number"><td><input type="checkbox" :checked="selectedRecords.includes(row.record_number)" @change="toggleRecord(row,$event)" /></td><td>{{row.record_number}}</td><td><span v-if="row.missing_incident_fields.length" class="error">Missing: {{row.missing_incident_fields.join(', ')}}</span><span v-else class="ready">Ready to create</span></td><td><pre>{{JSON.stringify(row.original_record,null,2)}}</pre></td></tr>
          </tbody></table></div>
          <div class="actions"><button :disabled="busy||previewOffset===0" @click="openSource(preview.id,Math.max(0,previewOffset-100))">Previous</button><button :disabled="busy||preview.next_offset==null" @click="openSource(preview.id,preview.next_offset||0)">Next</button><button class="btn primary" :disabled="busy||!canCreate" @click="createSelectedIncidents">Create {{selectedRows.length}} Incident(s)</button></div>
          <p v-if="invalidSelected.length" class="error">Deselect selected rows missing required fields before creating incidents.</p>
        </template>
        <ul v-if="createdIncidentIds.length" class="created-list"><li v-for="id in createdIncidentIds" :key="id"><RouterLink :to="'/rca/incidents/'+id">Open incident {{id}}</RouterLink></li></ul>
      </section>
    </section>
  </section>
</template>
<style scoped>
.incident-create{max-width:1200px}
.view-header,.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.card,.subcard{padding:1.2rem;background:#111c26;border:1px solid #30404e;border-radius:.65rem}
.subcard{background:#0e1821}
.stack{display:grid;gap:.8rem;margin-top:.8rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.grid label{display:grid;gap:.4rem}
.wide{grid-column:1/-1}
.mode-tabs,.actions,.bucket-row{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
.mode-tabs{margin:.8rem 0 1rem;gap:.3rem;overflow-x:auto;border-bottom:1px solid #344553;padding-bottom:.5rem}
.mode-tabs button{white-space:nowrap;border-color:transparent;background:transparent}
.mode-tabs button.active,.mode-tabs button:hover:not(:disabled){background:#203d3b;color:#8fe5ce;border-color:#47766b}
.bucket-row label{flex:1;min-width:15rem}
.bucket-row button{align-self:end}
.location,.muted,small{color:#a7bbc8}
.location{overflow-wrap:anywhere}
.required{color:#a7bbc8;font-size:.85rem}
.actions{margin-top:.5rem}
.table-wrap{max-height:28rem;overflow:auto}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:.6rem;border-bottom:1px solid #30404e;vertical-align:top}
pre{max-width:520px;max-height:12rem;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
label{display:grid;gap:.4rem}
input:not([type=checkbox]),select{width:100%;min-width:0;padding:.65rem;color:#e8edf2;background:#17232d;border:1px solid #405665;border-radius:.4rem;font:inherit}
input[type=checkbox]{width:1rem;height:1rem;accent-color:#6bd7bb}
button{padding:.55rem .75rem;color:#e8edf2;background:#243442;border:1px solid #405665;border-radius:.45rem;cursor:pointer;font:inherit}
button:hover:not(:disabled){border-color:#6bd7bb;background:#2b414c}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary,.btn.primary{background:#6bd7bb;border-color:#6bd7bb;color:#08131b;font-weight:700}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}
.btn{display:inline-block;padding:.55rem .75rem;color:#e8edf2;background:#243442;border:1px solid #405665;border-radius:.45rem;text-decoration:none;font:inherit;cursor:pointer}
.btn:hover{border-color:#6bd7bb;background:#2b414c}
.btn.primary:hover{background:#6bd7bb;border-color:#6bd7bb;color:#08131b}
button.link-button{padding:0;background:none;border:0;border-radius:0;color:#8fe5ce;text-decoration:underline;cursor:pointer}
button.link-button:hover:not(:disabled){background:transparent;border-color:transparent;color:#b8d9d3}
.review-box{border-top:1px solid #30404e;padding-top:.7rem}
.created-list{margin:0;padding-left:1.2rem}
.error{color:#ffaaa5}
.ready,.notice{color:#8fe5ce}
@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
