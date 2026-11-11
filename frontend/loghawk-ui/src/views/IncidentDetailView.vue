<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { addIncidentCollectorEvidence, deleteIncident, getBuckets, getIncident, incidentAction, importIncidentS3, listIncidentS3, previewIncidentSource, resolveIncidentS3, uploadIncidentFiles } from "../services/api";
import type { Incident, InvestigationPreview } from "../types/api";
import type { IncidentS3File } from "../services/api";
import ConfirmDialog from "../components/ConfirmDialog.vue";
const route=useRoute(),router=useRouter(),id=computed(()=>String(route.params.id));
const item=ref<Incident|null>(null),busy=ref(false),error=ref(""),notice=ref("");
const deleteConfirmationOpen=ref(false);
const deleteConfirmationMessage=computed(()=>item.value?`Permanently delete incident “${item.value.source_incident_id||item.value.id}”? This cannot be undone. Its investigation and incident-evidence links will be removed. Imported evidence and saved RCA snapshots will be retained.`:"");
const addOpen=ref(false),sourceMode=ref<"s3"|"local"|"collector"|"existing">("s3"),evidenceType=ref("other"),recordPath=ref(""),files=ref<File[]>([]),collectorCallId=ref(""),existingSourceId=ref("");
const buckets=ref<string[]>([]),bucket=ref(""),browseUri=ref(""),filesAtLocation=ref<IncidentS3File[]>([]),folders=ref<{uri:string}[]>([]),s3Selected=ref<string[]>([]),reviewed=ref<IncidentS3File[]>([]);
const preview=ref<InvestigationPreview|null>(null),previewSource=ref(""),previewOffset=ref(0),selected=ref<number[]>([]),allRecords=ref(false);
const sourceLabel=(s:any)=>s.name||s.original_uri||s.id;
async function removeIncident(){deleteConfirmationOpen.value=false;busy.value=true;error.value="";try{await deleteIncident(id.value);await router.push("/rca/incidents");}catch(e){error.value=String(e);}finally{busy.value=false;}}
const selectedCount=computed(()=>allRecords.value?(preview.value?.record_count||0):selected.value.length);
async function load(){busy.value=true;error.value="";try{item.value=await getIncident(id.value);}catch(e){error.value=String(e);}finally{busy.value=false;}}
function setFiles(event:Event){files.value=Array.from((event.target as HTMLInputElement).files||[]);}
async function loadBuckets(){try{buckets.value=await getBuckets();}catch(e){error.value=String(e);}}
function resultNotice(result:{sources:{status:string}[]}){return "Import finished: "+result.sources.filter(s=>s.status==="READY"||s.status==="DUPLICATE").length+" ready/existing, "+result.sources.filter(s=>s.status==="FAILED").length+" failed.";}
async function importLocal(){if(!files.value.length)return;busy.value=true;error.value="";notice.value="";try{const result=await uploadIncidentFiles(id.value,files.value,evidenceType.value,recordPath.value);notice.value=resultNotice(result);files.value=[];await load();}catch(e){error.value=String(e);}finally{busy.value=false;}}
async function browse(uri:string){if(!uri)return;busy.value=true;error.value="";browseUri.value=uri;s3Selected.value=[];reviewed.value=[];try{const result=await listIncidentS3(id.value,uri);filesAtLocation.value=result.files;folders.value=result.folders||[];}catch(e){error.value=String(e);}finally{busy.value=false;}}
function chooseBucket(){if(bucket.value)void browse("s3://"+bucket.value+"/");}
const parentUri=computed(()=>{const root="s3://"+bucket.value,clean=browseUri.value.replace(/\/$/,"");if(clean===root)return root+"/";const index=clean.lastIndexOf("/");return index>=root.length?clean.slice(0,index+1):root+"/";});
const breadcrumbs=computed(()=>{const root="s3://"+bucket.value+"/",tail=browseUri.value.startsWith(root)?browseUri.value.slice(root.length):"";let uri=root;return tail.split("/").filter(Boolean).map(name=>{uri+=name+"/";return{name,uri};});});
async function reviewS3(){if(!s3Selected.value.length)return;busy.value=true;error.value="";try{const result=await resolveIncidentS3(id.value,[...s3Selected.value]);reviewed.value=result.files;}catch(e){error.value=String(e);}finally{busy.value=false;}}
async function importS3(){if(!reviewed.value.length)return;busy.value=true;error.value="";try{const result=await importIncidentS3(id.value,reviewed.value,evidenceType.value,recordPath.value);notice.value=resultNotice(result);reviewed.value=[];s3Selected.value=[];await load();}catch(e){error.value=String(e);}finally{busy.value=false;}}
async function attachSavedCollector(){const callId=collectorCallId.value.trim();if(!callId)return;busy.value=true;error.value="";notice.value="";try{item.value=await addIncidentCollectorEvidence(id.value,callId);collectorCallId.value="";notice.value="Saved collector evidence attached. Preview it in the Evidence list.";}catch(e){error.value=String(e);}finally{busy.value=false;}}
async function attachExistingSource(){const sourceId=existingSourceId.value.trim();if(!sourceId)return;busy.value=true;error.value="";notice.value="";try{item.value=await incidentAction<Incident>(id.value,"evidence",{source_id:sourceId,records:null});existingSourceId.value="";notice.value="Existing source attached to the incident.";}catch(e){error.value=String(e);}finally{busy.value=false;}}
function restoreIncidentSelection(sourceId:string){
  const source=item.value?.sources.find(s=>s.id===sourceId);
  selected.value=Array.isArray(source?.attachment_selection)
    ?source.attachment_selection.filter((number):number is number=>typeof number==="number")
    :[];
  allRecords.value=!!source&&source.attachment_selection===null;
}
async function openPreview(sourceId:string,offset=0){
  const switchingSource=previewSource.value!==sourceId;
  busy.value=true;error.value="";
  try{
    preview.value=await previewIncidentSource(id.value,sourceId,offset);
    previewSource.value=sourceId;previewOffset.value=offset;
    if(switchingSource)restoreIncidentSelection(sourceId);
  }catch(e){error.value=String(e);}finally{busy.value=false;}
}
function toggleRecord(number:number,event:Event){const checked=(event.target as HTMLInputElement).checked;selected.value=checked?[...new Set([...selected.value,number])]:selected.value.filter(n=>n!==number);allRecords.value=false;}
async function attach(){if(!previewSource.value||(!allRecords.value&&!selected.value.length))return;busy.value=true;error.value="";try{item.value=await incidentAction(id.value,"evidence",{source_id:previewSource.value,records:allRecords.value?null:selected.value});notice.value="Evidence attached to the incident.";await load();}catch(e){error.value=String(e);}finally{busy.value=false;}}
async function unlink(sourceId:string){if(!window.confirm("Remove this evidence attachment from the incident? The imported source and saved RCA snapshots are retained."))return;busy.value=true;error.value="";try{await incidentAction(id.value,"evidence/unlink",{source_id:sourceId});notice.value="Evidence detached from this incident; the imported source is retained.";if(previewSource.value===sourceId)preview.value=null;await load();}catch(e){error.value=String(e);}finally{busy.value=false;}}
function pretty(value:unknown){return JSON.stringify(value,null,2)}
function display(value:unknown){if(value==null||value==="")return "-";return Array.isArray(value)?value.join(", "):String(value)}
function formatDate(value:string){const d=new Date(value);return Number.isNaN(d.getTime())?value:d.toLocaleString()}
watch(sourceMode,async mode=>{if(mode==="s3"&&!buckets.value.length)await loadBuckets();});
watch(addOpen,async open=>{if(open&&sourceMode.value==="s3"&&!buckets.value.length)await loadBuckets();});
onMounted(load);
</script>
<template>
<section v-if="item" class="view incident-detail">
  <RouterLink to="/rca/incidents">Back to List Incidents</RouterLink>
  <header class="view-header"><div><p class="eyebrow">Incident - {{item.userid}} - Created {{formatDate(item.created_at)}}</p><h1>{{item.primary_details.summary||item.source_incident_id}}</h1><p>{{item.source_incident_id}}</p></div><div class="actions"><button v-if="item.can_modify" class="btn primary" :disabled="busy" @click="addOpen=!addOpen">{{addOpen?'Cancel Add Evidence':'Add Evidence'}}</button><button v-if="item.can_delete" class="btn danger" :disabled="busy" @click="deleteConfirmationOpen=true">Permanently Delete</button></div></header>
  <p v-if="error" class="error" role="alert">{{error}}</p><p v-if="notice" class="notice" role="status">{{notice}}</p>
  <section class="card"><h2>Incident details</h2><dl class="details"><div><dt>Start</dt><dd>{{display(item.primary_details.start_time)}}</dd></div><div><dt>End</dt><dd>{{display(item.primary_details.end_time)}}</dd></div><div><dt>Maximum severity</dt><dd>{{display(item.primary_details.max_severity)}}</dd></div><div><dt>Primary service</dt><dd>{{display(item.primary_details.primary_service)}}</dd></div><div><dt>Related services</dt><dd>{{display(item.primary_details.services)}}</dd></div><div><dt>Incident record ID</dt><dd>{{item.id}}</dd></div></dl><details><summary>Full incident details</summary><pre>{{pretty(item.primary_details)}}</pre></details></section>
  <section v-if="addOpen&&item.can_modify" class="card stack"><h2>Add Evidence</h2><p>Import a source, preview its records, then attach selected records or all records to this incident.</p>
    <label>Source<select v-model="sourceMode"><option value="s3">Internal S3</option><option value="local">Local files / Folder</option><option value="collector">Saved collector evidence</option><option value="existing">Existing source</option></select></label>
    <template v-if="sourceMode==='local'||sourceMode==='s3'">
      <label>Evidence type<select v-model="evidenceType"><option value="stage-a">Stage A · Features</option><option value="stage-b">Stage B · Anomalies</option><option value="stage-c">Stage C · Incidents</option><option value="raw-logs">Raw logs</option><option value="other">Other</option></select></label>
      <details><summary>Advanced import options</summary><label>Nested JSON record path (JSON Pointer)<input v-model="recordPath" placeholder="/incidents" /></label><p class="muted">Leave blank for flat files. Use this only when records are inside a nested JSON object.</p></details>
    </template>
    <section v-if="sourceMode==='local'" class="stack">
      <div class="local-file-grid">
        <label>Choose files<input type="file" multiple accept=".csv,.parquet,.json,.jsonl,.log,.txt" @change="setFiles" /></label>
        <label>Choose folder<input type="file" multiple webkitdirectory accept=".csv,.parquet,.json,.jsonl,.log,.txt" @change="setFiles" /></label>
      </div>
      <p>{{files.length}} file(s) selected.</p>
      <button class="btn primary" :disabled="busy||!files.length" @click="importLocal">Import Files</button>
    </section>
    <section v-if="sourceMode==='s3'" class="s3-picker stack" aria-labelledby="s3-title">
      <div class="section-heading"><div><h3 id="s3-title">Internal S3</h3><p class="muted">Browse folders and choose files to import as incident evidence.</p></div></div>
      <div class="s3-toolbar">
        <label>Input bucket<select v-model="bucket" @change="chooseBucket"><option value="">Select an input bucket</option><option v-for="name in buckets" :key="name" :value="name">{{name}}</option></select></label>
        <button type="button" :disabled="busy" @click="loadBuckets">Refresh Buckets</button>
      </div>
      <p v-if="!bucket" class="muted">Choose an input bucket.</p>
      <div v-if="bucket&&browseUri" class="s3-browser stack">
        <nav class="s3-breadcrumbs" aria-label="S3 folder location">
          <button type="button" @click="browse('s3://'+bucket+'/')">{{bucket}}</button>
          <template v-for="crumb in breadcrumbs" :key="crumb.uri"><span aria-hidden="true">/</span><button type="button" @click="browse(crumb.uri)">{{crumb.name}}</button></template>
        </nav>
        <p class="s3-location"><code>{{browseUri}}</code></p>
        <div class="actions s3-toolbar-actions"><button type="button" :disabled="busy" @click="browse(parentUri)">Up one folder</button><button type="button" :disabled="busy" @click="browse(browseUri)">Refresh Folder</button></div>
        <div class="table-wrap s3-table-wrap"><table class="s3-table"><thead><tr><th>Select</th><th>Name</th><th>Type</th><th>Size</th></tr></thead><tbody>
          <tr v-for="folder in folders" :key="folder.uri"><td><input v-model="s3Selected" type="checkbox" :value="folder.uri" :aria-label="'Select folder '+folder.uri" /></td><td><button type="button" class="link-button" @click="browse(folder.uri)">{{folder.uri.split('/').filter(Boolean).at(-1)}}</button></td><td>Folder</td><td>-</td></tr>
          <tr v-for="file in filesAtLocation" :key="file.uri"><td><input v-model="s3Selected" type="checkbox" :value="file.uri" :aria-label="'Select file '+file.uri" /></td><td>{{file.uri.split('/').at(-1)}}</td><td>{{file.uri.split('.').at(-1)?.toUpperCase()}}</td><td>{{file.size.toLocaleString()}} bytes</td></tr>
          <tr v-if="!folders.length&&!filesAtLocation.length"><td colspan="4" class="muted">No supported files or folders at this location.</td></tr>
        </tbody></table></div>
        <p class="muted">Supported files: CSV, Parquet, JSON, JSONL, LOG and TXT. Selecting a folder includes supported files in its subfolders.</p>
        <div class="actions"><button type="button" :disabled="busy||!s3Selected.length" @click="reviewS3">Review Selected Files</button><button type="button" :disabled="busy||(!s3Selected.length&&!reviewed.length)" @click="s3Selected=[];reviewed=[]">Clear Selection</button></div>
        <div v-if="reviewed.length" class="s3-review">
          <h4>Review the files below before importing.</h4>
          <ul class="s3-file-list"><li v-for="file in reviewed" :key="file.uri">{{file.uri}}</li></ul>
          <div class="actions"><button type="button" class="primary" :disabled="busy" @click="importS3">Import Files</button></div>
        </div>
      </div>
    </section>
    <section v-if="sourceMode==='collector'" class="stack"><label>Collector tool-call event ID<input v-model="collectorCallId" placeholder="Paste an event ID from RCA History" /></label><p class="muted">References saved collector output from your own RCA history; this does not run the collector again.</p><button class="btn primary" :disabled="busy||!collectorCallId.trim()" @click="attachSavedCollector">Attach Saved Collector Evidence</button></section>
    <section v-if="sourceMode==='existing'" class="stack"><label>Existing source ID<input v-model="existingSourceId" placeholder="Paste an owned source ID" /></label><p class="muted">Attaches all records from an existing evidence source that you own.</p><button class="btn primary" :disabled="busy||!existingSourceId.trim()" @click="attachExistingSource">Attach Existing Source</button></section>
  </section>
  <section class="card stack">
    <div class="section-heading"><div><h2>Evidence</h2><p class="muted">Evidence attached to this incident is also available in investigations linked to it.</p></div></div>
    <p v-if="!item.sources.length" class="muted">No evidence attached yet.</p>
    <div v-else class="table-wrap"><table class="source-table"><thead><tr><th>File / source</th><th>Evidence type</th><th>Import status</th><th>Records attached</th><th>Action</th></tr></thead><tbody>
      <tr v-for="source in item.sources" :key="source.id">
        <td class="source-name">{{sourceLabel(source)}}</td>
        <td>{{source.evidence_type}}</td>
        <td><span class="status-pill" :class="{'status-failed':source.status==='FAILED'}">{{source.status}}</span><p v-if="source.error" class="error source-error">{{source.error}}</p></td>
        <td>{{source.attachment_selection?.length===0?'Not attached':source.attachment_selection===null?'All records':source.attachment_selection?.length}}</td>
        <td><div class="actions"><button type="button" :disabled="busy||source.status!=='READY'" @click="openPreview(source.id)">Preview / Select</button><button v-if="item.can_modify" type="button" class="danger" :disabled="busy" @click="unlink(source.id)">Remove Attachment</button></div></td>
      </tr>
    </tbody></table></div>
  </section>
  <section v-if="preview" class="card stack preview-panel">
    <div class="section-heading"><div><p class="eyebrow">Source preview</p><h2>{{sourceLabel(preview)}}</h2><p class="muted">{{preview.record_count}} records</p></div><button type="button" @click="preview=null">Close Preview</button></div>
    <label class="inline"><input v-model="allRecords" type="checkbox" @change="selected=[]" />Attach all {{preview.record_count}} records</label>
    <div class="table-wrap record-table"><table><thead><tr><th>Select</th><th>Record</th><th>Values</th></tr></thead><tbody><tr v-for="row in preview.records" :key="row.record_number"><td><input type="checkbox" :checked="selected.includes(row.record_number)" :disabled="allRecords" @change="toggleRecord(row.record_number,$event)" /></td><td>{{row.record_number}}</td><td><pre>{{pretty(row.original_record)}}</pre></td></tr></tbody></table></div>
    <div class="record-footer"><span>{{selectedCount}} record(s) selected</span><div class="actions"><button type="button" :disabled="busy||previewOffset===0" @click="openPreview(previewSource,Math.max(0,previewOffset-100))">Previous</button><button type="button" :disabled="busy||preview.next_offset==null" @click="openPreview(previewSource,preview.next_offset||0)">Next</button><button type="button" class="primary" :disabled="busy||(!allRecords&&!selected.length)" @click="attach">Attach {{selectedCount}} record(s)</button></div></div>
  </section>
  <section class="card stack"><h2>Linked investigations</h2><p v-if="!item.investigations.length" class="muted">This incident is not linked to an investigation.</p><ul v-else><li v-for="inv in item.investigations" :key="inv.id"><RouterLink :to="'/rca/investigations/'+inv.id">{{inv.problem}}</RouterLink> - {{inv.status}} - {{formatDate(inv.created_at)}}</li></ul></section>
  <ConfirmDialog v-if="deleteConfirmationOpen" title="Permanently delete incident?" :message="deleteConfirmationMessage" confirm-label="Yes, permanently delete" cancel-label="No, keep incident" @confirm="removeIncident" @cancel="deleteConfirmationOpen=false" />
</section>
<p v-else-if="busy" class="muted">Loading incident...</p><p v-else-if="error" class="error" role="alert">{{error}}</p>
</template>
<style scoped>
.incident-detail{max-width:1400px}
.view-header,.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}
.view-header>div:first-child{flex:1;min-width:200px}
.stack{display:grid;gap:1rem;min-width:0}
.local-file-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
h1{overflow-wrap:anywhere;white-space:pre-wrap}h2,h3,h4{margin:.3rem 0 .7rem}
.eyebrow{font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;color:#aebbc7;margin:.2rem 0 .5rem}
.section-heading p{margin:.3rem 0}
.incident-detail .card{padding:1.25rem;background:#131f28;border:1px solid #2d404e;border-radius:.75rem}
.details{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.8rem}.details dt{color:#aebbc7}.details dd{margin:.2rem 0;overflow-wrap:anywhere}
.actions{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.inline{display:flex;gap:.5rem;align-items:center}
.incident-detail button{padding:.55rem .75rem;color:#e8edf2;background:#243442;border:1px solid #405665;border-radius:.45rem;cursor:pointer;font:inherit}
.incident-detail button:hover:not(:disabled){border-color:#6bd7bb;background:#2b414c}.incident-detail button:disabled{opacity:.45;cursor:not-allowed}
.incident-detail button.primary{background:#6bd7bb;border-color:#6bd7bb;color:#08131b;font-weight:700}
.incident-detail button.danger{background:#34242a;border-color:#89515c;color:#ffc4cc}
.incident-detail button:focus-visible,.incident-detail input:focus-visible,.incident-detail textarea:focus-visible,.incident-detail select:focus-visible,.incident-detail summary:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}
.incident-detail label{display:grid;gap:.4rem;margin:.65rem 0;color:#bbc9d2;font-size:.9rem}
.incident-detail input:not([type=checkbox]),.incident-detail select,.incident-detail textarea{width:100%;min-width:0;padding:.65rem;color:#e8edf2;background:#17232d;border:1px solid #405665;border-radius:.4rem;font:inherit}
.incident-detail input[type=checkbox]{width:1rem;height:1rem;accent-color:#6bd7bb}.incident-detail label.inline{display:flex;margin:.4rem 0}
.incident-detail details{margin:.8rem 0}.incident-detail summary{cursor:pointer;color:#b8d9d3;padding:.25rem 0}
.incident-detail pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:22rem;overflow:auto;background:#0d151c;padding:.8rem;border-radius:.4rem;margin:0}
.incident-detail code{overflow-wrap:anywhere}.muted,.eyebrow{color:#aebbc7}.notice{padding:.85rem 1rem;background:#172d30;border:1px solid #355c59;border-radius:.5rem;color:#d6fff3;overflow-wrap:anywhere}.error{color:#ffc4cc}.source-error{margin:.45rem 0 0;overflow-wrap:anywhere}
.link-button{padding:0!important;background:none!important;border:0!important;color:#8fe5ce!important;text-decoration:underline}
.s3-toolbar{display:flex;align-items:end;gap:.6rem;flex-wrap:wrap}.s3-toolbar label{flex:1;min-width:14rem;margin:0}
.s3-breadcrumbs{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;margin:.15rem 0;color:#aebbc7}.s3-location{margin:0;overflow-wrap:anywhere;font-family:monospace;color:#dce8ed}
.s3-table-wrap{max-height:22rem;overflow:auto}.s3-table th,.s3-table td{padding:.65rem;border-bottom:1px solid #344553}.s3-table th,.source-table th,.record-table th{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:#aebbc7}
.s3-table th,.record-table th{position:sticky;top:0;background:#17232d;z-index:1}.s3-table input[type=checkbox]{display:block}
.s3-review{border-top:1px solid #344553;margin-top:.25rem;padding-top:.8rem}.s3-file-list{margin:.2rem 0;padding-left:1.25rem;overflow-wrap:anywhere}.s3-file-list li{margin:.25rem 0}
.source-table td{vertical-align:top;overflow-wrap:anywhere}.source-name{min-width:12rem}.status-pill{display:inline-block;padding:.18rem .55rem;border-radius:999px;background:#263944;color:#d7e7ee;font-size:.78rem}.status-failed{background:#572e38;color:#ffc4cc}
.record-table{max-height:28rem;overflow:auto}.record-table td{vertical-align:top;overflow-wrap:anywhere}.record-table pre{max-width:650px}.record-footer{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;padding-top:.8rem;border-top:1px solid #344553}
@media(max-width:800px){.details{grid-template-columns:1fr 1fr}.record-footer{align-items:flex-start}}
@media(max-width:600px){.details{grid-template-columns:1fr}.s3-toolbar{align-items:stretch}.s3-toolbar label{min-width:100%}.local-file-grid{grid-template-columns:1fr}}
</style>
