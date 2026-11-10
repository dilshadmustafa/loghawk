<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { deleteIncident, listIncidents } from "../services/api";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import type { Incident } from "../types/api";
const q=ref(""),severity=ref(""),service=ref(""),createdFrom=ref(""),createdTo=ref(""),items=ref<Incident[]>([]),offset=ref(0),busy=ref(false),error=ref(""),notice=ref("");
const pendingDelete=ref<Incident|null>(null);
const deleteMessage=computed(()=>pendingDelete.value?`Permanently delete incident “${pendingDelete.value.source_incident_id||pendingDelete.value.id}”? This cannot be undone. Its investigation and incident-evidence links will be removed. Imported evidence and saved RCA snapshots will be retained.`:"");
let timer:ReturnType<typeof setTimeout>|undefined;
async function load(next=0){busy.value=true;error.value="";const query=new URLSearchParams({offset:String(next),limit:"50"});if(q.value.trim())query.set("q",q.value.trim());if(severity.value.trim())query.set("severity",severity.value.trim());if(service.value.trim())query.set("service",service.value.trim());if(createdFrom.value)query.set("created_from",createdFrom.value);if(createdTo.value)query.set("created_to",createdTo.value);try{items.value=await listIncidents(query);offset.value=next;}catch(e){error.value=String(e);}finally{busy.value=false;}}
watch([q,severity,service,createdFrom,createdTo],()=>{if(timer)clearTimeout(timer);timer=setTimeout(()=>void load(0),250);});
onMounted(()=>void load());
function details(item:Incident,key:string){return item.primary_details?.[key]??"-"}
function remove(item:Incident){pendingDelete.value=item;}
function cancelDelete(){pendingDelete.value=null;}
async function confirmDelete(){const item=pendingDelete.value;pendingDelete.value=null;if(!item)return;busy.value=true;error.value="";notice.value="";try{await deleteIncident(item.id);notice.value="Incident permanently deleted. Investigation/evidence links were removed; imported evidence and RCA snapshots were retained.";await load(offset.value);}catch(e){error.value=String(e);}finally{busy.value=false;}}
</script>
<template>
  <section class="view incidents">
    <header class="view-header"><div><h1>List Incidents</h1><p>Search active incidents. Permanently deleting an incident removes its links, while imported evidence and saved RCA history are retained.</p></div><div class="actions"><RouterLink class="btn" to="/rca/investigations">List Investigations</RouterLink><RouterLink class="btn primary" to="/rca/incidents/new">Create Incident</RouterLink></div></header>
    <section class="card filters"><h2>Find incidents</h2><div class="filter-grid"><label>Search incident ID, summary, or details<input v-model="q" type="search" maxlength="200" placeholder="Search incidents" /></label><label>Severity<input v-model="severity" placeholder="e.g. critical" /></label><label>Service<input v-model="service" placeholder="e.g. payment" /></label><label>Created from<input v-model="createdFrom" type="date" /></label><label>Created to<input v-model="createdTo" type="date" /></label></div></section>
    <p v-if="error" class="error" role="alert">{{error}}</p><p v-if="notice" class="notice" role="status">{{notice}}</p><p v-if="busy" role="status" class="muted">Loading incidents...</p>
    <div v-else-if="!items.length" class="card empty">No active incidents match these filters. <RouterLink to="/rca/incidents/new">Create an incident</RouterLink>.</div>
    <div v-else class="table-wrap"><table><thead><tr><th>Incident ID</th><th>Summary</th><th>Start</th><th>End</th><th>Severity</th><th>Primary service</th><th>Owner</th><th>Evidence</th><th>Actions</th></tr></thead><tbody>
      <tr v-for="item in items" :key="item.id"><td><RouterLink :to="'/rca/incidents/'+item.id">{{item.source_incident_id||item.id}}</RouterLink></td><td>{{details(item,'summary')}}</td><td>{{details(item,'start_time')}}</td><td>{{details(item,'end_time')}}</td><td>{{details(item,'max_severity')}}</td><td>{{details(item,'primary_service')}}</td><td>{{item.userid}}</td><td>{{item.evidence_count||0}}</td><td class="row-actions"><RouterLink class="btn" :to="'/rca/incidents/'+item.id">Open</RouterLink><button v-if="item.can_delete" class="btn danger" :disabled="busy" @click="remove(item)">Permanently Delete</button></td></tr>
    </tbody></table></div>
    <div class="actions pager"><button class="btn" :disabled="busy||offset===0" @click="load(Math.max(0,offset-50))">Previous</button><span>Showing {{offset+1}} to {{offset+items.length}}</span><button class="btn" :disabled="busy||items.length<50" @click="load(offset+50)">Next</button></div>
  </section>
  <ConfirmDialog v-if="pendingDelete" title="Permanently delete incident?" :message="deleteMessage" confirm-label="Yes, permanently delete" cancel-label="No, keep incident" @confirm="confirmDelete" @cancel="cancelDelete" />
</template>
<style scoped>
.incidents{max-width:1450px}.view-header{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap}.actions{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}.row-actions{white-space:nowrap}.row-actions>*{margin-right:.35rem}.card{padding:1rem;background:#111c26;border:1px solid #30404e;border-radius:.65rem}.filter-grid{display:grid;grid-template-columns:1fr 2fr 1fr 1fr 1fr 1fr;gap:.8rem}.filter-grid label{display:grid;gap:.35rem}input,select{padding:.65rem;color:#e8edf2;background:#17232d;border:1px solid #344553;border-radius:.4rem;font:inherit}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.65rem;border-bottom:1px solid #30404e;vertical-align:top}.empty{padding:1rem}.pager{justify-content:flex-end;margin-top:1rem}.muted{color:#a7bbc8}.error{color:#ffaaa5}.notice{color:#8fe5ce}.btn{display:inline-block;padding:.45rem .65rem;border:1px solid #486071;border-radius:.4rem;color:inherit;text-decoration:none;background:#17232d;font:inherit;cursor:pointer}.primary{background:#078ed0;border-color:#078ed0;color:white}.danger{border-color:#b94a48;color:#ffaaa5}@media(max-width:1100px){.filter-grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.filter-grid{grid-template-columns:1fr}}
</style>
