<script setup lang="ts">
import { onMounted, ref } from "vue";
import { listInvestigations } from "../services/api";
import type { Investigation } from "../types/api";

const items=ref<Investigation[]>([]);
const q=ref(""), status=ref(""), createdFrom=ref(""), createdTo=ref("");
const applied=ref({q:"",status:"",created_from:"",created_to:""});
const busy=ref(false), error=ref(""), offset=ref(0);
const date=(value:string)=>{const parsed=new Date(value);return Number.isNaN(parsed.getTime())?value:parsed.toLocaleString();};
async function load(nextOffset=0){
  if(busy.value)return;
  busy.value=true;error.value="";
  try{items.value=await listInvestigations(nextOffset,applied.value);offset.value=nextOffset;}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
async function search(){
  if(createdFrom.value&&createdTo.value&&createdFrom.value>createdTo.value){error.value="Created-from date must be on or before created-to date.";return;}
  applied.value={q:q.value.trim(),status:status.value,created_from:createdFrom.value,created_to:createdTo.value};
  await load(0);
}
async function clearFilters(){q.value="";status.value="";createdFrom.value="";createdTo.value="";applied.value={q:"",status:"",created_from:"",created_to:""};await load(0);}
onMounted(()=>void load());
</script>
<template>
<section class="view investigations">
  <header class="view-header"><div><h1>List Investigations</h1><p>Search and filter investigations, then open one to review its incidents, evidence, findings and RCA history.</p></div><RouterLink class="btn primary" to="/rca/investigations/new">Create Investigation</RouterLink></header>
  <form class="card filters" @submit.prevent="search">
    <h2>Find investigations</h2>
    <div class="form-grid">
      <label class="field">Search description or owner ID<input v-model="q" type="search" maxlength="200" placeholder="Search investigations" /></label>
      <label class="field">Status<select v-model="status"><option value="">All statuses</option><option value="OPEN">Open</option><option value="CLOSED">Closed</option></select></label>
      <label class="field">Created from<input v-model="createdFrom" type="date" /></label>
      <label class="field">Created through<input v-model="createdTo" type="date" /></label>
    </div>
    <div class="actions"><button class="btn primary" :disabled="busy" type="submit">Search</button><button class="btn" :disabled="busy" type="button" @click="clearFilters">Clear filters</button><button class="btn" :disabled="busy" type="button" @click="load(offset)">Refresh</button></div>
  </form>
  <section class="card">
    <p v-if="error" class="error" role="alert">{{error}}</p><p v-if="busy" role="status" class="muted">Loading investigations…</p>
    <div v-if="!busy && !error && !items.length" class="empty"><h3>{{offset?'No more investigations':Object.values(applied).some(Boolean)?'No matching investigations':'No investigations yet'}}</h3><p>{{Object.values(applied).some(Boolean)?'Try changing or clearing the search and filters.':'Create an investigation to organize incidents, evidence and findings.'}}</p><RouterLink v-if="!offset&&!Object.values(applied).some(Boolean)" class="btn primary" to="/rca/investigations/new">Create Investigation</RouterLink></div>
    <div v-if="items.length" class="table-wrap"><table><caption class="sr-only">Investigations</caption><thead><tr><th>Problem</th><th>Owner</th><th>Status</th><th>Created</th><th>Updated</th></tr></thead><tbody>
      <tr v-for="item in items" :key="item.id"><td><RouterLink :to="`/rca/investigations/${item.id}`">{{item.problem}}</RouterLink></td><td>{{item.userid}}</td><td><span class="pill" :class="{open:item.status==='OPEN'}">{{item.status}}</span></td><td>{{date(item.created_at)}}</td><td>{{date(item.updated_at)}}</td></tr>
    </tbody></table></div>
    <div class="actions pagination"><button class="btn" :disabled="busy||offset===0" @click="load(Math.max(0,offset-50))">Previous</button><span class="muted">Page {{Math.floor(offset/50)+1}} · {{items.length}} shown</span><button class="btn" :disabled="busy||items.length<50" @click="load(offset+50)">Next</button></div>
    <p class="muted timezone">Dates are shown in your browser's local timezone.</p>
  </section>
</section>
</template>
<style scoped>
.investigations{max-width:1400px}.view-header{flex-wrap:wrap}.section-heading{display:flex;justify-content:space-between;align-items:start;gap:1rem}.section-heading p{margin:.3rem 0 1rem}
.field{margin:.35rem 0}.actions{display:flex;gap:.5rem;flex-wrap:wrap}.empty{text-align:center;padding:2rem;color:#aebbc7}.pagination{justify-content:space-between;margin-top:1rem}.timezone{font-size:.8rem}
input,select{width:100%;padding:.65rem;color:#e8edf2;background:#17232d;border:1px solid #344553;border-radius:.45rem;font:inherit}.open{background:#203d3b;color:#8fe5ce}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
@media(max-width:650px){.form-grid{grid-template-columns:1fr}.card{padding:.85rem}.section-heading{flex-wrap:wrap}}
</style>
