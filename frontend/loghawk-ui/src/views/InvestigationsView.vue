<script setup lang="ts">
import { nextTick, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { listInvestigations, createInvestigation } from "../services/api";
import type { Investigation } from "../types/api";

const router=useRouter();
const items=ref<Investigation[]>([]), problem=ref(""), entity=ref(""), environment=ref(""), start=ref(""), end=ref("");
const busy=ref(false), creating=ref(false), error=ref(""), createError=ref(""), offset=ref(0), newOpen=ref(false);
const problemInput=ref<HTMLTextAreaElement|null>(null);
const date=(value:string)=>{const parsed=new Date(value);return Number.isNaN(parsed.getTime())?value:parsed.toLocaleString();};
async function openCreate(){newOpen.value=true;await nextTick();problemInput.value?.focus();}
async function load(nextOffset=offset.value){
  if(busy.value)return;
  busy.value=true;error.value="";
  try{const result=await listInvestigations(nextOffset);items.value=result;offset.value=nextOffset;}
  catch(e){error.value=String(e);}finally{busy.value=false;}
}
function timestamp(value:string){
  if(!value.trim())return null;
  if(!/(Z|[+-]\d{2}:\d{2})$/i.test(value.trim())||Number.isNaN(Date.parse(value)))throw new Error("Include a timezone in each time, for example 2026-11-03T10:00:00Z or 2026-11-03T15:30:00+05:30.");
  return value.trim();
}
async function create(){
  if(creating.value||!problem.value.trim())return;
  creating.value=true;createError.value="";
  try{
    const first=timestamp(start.value),last=timestamp(end.value);
    if(first&&last&&Date.parse(last)<Date.parse(first))throw new Error("Incident end must be at or after incident start.");
    const item=await createInvestigation(problem.value.trim(),{entity_id:entity.value.trim(),environment:environment.value.trim(),time_range:{start:first,end:last}});
    await router.push(`/rca/investigations/${item.id}`);
  }catch(e){createError.value=String(e);}finally{creating.value=false;}
}
onMounted(()=>void load());
</script>
<template>
<section class="view investigations">
  <header class="view-header"><div><h1>Investigations</h1><p>Bring incidents, evidence and findings together, then investigate with RCA Chat.</p></div><button class="btn primary" :disabled="creating" @click="openCreate">New Investigation</button></header>
  <form v-if="newOpen" class="card" @submit.prevent="create">
    <h2>Start an investigation</h2><p class="muted">Describe the problem first. You can add files, incidents and evidence after creating it.</p>
    <p v-if="createError" class="error" role="alert">{{createError}}</p>
    <fieldset :disabled="creating">
      <label class="field">Problem description <span class="muted">Required</span><textarea ref="problemInput" v-model="problem" required maxlength="10000" rows="3" placeholder="For example: Payment requests failed during the morning deployment." /></label>
      <details><summary>Reported context (optional)</summary><p class="muted">These details describe the reported issue; they are not independently verified facts.</p>
        <div class="form-grid">
          <label class="field">Entity / service<input v-model="entity" placeholder="payment-service" /></label>
          <label class="field">Environment<input v-model="environment" placeholder="Production" /></label>
          <label class="field">Incident start<input v-model="start" placeholder="2026-11-03T10:00:00Z" aria-describedby="timezone-hint" /></label>
          <label class="field">Incident end<input v-model="end" placeholder="2026-11-03T10:15:00Z" aria-describedby="timezone-hint" /></label>
        </div><p id="timezone-hint" class="muted">Include Z for UTC or an offset such as +05:30. Leave unknown times blank.</p>
      </details>
      <div class="actions"><button class="btn primary" :disabled="!problem.trim()">{{creating?'Creating…':'Create Investigation'}}</button><button type="button" class="btn" @click="newOpen=false">Cancel</button></div>
    </fieldset>
  </form>
  <section class="card">
    <div class="section-heading"><div><h2>Saved investigations</h2><p class="muted">Open an investigation to continue or review its evidence and history.</p></div><button class="btn" :disabled="busy||creating" @click="load()">Refresh</button></div>
    <p v-if="error" class="error" role="alert">{{error}}</p><p v-if="busy" role="status" class="muted">Loading investigations…</p>
    <div v-if="!busy && !error && !items.length" class="empty"><h3>{{offset?'No more investigations':'No investigations yet'}}</h3><p>{{offset?'Go back to the previous page.':'Start with a problem description, then attach evidence from files or saved collector results.'}}</p><button v-if="!offset" class="btn primary" @click="openCreate">Create Your First Investigation</button></div>
    <div v-if="items.length" class="table-wrap"><table><caption class="sr-only">Saved investigations</caption><thead><tr><th>Problem</th><th>Owner</th><th>Status</th><th>Created</th><th>Updated</th></tr></thead><tbody>
      <tr v-for="item in items" :key="item.id"><td><RouterLink :to="`/rca/investigations/${item.id}`">{{item.problem}}</RouterLink></td><td>{{item.userid}}</td><td><span class="pill" :class="{open:item.status==='OPEN'}">{{item.status}}</span></td><td>{{date(item.created_at)}}</td><td>{{date(item.updated_at)}}</td></tr>
    </tbody></table></div>
    <div class="actions pagination"><button class="btn" :disabled="busy||creating||offset===0" @click="load(Math.max(0,offset-50))">Previous</button><span class="muted">Page {{Math.floor(offset/50)+1}} · {{items.length}} shown</span><button class="btn" :disabled="busy||creating||items.length<50" @click="load(offset+50)">Next</button></div>
    <p class="muted timezone">Dates are shown in your browser's local timezone.</p>
  </section>
</section>
</template>
<style scoped>
.investigations{max-width:1400px}.view-header{flex-wrap:wrap}.section-heading{display:flex;justify-content:space-between;align-items:start;gap:1rem}.section-heading p{margin:.3rem 0 1rem}
fieldset{border:0;padding:0;min-width:0}.field{margin:.8rem 0}textarea{width:100%;padding:.7rem;color:#e8edf2;background:#17232d;border:1px solid #344553;border-radius:.45rem;font:inherit;resize:vertical}
details{margin:1rem 0}summary{cursor:pointer;color:#b8d9d3}.empty{text-align:center;padding:2rem;color:#aebbc7}.pagination{justify-content:space-between;margin-top:1rem}.timezone{font-size:.8rem}
td:first-child{min-width:15rem;max-width:34rem;overflow-wrap:anywhere}.open{background:#203d3b;color:#8fe5ce}button:focus-visible,textarea:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
@media(max-width:650px){.form-grid{grid-template-columns:1fr}.card{padding:.85rem}.section-heading{flex-wrap:wrap}}
</style>
