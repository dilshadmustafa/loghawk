<script setup lang="ts">
import { nextTick, ref } from "vue";
import { useRouter } from "vue-router";
import { createInvestigation } from "../services/api";
const router=useRouter();
const problem=ref(""), entity=ref(""), environment=ref(""), start=ref(""), end=ref("");
const creating=ref(false), error=ref(""), problemInput=ref<HTMLTextAreaElement|null>(null);
const timestamp=(value:string)=>{
  if(!value.trim())return null;
  if(!/(Z|[+-]\d{2}:\d{2})$/i.test(value.trim())||Number.isNaN(Date.parse(value)))throw new Error("Include a timezone in each time, for example 2026-11-03T10:00:00Z or 2026-11-03T15:30:00+05:30.");
  return value.trim();
};
async function create(){
  if(creating.value||!problem.value.trim())return;
  creating.value=true;error.value="";
  try{
    const first=timestamp(start.value),last=timestamp(end.value);
    if(first&&last&&Date.parse(last)<Date.parse(first))throw new Error("Incident end must be at or after incident start.");
    const item=await createInvestigation(problem.value.trim(),{entity_id:entity.value.trim(),environment:environment.value.trim(),time_range:{start:first,end:last}});
    await router.push(`/rca/investigations/${item.id}`);
  }catch(e){error.value=String(e);}finally{creating.value=false;}
}
void nextTick(()=>problemInput.value?.focus());
</script>
<template>
<section class="view create-investigation">
  <header class="view-header"><div><h1>Create Investigation</h1><p>Describe the operational problem. You can add incidents, evidence and findings after creating it.</p></div><RouterLink class="btn" to="/rca/investigations">List Investigations</RouterLink></header>
  <form class="card" @submit.prevent="create">
    <h2>Investigation details</h2>
    <p v-if="error" class="error" role="alert">{{error}}</p>
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
      <div class="actions"><button class="btn primary" :disabled="!problem.trim()">{{creating?'Creating…':'Create Investigation'}}</button><RouterLink class="btn" to="/rca/investigations">Cancel</RouterLink></div>
    </fieldset>
  </form>
</section>
</template>
<style scoped>
.create-investigation{max-width:1000px}.view-header{flex-wrap:wrap}fieldset{border:0;padding:0;min-width:0}.field{display:block;margin:.8rem 0}textarea,input{width:100%;padding:.7rem;color:#e8edf2;background:#17232d;border:1px solid #344553;border-radius:.45rem;font:inherit}textarea{resize:vertical}details{margin:1rem 0}summary{cursor:pointer;color:#b8d9d3}.actions{display:flex;gap:.5rem;flex-wrap:wrap}button:focus-visible,textarea:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid #8fe5ce;outline-offset:3px}@media(max-width:650px){.form-grid{grid-template-columns:1fr}.card{padding:.85rem}.view-header{flex-wrap:wrap}}
</style>
