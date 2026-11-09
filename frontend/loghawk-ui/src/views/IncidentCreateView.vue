<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { createIncident } from "../services/api";
const router=useRouter();
const incidentId=ref(""), summary=ref(""), start=ref(""), end=ref(""), severity=ref(""), service=ref(""), services=ref("");
const busy=ref(false), error=ref("");
async function submit(){
  if(busy.value)return;
  error.value="";
  if(!incidentId.value.trim()||!start.value||!end.value){error.value="Incident ID, start time, and end time are required.";return;}
  if(Date.parse(end.value)<Date.parse(start.value)){error.value="Incident end must be at or after its start.";return;}
  busy.value=true;
  try{
    const item=await createIncident({incident_id:incidentId.value.trim(),summary:summary.value.trim(),start_time:new Date(start.value).toISOString(),end_time:new Date(end.value).toISOString(),severity:severity.value.trim(),primary_service:service.value.trim(),services:services.value.split(",").map(v=>v.trim()).filter(Boolean)});
    await router.push("/rca/incidents/"+item.id);
  }catch(e){error.value=String(e);}finally{busy.value=false;}
}
</script>
<template>
  <section class="view incident-create">
    <header class="view-header"><div><h1>Create Incident</h1><p>Create an incident record, then add supporting files, logs, metrics, or other evidence.</p></div><RouterLink class="btn" to="/rca/incidents">List Incidents</RouterLink></header>
    <form class="card" @submit.prevent="submit">
      <p v-if="error" class="error" role="alert">{{error}}</p>
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
  </section>
</template>
<style scoped>
.incident-create{max-width:1050px}.view-header{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}.card{padding:1.2rem;background:#111c26;border:1px solid #30404e;border-radius:.65rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.grid label{display:grid;gap:.4rem}.wide{grid-column:1/-1}input{padding:.7rem;color:#e8edf2;background:#17232d;border:1px solid #344553;border-radius:.4rem;font:inherit}.required,small{color:#a7bbc8;font-size:.85rem}.actions{display:flex;gap:.6rem;margin-top:1.2rem}.error{color:#ffaaa5}@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
