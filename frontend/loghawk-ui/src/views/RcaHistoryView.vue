<script setup lang="ts">
import { onMounted, ref } from "vue";
import { getRcaHistory, getCurrentUser } from "../services/api";
import type { RcaHistoryPage } from "../types/api";
const result = ref<RcaHistoryPage | null>(null);
const busy = ref(false), error = ref(""), allUsers = ref(false);
const search = ref(""), userid = ref(""), origin = ref(""), collector = ref(""), status = ref("");
const from = ref(""), to = ref("");
const offset = ref(0), pageSize = 25;
const localTime = (v: string) => new Date(v).toLocaleString();
async function load(reset = false) {
  if (busy.value) return;
  if (reset) offset.value = 0;
  busy.value = true; error.value = "";
  try {
    const query = new URLSearchParams({ search: search.value, userid: userid.value,
      origin: origin.value, collector: collector.value, status: status.value,
      limit: String(pageSize), offset: String(offset.value) });
    if (from.value) query.set("date_from", new Date(from.value).toISOString());
    if (to.value) query.set("date_to", new Date(to.value).toISOString());
    result.value = await getRcaHistory(query);
  } catch (e) { error.value = e instanceof Error ? e.message : String(e); result.value = null; }
  finally { busy.value = false; }
}
function page(delta: number) { offset.value = Math.max(0, offset.value + delta); void load(); }
onMounted(async () => {
  try { allUsers.value = ["Administrator", "Power User"].includes((await getCurrentUser()).role); }
  catch { /* The history endpoint independently enforces authorization. */ }
  await load();
});
</script>
<template>
  <section>
    <h1>RCA History</h1>
    <p>Saved server investigations from Web UI, Console and CLI. Times use your local timezone.
      {{ allUsers ? 'You can view all users’ history.' : 'Only your own history is shown.' }}</p>
    <form @submit.prevent="load(true)" class="filters">
      <label>Question / Request ID / Run ID<input v-model="search" maxlength="300" /></label>
      <label v-if="allUsers">User ID<input v-model="userid" maxlength="200" /></label>
      <label>Origin<select v-model="origin"><option value="">All</option><option>web</option><option>console</option><option>cli</option></select></label>
      <label>Collector name<input v-model="collector" maxlength="200" /></label>
      <label>Request status<select v-model="status"><option value="">All</option><option>PROCESSING</option><option>PENDING</option><option>EXPLAINING</option><option>COMPLETED</option><option>FAILED</option><option>CANCELLED</option><option>EXPIRED</option></select></label>
      <label>From<input type="datetime-local" v-model="from" /></label>
      <label>Before<input type="datetime-local" v-model="to" /></label>
      <button :disabled="busy">Search / Refresh</button>
    </form>
    <p v-if="busy" role="status">Loading history…</p><p v-if="error" role="alert">{{ error }}</p>
    <div class="table-scroll" v-if="result">
      <table><thead><tr><th>Started</th><th>Question</th><th>User</th><th>Origin</th><th>Execution</th><th>AI explanation</th><th>Request ID</th></tr></thead>
        <tbody><tr v-for="item in result.items" :key="item.id">
          <td>{{ localTime(item.created_at) }}</td><td>{{ item.question_text ?? 'Not recorded' }}</td>
          <td>{{ item.userid }}</td><td>{{ item.origin_software }}</td><td>{{ item.execution_summary }}</td>
          <td>{{ item.explanation_status ?? 'Not recorded' }}</td>
          <td><RouterLink :to="`/rca/history/${item.id}`">{{ item.id }}</RouterLink></td>
        </tr></tbody></table>
      <p v-if="!result.items.length">No matching RCA requests.</p>
      <p>{{ result.total }} request(s) · Page {{ Math.floor(offset / pageSize) + 1 }}</p>
      <button :disabled="busy || offset === 0" @click="page(-pageSize)">Previous</button>
      <button :disabled="busy || offset + pageSize >= result.total" @click="page(pageSize)">Next</button>
    </div>
  </section>
</template>
<style scoped>
.filters { display: flex; flex-wrap: wrap; align-items: end; gap: .8rem; margin: 1rem 0; }
.filters label { display: flex; flex-direction: column; gap: .3rem; }
.table-scroll { overflow-x: auto; } td { vertical-align: middle; overflow-wrap: anywhere; max-width: 26rem; }
button { margin-right: .5rem; }
</style>
