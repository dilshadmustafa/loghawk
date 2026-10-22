<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { login, setAuthToken } from "../services/api";

const userid = ref("");
const password = ref("");
const error = ref("");
const busy = ref(false);
const route = useRoute();
const router = useRouter();

async function submit() {
  busy.value = true;
  error.value = "";
  try {
    const result = await login(userid.value, password.value);
    setAuthToken(result.access_token);
    await router.replace(typeof route.query.redirect === "string" ? route.query.redirect : "/quick-run");
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Sign in failed";
  } finally { busy.value = false; }
}
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>LogHawk Sign In</h1><p>Sign in with your LogHawk account.</p></div></header>
    <form class="card form-grid" @submit.prevent="submit">
      <label class="field">User ID<input v-model="userid" autocomplete="username" required /></label>
      <label class="field">Password<input v-model="password" type="password" autocomplete="current-password" required /></label>
      <p v-if="error" class="error" style="grid-column:1/-1">{{ error }}</p>
      <div class="actions" style="grid-column:1/-1"><button class="btn primary" :disabled="busy">{{ busy ? "Signing in…" : "Sign In" }}</button></div>
    </form>
  </div>
</template>
