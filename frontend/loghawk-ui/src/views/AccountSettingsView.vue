<script setup lang="ts">
import { ref } from "vue";
import { changePassword, getCurrentUser } from "../services/api";

const user = ref<{ userid: string; role: string } | null>(null);
const currentPassword = ref("");
const newPassword = ref("");
const confirmPassword = ref("");
const error = ref("");
const success = ref("");
void getCurrentUser().then((value) => { user.value = value; });

async function submit() {
  error.value = ""; success.value = "";
  if (newPassword.value !== confirmPassword.value) { error.value = "The new passwords do not match."; return; }
  try {
    await changePassword(currentPassword.value, newPassword.value);
    currentPassword.value = ""; newPassword.value = ""; confirmPassword.value = "";
    success.value = "Password changed.";
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not change password"; }
}
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>Account Settings</h1><p v-if="user">{{ user.userid }} · {{ user.role }}</p></div></header>
    <form class="card form-grid" @submit.prevent="submit">
      <label class="field">Current password<input v-model="currentPassword" type="password" autocomplete="current-password" required /></label>
      <label class="field">New password<input v-model="newPassword" type="password" autocomplete="new-password" minlength="8" required /></label>
      <label class="field">Confirm new password<input v-model="confirmPassword" type="password" autocomplete="new-password" minlength="8" required /></label>
      <p v-if="error" class="error" style="grid-column:1/-1">{{ error }}</p><p v-if="success" class="success" style="grid-column:1/-1">{{ success }}</p>
      <div class="actions" style="grid-column:1/-1"><button class="btn primary">Change Password</button></div>
    </form>
  </div>
</template>
