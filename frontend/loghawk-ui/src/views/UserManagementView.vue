<script setup lang="ts">
import { onMounted, ref } from "vue";
import { createUser, deleteUser, getCurrentUser, getUsers, type AuthUser } from "../services/api";

const users = ref<AuthUser[]>([]);
const userid = ref(""); const password = ref(""); const confirmPassword = ref(""); const role = ref<"User" | "Administrator">("User");
const error = ref(""); const success = ref(""); const busy = ref(false); const deletingUserid = ref("");
const currentUserid = ref("");
async function refresh() {
  const [listedUsers, currentUser] = await Promise.all([getUsers(), getCurrentUser()]);
  users.value = listedUsers;
  currentUserid.value = currentUser.userid;
}
function canDelete(item: AuthUser) {
  if (item.userid === currentUserid.value) return false;
  const activeAdministrators = users.value.filter((user) => user.role === "Administrator" && user.active);
  return !(item.role === "Administrator" && item.active && activeAdministrators.length <= 1);
}
async function submit() {
  error.value = ""; success.value = ""; busy.value = true;
  if (password.value !== confirmPassword.value) {
    error.value = "Passwords do not match.";
    busy.value = false;
    return;
  }
  try { await createUser(userid.value, password.value, role.value); userid.value = ""; password.value = ""; confirmPassword.value = ""; role.value = "User"; await refresh(); success.value = "User created."; }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not create user"; }
  finally { busy.value = false; }
}
async function remove(item: AuthUser) {
  if (!window.confirm(
    `Delete user "${item.userid}"? Their sessions and Console chat/command history will be removed. ` +
    "Tasks and RAG documents will remain associated with this user ID."
  )) return;
  deletingUserid.value = item.userid; error.value = ""; success.value = "";
  try {
    await deleteUser(item.userid);
    await refresh();
    success.value = `User "${item.userid}" deleted.`;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not delete user";
  } finally { deletingUserid.value = ""; }
}
onMounted(() => { void refresh().catch((err) => { error.value = err instanceof Error ? err.message : "Could not load users"; }); });
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>User Management</h1><p>Available to Administrators.</p></div></header>
    <form class="card form-grid" @submit.prevent="submit">
      <label class="field">User ID<input v-model="userid" autocomplete="off" required /></label>
      <label class="field">Initial password<input v-model="password" type="password" minlength="8" required /></label>
      <label class="field">Role<select v-model="role"><option>User</option><option>Administrator</option></select></label>
      <label class="field">Confirm password<input v-model="confirmPassword" type="password" minlength="8" required /></label>
      <p v-if="confirmPassword && password !== confirmPassword" class="error" style="grid-column:1/-1">Passwords do not match.</p>
      <p class="muted" style="grid-column:1/-1">Users can change their password from Account Settings after signing in.</p>
      <p v-if="error" class="error" style="grid-column:1/-1">{{ error }}</p><p v-if="success" class="success" style="grid-column:1/-1">{{ success }}</p>
      <div class="actions" style="grid-column:1/-1"><button class="btn primary" :disabled="busy || password !== confirmPassword">Create User</button></div>
    </form>
    <section class="card table-wrap"><h2>Users</h2><table><thead><tr><th>User ID</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead><tbody><tr v-for="item in users" :key="item.userid"><td>{{ item.userid }}</td><td>{{ item.role }}</td><td>{{ item.active ? "Active" : "Disabled" }}</td><td><button class="btn" :disabled="deletingUserid === item.userid || !canDelete(item)" :title="item.userid === currentUserid ? 'You cannot delete your own account.' : undefined" @click="remove(item)">{{ deletingUserid === item.userid ? "Deleting…" : "Delete" }}</button></td></tr></tbody></table></section>
  </div>
</template>
