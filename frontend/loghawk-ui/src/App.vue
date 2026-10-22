<template>
  <div class="app-shell">
    <div class="left-navigation">
      <nav class="nav-rail" aria-label="Navigation groups">
        <button
          v-for="group in navGroups"
          :key="group.id"
          class="nav-group-button"
          :class="{ selected: selectedGroup === group.id }"
          :aria-pressed="selectedGroup === group.id"
          @click="selectedGroup = group.id"
        >
          <span class="nav-group-icon">{{ group.short }}</span>
          <span class="nav-group-label">{{ group.label }}</span>
        </button>
      </nav>

      <aside class="sidebar">
        <RouterLink class="brand" to="/quick-run">
          <img class="brand-logo" src="/loghawk_logo.jpg" alt="LogHawk" />
        </RouterLink>
        <section class="link-panel" :aria-label="activeGroup.label">
          <h2>{{ activeGroup.label }}</h2>
          <RouterLink
            v-for="link in activeGroup.links"
            :key="link.to"
            :to="link.to"
          >{{ link.label }}</RouterLink>
          <template v-if="selectedGroup === 'account'">
            <p v-if="!authEnabled" class="sidebar-note">Login is disabled.</p>
            <p v-else-if="!user" class="sidebar-note">Sign in to access account settings.</p>
            <template v-else>
              <RouterLink to="/account">Account Settings</RouterLink>
              <RouterLink v-if="user.role === 'Administrator'" to="/users">Manage Users</RouterLink>
              <button class="nav-button" @click="signOut">Sign Out · {{ user.userid }}</button>
            </template>
          </template>
        </section>
        <p class="sidebar-note brand-note">AI decides and explains.<br />Deterministic workflows execute and verify.</p>
      </aside>
    </div>
    <main class="content"><RouterView /></main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { clearAuthToken, getAuthStatus, getCurrentUser, getAuthToken, logout } from "./services/api";

const route = useRoute(); const router = useRouter();
const authEnabled = ref(false);
const user = ref<{ userid: string; role: string } | null>(null);
const navGroups = [
  {
    id: "config-pipelines",
    label: "ConfigSets & Pipelines",
    short: "CP",
    links: [
      { label: "Create Config Set", to: "/config-sets/new" },
      { label: "List Config Sets", to: "/config-sets" },
      { label: "Create Pipeline", to: "/pipelines/new" },
      { label: "List Pipelines", to: "/pipelines" },
      { label: "Start a quick run", to: "/quick-run" },
    ],
  },
  {
    id: "rag",
    label: "RAG Operations",
    short: "RAG",
    links: [
      { label: "Document Upload", to: "/rag/documents" },
      { label: "Web Scrape", to: "/rag/web-scrape" },
    ],
  },
  {
    id: "tasks",
    label: "Tasks & Collectors",
    short: "TC",
    links: [
      { label: "Create Task", to: "/tasks/new" },
      { label: "List Tasks", to: "/tasks" },
    ],
  },
  { id: "account", label: "Account", short: "A", links: [] },
];
const selectedGroup = ref(groupForPath(route.path));
const activeGroup = computed(() => navGroups.find((group) => group.id === selectedGroup.value) ?? navGroups[0]);

function groupForPath(path: string) {
  if (path.startsWith("/rag/")) return "rag";
  if (path.startsWith("/tasks")) return "tasks";
  if (path === "/account" || path === "/users" || path === "/login") return "account";
  return "config-pipelines";
}

async function refreshAuth() {
  const status = await getAuthStatus().catch(() => ({ webui_login: false, console_login: false }));
  authEnabled.value = status.webui_login;
  user.value = status.webui_login && getAuthToken() ? await getCurrentUser().catch(() => null) : null;
}
async function signOut() {
  try { await logout(); } catch { /* The server may have already expired the session. */ }
  clearAuthToken(); user.value = null; await router.replace("/login");
}
onMounted(() => void refreshAuth());
watch(() => route.fullPath, () => {
  selectedGroup.value = groupForPath(route.path);
  void refreshAuth();
});
</script>

<style>
:root {
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  color: #e8edf2;
  background: #101820;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; min-height: 100vh; }
button, select, input { font: inherit; }
.app-shell { min-height: 100vh; display: grid; grid-template-columns: 360px minmax(0, 1fr); }
.left-navigation { position: sticky; top: 0; height: 100vh; display: grid; grid-template-columns: 112px minmax(0, 1fr); background: #0d151c; border-right: 1px solid #263743; }
.nav-rail { display: flex; flex-direction: column; align-items: center; gap: .65rem; overflow-y: auto; padding: 1rem .45rem; border-right: 1px solid #263743; }
.nav-group-button { width: 96px; min-height: 96px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .4rem; padding: .55rem .4rem; color: #c4d0d8; background: #13202a; border: 1px solid #2d404e; border-radius: .65rem; cursor: pointer; text-align: center; }
.nav-group-button:hover { border-color: #6bd7bb; background: #172a31; }
.nav-group-button.selected { color: #071a19; background: #6bd7bb; border-color: #6bd7bb; }
.nav-group-icon { width: 1.8rem; height: 1.8rem; display: grid; place-items: center; border: 1px solid currentColor; border-radius: .45rem; font-size: .72rem; font-weight: 800; }
.nav-group-label { font-size: .67rem; font-weight: 650; line-height: 1.2; overflow-wrap: anywhere; }
.sidebar { min-width: 0; display: flex; flex-direction: column; gap: 1.5rem; padding: 1.25rem .75rem; }
.brand { display: block; text-decoration: none; }
.brand-logo { display: block; width: 100%; max-width: 216px; height: auto; object-fit: contain; }
.link-panel { display: grid; align-content: start; gap: .45rem; }
.link-panel h2 { margin: 0 0 .5rem; color: #93a7b6; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; line-height: 1.4; }
.link-panel a { padding: .55rem .65rem; border-radius: .4rem; color: #d4dee5; text-decoration: none; font-size: .9rem; }
.link-panel a:hover, .link-panel a.router-link-active { color: #8fe5ce; background: #17252e; }
.nav-button { padding: .55rem .65rem; border: 0; border-radius: .4rem; color: #d4dee5; background: transparent; text-align: left; cursor: pointer; font-size: .9rem; }
.nav-button:hover { color: #8fe5ce; background: #17252e; }
.sidebar-note { margin: .35rem 0; color: #728695; font-size: .76rem; line-height: 1.6; }
.brand-note { margin-top: auto; }
.content { min-width: 0; padding: 2rem clamp(1rem, 4vw, 3.5rem); }
.view { width: min(1100px, 100%); margin: 0 auto; display: grid; gap: 1.2rem; }
.view-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.view-header h1 { margin: 0; font-size: 1.65rem; }
.view-header p { margin: .45rem 0 0; color: #aebbc7; }
.card { padding: 1.25rem; background: #131f28; border: 1px solid #2d404e; border-radius: .75rem; }
.card h2 { margin: 0 0 1rem; font-size: 1.05rem; }
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.field { min-width: 0; display: grid; gap: .4rem; color: #aebbc7; font-size: .88rem; }
.field input, .field select { width: 100%; padding: .7rem .75rem; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .45rem; }
.field input:disabled, .field select:disabled { opacity: .65; }
.muted { color: #aebbc7; }
.error { color: #ff9f9f; }
.success { color: #8fe5ce; }
.actions { display: flex; flex-wrap: wrap; align-items: center; gap: .65rem; }
.btn { padding: .65rem .9rem; color: #e8edf2; background: #243442; border: 1px solid #405665; border-radius: .45rem; cursor: pointer; text-decoration: none; }
.btn.primary { color: #08131b; background: #6bd7bb; border-color: #6bd7bb; font-weight: 700; }
.btn:disabled { cursor: not-allowed; opacity: .45; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; text-align: left; }
.table-wrap th, .table-wrap td { padding: .75rem .65rem; border-bottom: 1px solid #293a46; vertical-align: middle; }
th { color: #93a7b6; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
td { color: #d8e1e7; font-size: .88rem; }
a { color: #8fe5ce; }
.pill { display: inline-block; padding: .2rem .5rem; color: #c5d1d9; background: #263642; border-radius: 999px; font-size: .75rem; }
.source-list { display: grid; gap: .45rem; margin: .8rem 0; }
.source-item { display: flex; justify-content: space-between; gap: 1rem; align-items: center; padding: .55rem .7rem; border: 1px solid #2d404e; border-radius: .45rem; overflow-wrap: anywhere; }
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; }
  .left-navigation { position: static; height: auto; grid-template-columns: 1fr; }
  .nav-rail { flex-direction: row; justify-content: flex-start; overflow-x: auto; overflow-y: hidden; padding: .7rem; border-right: 0; border-bottom: 1px solid #263743; }
  .nav-group-button { flex: 0 0 96px; }
  .sidebar { position: static; height: auto; display: grid; grid-template-columns: minmax(130px, 190px) minmax(0, 1fr); gap: 1rem; padding: .8rem 1rem; }
  .link-panel { grid-column: 2; grid-row: 1 / span 2; }
  .brand-note { display: none; }
  .content { padding: 1rem; }
}
@media (max-width: 520px) {
  .nav-rail { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); overflow: visible; }
  .nav-group-button { width: 100%; min-height: 76px; }
  .sidebar { grid-template-columns: 1fr; }
  .link-panel { grid-column: 1; grid-row: auto; }
  .form-grid { grid-template-columns: 1fr; }
}
</style>
