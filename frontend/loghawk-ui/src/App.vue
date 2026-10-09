<template>
  <div class="app-shell">
    <aside class="sidebar">
      <RouterLink class="brand" to="/quick-run">
        <img class="brand-logo" src="/loghawk_logo.jpg" alt="LogHawk" />
      </RouterLink>
      <nav>
        <section>
          <h2>Config Sets</h2>
          <RouterLink to="/config-sets/new">Create Config Set</RouterLink>
          <RouterLink to="/config-sets">List Config Sets</RouterLink>
        </section>
        <section>
          <h2>Pipelines</h2>
          <RouterLink to="/pipelines/new">Create Pipeline</RouterLink>
          <RouterLink to="/pipelines">List Pipelines</RouterLink>
        </section>
        <section>
          <h2>Quick Run</h2>
          <RouterLink to="/quick-run">Start a quick run</RouterLink>
        </section>
      </nav>
      <p class="sidebar-note">AI decides and explains.<br />Deterministic workflows execute and verify.</p>
    </aside>
    <main class="content"><RouterView /></main>
  </div>
</template>

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
.app-shell { min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); }
.sidebar { position: sticky; top: 0; height: 100vh; display: flex; flex-direction: column; gap: 2rem; padding: 1.5rem 1rem; background: #0d151c; border-right: 1px solid #263743; }
.brand { display: block; text-decoration: none; }
.brand-logo { display: block; width: 100%; max-width: 216px; height: auto; object-fit: contain; }
.sidebar nav { display: grid; gap: 1.6rem; }
.sidebar nav section { display: grid; gap: .45rem; }
.sidebar nav h2 { margin: 0 0 .25rem; color: #93a7b6; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; }
.sidebar nav a { padding: .55rem .65rem; border-radius: .4rem; color: #d4dee5; text-decoration: none; font-size: .9rem; }
.sidebar nav a:hover, .sidebar nav a.router-link-active { color: #8fe5ce; background: #17252e; }
.sidebar-note { margin: auto 0 0; color: #728695; font-size: .76rem; line-height: 1.6; }
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
.actions { display: flex; flex-wrap: wrap; gap: .65rem; }
.btn { padding: .65rem .9rem; color: #e8edf2; background: #243442; border: 1px solid #405665; border-radius: .45rem; cursor: pointer; text-decoration: none; }
.btn.primary { color: #08131b; background: #6bd7bb; border-color: #6bd7bb; font-weight: 700; }
.btn:disabled { cursor: not-allowed; opacity: .45; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: .75rem .65rem; border-bottom: 1px solid #293a46; vertical-align: top; }
th { color: #93a7b6; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }
td { color: #d8e1e7; font-size: .88rem; }
a { color: #8fe5ce; }
.pill { display: inline-block; padding: .2rem .5rem; color: #c5d1d9; background: #263642; border-radius: 999px; font-size: .75rem; }
.source-list { display: grid; gap: .45rem; margin: .8rem 0; }
.source-item { display: flex; justify-content: space-between; gap: 1rem; align-items: center; padding: .55rem .7rem; border: 1px solid #2d404e; border-radius: .45rem; overflow-wrap: anywhere; }
@media (max-width: 760px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; gap: 1rem; padding: .8rem 1rem; }
  .sidebar nav { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .5rem; }
  .sidebar nav section { align-content: start; }
  .sidebar-note { display: none; }
  .content { padding: 1rem; }
}
@media (max-width: 520px) { .sidebar nav { grid-template-columns: 1fr; } .form-grid { grid-template-columns: 1fr; } }
</style>
