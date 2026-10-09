import { createRouter, createWebHistory } from "vue-router";
import PipelineRunsView from "../views/PipelineRunsView.vue";
import ConfigSetListView from "../views/ConfigSetListView.vue";
import ConfigSetFormView from "../views/ConfigSetFormView.vue";
import PipelineListView from "../views/PipelineListView.vue";
import PipelineFormView from "../views/PipelineFormView.vue";
import PipelineRunDetailView from "../views/PipelineRunDetailView.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/quick-run" },
    { path: "/quick-run", name: "quick-run", component: PipelineRunsView },
    { path: "/config-sets", name: "config-sets", component: ConfigSetListView },
    { path: "/config-sets/new", name: "config-set-create", component: ConfigSetFormView },
    { path: "/config-sets/:id/edit", name: "config-set-edit", component: ConfigSetFormView },
    { path: "/pipelines", name: "pipelines", component: PipelineListView },
    { path: "/pipelines/new", name: "pipeline-create", component: PipelineFormView },
    { path: "/pipelines/:id/edit", name: "pipeline-edit", component: PipelineFormView },
    { path: "/pipeline-runs/:workflowId", name: "pipeline-run-detail", component: PipelineRunDetailView },
  ],
});
