import { createRouter, createWebHistory } from "vue-router";
import PipelineRunsView from "../views/PipelineRunsView.vue";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "pipeline-runs", component: PipelineRunsView },
  ],
});
