<script setup lang="ts">
import { ref } from "vue";
import { askRagQuestion } from "../services/api";
import type { RagAnswer } from "../types/api";

interface ChatEntry {
  role: "user" | "assistant";
  text: string;
  sources?: RagAnswer["sources"];
}

const question = ref("");
const sending = ref(false);
const error = ref("");
const entries = ref<ChatEntry[]>([]);

async function send() {
  const value = question.value.trim();
  if (!value || sending.value) return;
  const ragQuestion = value.replace(/^\/rag(?:\s+|$)/i, "").trim();
  if (!ragQuestion) {
    error.value = "Enter a question after /rag, for example: /rag What does the runbook say?";
    return;
  }
  entries.value.push({ role: "user", text: value });
  question.value = "";
  sending.value = true;
  error.value = "";
  try {
    const result: RagAnswer = await askRagQuestion(ragQuestion);
    entries.value.push({ role: "assistant", text: result.answer, sources: result.sources });
  } catch (err) {
    error.value = err instanceof Error ? err.message : "RAG chat failed";
  } finally {
    sending.value = false;
  }
}
</script>

<template>
  <section class="card rag-chat">
    <h2>Ask about indexed content</h2>
    <p class="muted">Answers use retrieved LanceDB content and show the source documents.</p>
    <div v-if="entries.length" class="rag-chat-history" aria-live="polite">
      <article v-for="(entry, index) in entries" :key="index" class="rag-chat-entry">
        <strong>{{ entry.role === "user" ? "You" : "LogHawk" }}</strong>
        <p>{{ entry.text }}</p>
        <ul v-if="entry.sources?.length" class="rag-sources">
          <li v-for="source in entry.sources" :key="source.reference">[{{ source.reference }}] {{ source.source }}</li>
        </ul>
      </article>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <form class="rag-chat-form" @submit.prevent="send">
      <label class="field">
        <span>Question</span>
        <textarea v-model="question" rows="3" placeholder="/rag Ask a question about the indexed documents…"></textarea>
      </label>
      <div class="actions"><button class="btn primary" :disabled="sending || !question.trim()">{{ sending ? "Searching…" : "Ask" }}</button></div>
    </form>
  </section>
</template>

<style scoped>
.rag-chat { display: grid; gap: .8rem; }
.rag-chat h2, .rag-chat p { margin-top: 0; }
.rag-chat-history { display: grid; gap: .8rem; max-height: 32rem; overflow: auto; }
.rag-chat-entry { padding: .8rem; border: 1px solid #2d404e; border-radius: .5rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.rag-chat-entry p { margin: .45rem 0; line-height: 1.55; }
.rag-sources { margin: .5rem 0 0; padding-left: 1.3rem; color: #aebbc7; font-size: .82rem; overflow-wrap: anywhere; }
.rag-chat-form { display: grid; gap: .7rem; }
textarea { width: 100%; padding: .7rem .75rem; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .45rem; resize: vertical; }
</style>
