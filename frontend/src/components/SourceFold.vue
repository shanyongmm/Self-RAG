<script setup>
import { isUrl, sourceTitle } from "../utils/trace";
import { truncate } from "../utils/format";
import AppIcon from "./AppIcon.vue";

const props = defineProps({
  title: { type: String, required: true },
  sources: { type: Array, default: () => [] },
  tone: { type: String, default: "accepted" }, // 'accepted' | 'rejected'
});

function urlOf(source) {
  if (!source) return "";
  if (isUrl(source.source)) return source.source;
  if (isUrl(source.chunk_id)) return source.chunk_id;
  return "";
}

function snippet(source) {
  return truncate(source && source.text, 200);
}

function scoreOf(source) {
  const value = source && source.score;
  if (typeof value !== "number") return "";
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(4);
}
</script>

<template>
  <details class="fold">
    <summary>
      <span class="chev"><AppIcon name="chev" :size="14" /></span>
      {{ title }} · {{ props.sources.length }}
    </summary>
    <div class="fold-body">
      <div v-for="(source, index) in props.sources" :key="index" class="source-item">
        <div class="source-head">
          <template v-if="urlOf(source)">
            <a :href="urlOf(source)" target="_blank" rel="noopener noreferrer">
              {{ sourceTitle(source) }}
            </a>
          </template>
          <template v-else>
            {{ sourceTitle(source) }}
          </template>
        </div>
        <div class="source-snippet">{{ snippet(source) }}</div>
        <div class="source-meta">
          <span v-if="scoreOf(source)">相关度 {{ scoreOf(source) }}</span>
          <a
            v-if="urlOf(source)"
            class="source-open"
            :href="urlOf(source)"
            target="_blank"
            rel="noopener noreferrer"
          >
            <AppIcon name="link" :size="12" /> 打开原文
          </a>
        </div>
      </div>
    </div>
  </details>
</template>
