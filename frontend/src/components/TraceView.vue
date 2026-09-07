<script setup>
import { computed } from "vue";

import { summarizeTrace } from "../utils/trace";
import AppIcon from "./AppIcon.vue";

const props = defineProps({
  trace: { type: Array, default: () => [] },
});

const steps = computed(() => summarizeTrace(props.trace));
</script>

<template>
  <details class="fold">
    <summary>
      <span class="chev"><AppIcon name="chev" :size="14" /></span>
      处理过程 · {{ steps.length }} 步
    </summary>
    <div class="fold-body">
      <div v-for="step in steps" :key="step.key" class="trace-step">
        <div class="trace-title">
          {{ step.label }}
          <span v-if="step.round !== null" class="round">第 {{ step.round }} 轮</span>
        </div>
        <div class="trace-lines">
          <div v-for="(line, i) in step.lines" :key="i">{{ line }}</div>
        </div>
      </div>
    </div>
  </details>
</template>
