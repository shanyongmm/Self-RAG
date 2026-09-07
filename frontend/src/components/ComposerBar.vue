<script setup>
import { nextTick, ref } from "vue";

import AppIcon from "./AppIcon.vue";

const props = defineProps({
  disabled: { type: Boolean, default: false },
  placeholder: { type: String, default: "输入医疗知识相关的问题,例如:高血压患者血压控制在什么范围?" },
});

const emit = defineEmits(["send"]);

const text = ref("");
const area = ref(null);

function autoResize() {
  const el = area.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
}

function submit() {
  const value = text.value.trim();
  if (!value || props.disabled) return;
  emit("send", value);
  text.value = "";
  nextTick(() => {
    autoResize();
    if (area.value) area.value.focus();
  });
}

function onKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submit();
  }
}
</script>

<template>
  <div class="composer-wrap">
    <div class="composer-inner">
      <div class="composer">
        <textarea
          ref="area"
          v-model="text"
          rows="1"
          :placeholder="placeholder"
          :disabled="disabled"
          @input="autoResize"
          @keydown="onKeydown"
        ></textarea>
        <button
          class="composer-send"
          type="button"
          title="发送(Enter)"
          :disabled="disabled || !text.trim()"
          @click="submit"
        >
          <AppIcon name="send" :size="19" />
        </button>
      </div>
      <div class="composer-hint">Enter 发送 · Shift + Enter 换行</div>
    </div>
  </div>
</template>
