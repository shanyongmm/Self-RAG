<script setup>
import { computed } from "vue";

const props = defineProps({
  name: { type: String, required: true },
  size: { type: Number, default: 18 },
});

// 内置少量线性图标(Lucide 风格),避免为几个图标引入额外依赖。
const ICONS = {
  spark: `<path d="M12 3l1.9 5.8L20 10.7l-6.1 1.9L12 18.4l-1.9-5.8L4 10.7l6.1-1.9z"/><path d="M19 15.5l.8 2.4 2.4.8-2.4.8-.8 2.4-.8-2.4-2.4-.8 2.4-.8z"/>`,
  send: `<path d="M22 2 11 13"/><path d="M22 2l-7 20-4-9-9-4z"/>`,
  user: `<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>`,
  refresh: `<path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>`,
  plus: `<path d="M12 5v14"/><path d="M5 12h14"/>`,
  chev: `<polyline points="9 18 15 12 9 6"/>`,
  link: `<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>`,
  x: `<path d="M18 6 6 18"/><path d="M6 6l12 12"/>`,
  doc: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/>`,
};

const inner = computed(() => ICONS[props.name] || ICONS.doc);
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <g v-html="inner"></g>
  </svg>
</template>
