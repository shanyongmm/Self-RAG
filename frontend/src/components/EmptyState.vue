<script setup>
import AppIcon from "./AppIcon.vue";

defineEmits(["pick"]);

const SAMPLES = [
  "高血压患者血压一般控制在什么范围?",
  "国家基本药物目录包含哪些高血压常用药?",
  "糖尿病患者的转诊标准是什么?",
  "医疗器械唯一标识(UDI)有什么作用?",
  "阿莫西林常见的用药禁忌有哪些?",
];
</script>

<template>
  <div class="empty-state">
    <div class="empty-hero"><AppIcon name="doc" :size="30" /></div>
    <h1>医疗知识库智能问答</h1>
    <p>
      系统会先在本地知识库(临床指南 / 器械监管 / 药品目录)中检索,逐段判断相关性后再作答;
      证据不足时会自动改写提问重新检索,并给出可追溯的引用来源。
    </p>
    <div class="samples">
      <button
        v-for="sample in SAMPLES"
        :key="sample"
        class="sample-chip"
        type="button"
        @click="$emit('pick', sample)"
      >
        {{ sample }}
      </button>
    </div>
  </div>
</template>
