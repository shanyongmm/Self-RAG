<script setup>
import { computed } from "vue";

import { formatAnswer } from "../utils/format";
import SourceFold from "./SourceFold.vue";
import TraceView from "./TraceView.vue";

const props = defineProps({
  message: { type: Object, required: true },
});

const isUser = computed(() => props.message.role === "user");

const answerHtml = computed(() => {
  const answer = props.message.answer || props.message.text || "";
  return formatAnswer(answer);
});

const note = computed(() => {
  const m = props.message;
  if (m.answer_source === "web" || m.evidence_origin === "web") {
    return {
      className: "note web",
      title: "联网检索",
      body: "本地知识库未找到直接支撑该问题的权威片段,以下内容由联网公开信息整理,仅供参考,不构成诊疗或政策建议。",
    };
  }
  if (m.answer_source === "uncovered" || m.uncovered) {
    return {
      className: "note uncovered",
      title: "未能回答",
      body: "本地知识库与联网公开信息均未找到足以支撑回答该问题的资料,建议补充更多信息后重试,或咨询专业医师。",
    };
  }
  return null;
});

const chips = computed(() => {
  const m = props.message;
  const list = [];

  if (m.answer_source === "web" || m.evidence_origin === "web") {
    list.push({ text: "来源:联网公开信息", cls: "chip amber" });
  } else if (m.answer_source === "uncovered" || m.uncovered) {
    list.push({ text: "来源:知识库未覆盖", cls: "chip slate" });
  } else if (m.status === "done") {
    list.push({ text: "来源:本地知识库", cls: "chip teal" });
  }

  if (m.retry_count && m.retry_count > 0) {
    list.push({ text: `改写重试 ${m.retry_count} 次`, cls: "chip" });
  }
  if (m.is_answerable === true && m.answer_source !== "web") {
    list.push({ text: "已给出可追溯回答", cls: "chip teal" });
  }
  if (Array.isArray(m.long_term_hits) && m.long_term_hits.length > 0) {
    list.push({ text: "已参考该用户历史记忆", cls: "chip" });
  }
  return list;
});

const hasAnswerText = computed(() => {
  const text = props.message.answer || props.message.text || "";
  return text.trim().length > 0;
});
</script>

<template>
  <article class="msg" :class="isUser ? 'msg-user' : 'msg-assistant'">
    <div class="bubble" :class="isUser ? 'bubble-user' : 'bubble-assistant'">
      <!-- 用户消息 -->
      <template v-if="isUser">
        <div class="user-text">{{ message.text }}</div>
      </template>

      <!-- 助手消息:加载中 -->
      <div v-else-if="message.status === 'pending'" class="typing-line">
        <span class="dots"><span></span><span></span><span></span></span>
        <span>正在检索知识库、判断相关性并组织回答…</span>
      </div>

      <!-- 助手消息:请求出错 -->
      <div v-else-if="message.status === 'error'" class="error-box">
        {{ message.error || "请求失败,请稍后重试" }}
      </div>

      <!-- 助手消息:完成 -->
      <template v-else>
        <div v-if="note" class="note" :class="note.className">
          <span class="note-title">{{ note.title }}</span>
          <span>{{ note.body }}</span>
        </div>

        <div v-if="hasAnswerText" class="answer-text" v-html="answerHtml"></div>
        <div v-else class="answer-text muted-text">(未返回答案文本)</div>

        <div v-if="chips.length" class="meta-row">
          <span v-for="(chip, index) in chips" :key="index" class="chip" :class="chip.cls">
            {{ chip.text }}
          </span>
        </div>

        <SourceFold
          v-if="Array.isArray(message.sources) && message.sources.length"
          title="引用来源"
          :sources="message.sources"
        />
        <SourceFold
          v-if="Array.isArray(message.rejected_sources) && message.rejected_sources.length"
          title="被过滤片段"
          :sources="message.rejected_sources"
          tone="rejected"
        />
        <TraceView
          v-if="Array.isArray(message.trace) && message.trace.length"
          :trace="message.trace"
        />
      </template>
    </div>
  </article>
</template>

<style scoped>
.muted-text {
  color: var(--muted);
  font-size: 14px;
}
</style>
