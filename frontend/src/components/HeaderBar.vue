<script setup>
import AppIcon from "./AppIcon.vue";

defineProps({
  user: { type: Object, default: null },
  busy: { type: Boolean, default: false },
});

defineEmits(["open-auth", "logout", "new-chat"]);
</script>

<template>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <span class="brand-badge"><AppIcon name="spark" :size="19" /></span>
        <div>
          <div class="brand-title">医疗知识库问答</div>
          <div class="brand-sub">Self/Corrective RAG · LangGraph</div>
        </div>
      </div>

      <div class="topbar-actions">
        <button
          class="btn ghost small"
          type="button"
          title="开始新对话"
          :disabled="busy"
          @click="$emit('new-chat')"
        >
          <AppIcon name="refresh" :size="15" />
          新对话
        </button>

        <template v-if="user">
          <span class="user-chip"><span class="dot"></span>{{ user.username }}</span>
          <button class="btn ghost small" type="button" @click="$emit('logout')">
            退出登录
          </button>
        </template>
        <button v-else class="btn small" type="button" @click="$emit('open-auth')">
          <AppIcon name="user" :size="15" />
          登录
        </button>
      </div>
    </div>
  </header>
</template>
