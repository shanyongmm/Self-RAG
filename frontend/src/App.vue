<script setup>
import { computed, nextTick, onMounted, reactive, ref } from "vue";

import { askSelfRag, clearAuth, getAuth, newThreadId, setAuth } from "./api";
import AuthModal from "./components/AuthModal.vue";
import ChatMessage from "./components/ChatMessage.vue";
import ComposerBar from "./components/ComposerBar.vue";
import EmptyState from "./components/EmptyState.vue";
import HeaderBar from "./components/HeaderBar.vue";

const THREAD_KEY = "self-rag-thread-id";

const messages = ref([]);
const sending = ref(false);
const authOpen = ref(false);
const user = ref(getAuth());
const scrollEl = ref(null);
const toast = ref("");
let toastTimer = null;

const threadId = ref("");

function loadThreadId() {
  try {
    const saved = localStorage.getItem(THREAD_KEY);
    threadId.value = saved && saved.length ? saved : newThreadId();
    localStorage.setItem(THREAD_KEY, threadId.value);
  } catch {
    threadId.value = newThreadId();
  }
}

function showToast(text) {
  toast.value = text;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.value = "";
  }, 3000);
}

function scrollToBottom() {
  nextTick(() => {
    const el = scrollEl.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

function pushMessage(partial) {
  // 用 reactive 包裹:send() 里请求回来后要 Object.assign/改 status,
  // 普通对象在 push 后直接原地改会绕过响应式代理,Vue 不会刷新(气泡卡在加载态)。
  const message = reactive({ id: newThreadId(), ...partial });
  messages.value.push(message);
  scrollToBottom();
  return message;
}

const showEmpty = computed(() => messages.value.length === 0 && !sending.value);

async function send(rawQuestion) {
  const question = String(rawQuestion || "").trim();
  if (!question || sending.value) return;

  pushMessage({ role: "user", text: question });
  const assistant = pushMessage({ role: "assistant", status: "pending", text: "" });
  sending.value = true;

  try {
    const data = await askSelfRag(question, threadId.value, user.value);
    Object.assign(assistant, data);
    assistant.status = "done";
  } catch (error) {
    assistant.status = "error";
    if (error && error.status === 401) {
      clearAuth();
      user.value = null;
      authOpen.value = true;
      assistant.error = "登录状态已失效,请重新登录后再试。";
    } else {
      assistant.error =
        (error && error.message) || "请求失败,请确认后端服务已启动后重试。";
    }
  } finally {
    sending.value = false;
    scrollToBottom();
  }
}

function resetConversation() {
  if (sending.value) return;
  threadId.value = newThreadId();
  try {
    localStorage.setItem(THREAD_KEY, threadId.value);
  } catch {
    // 忽略存储不可用的情况
  }
  messages.value = [];
  showToast("已开启新对话,上下文已清空");
}

function onAuthed(credentials) {
  try {
    setAuth(credentials.token, credentials.username);
  } catch {
    // 忽略
  }
  user.value = getAuth();
  authOpen.value = false;
  showToast(`已登录:${credentials.username},问答将按用户隔离并沉淀记忆`);
}

function logout() {
  clearAuth();
  user.value = null;
  showToast("已退出登录,当前为匿名会话");
}

onMounted(loadThreadId);
</script>

<template>
  <div class="app">
    <HeaderBar
      :user="user"
      :busy="sending"
      @open-auth="authOpen = true"
      @logout="logout"
      @new-chat="resetConversation"
    />

    <main ref="scrollEl" class="chat-shell">
      <div class="chat-inner">
        <EmptyState v-if="showEmpty" @pick="send" />
        <div v-else class="messages">
          <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
        </div>
      </div>
    </main>

    <ComposerBar :disabled="sending" @send="send" />

    <AuthModal v-if="authOpen" @close="authOpen = false" @authed="onAuthed" />

    <Transition name="toast">
      <div v-if="toast" class="toast-msg">{{ toast }}</div>
    </Transition>
  </div>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translate(-50%, 6px);
}
</style>
