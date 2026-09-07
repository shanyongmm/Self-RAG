<script setup>
import { reactive, ref } from "vue";

import { login, register } from "../api";

const emit = defineEmits(["close", "authed"]);

const mode = ref("login"); // 'login' | 'register'
const form = reactive({ username: "", password: "" });
const busy = ref(false);
const errorMsg = ref("");

function toggleMode() {
  mode.value = mode.value === "login" ? "register" : "login";
  errorMsg.value = "";
  form.password = "";
}

function close() {
  if (busy.value) return;
  emit("close");
}

async function submit() {
  const username = form.username.trim();
  const password = form.password;
  if (!username) {
    errorMsg.value = "请输入用户名";
    return;
  }
  if (mode.value === "register" && password.length < 6) {
    errorMsg.value = "密码至少需要 6 位";
    return;
  }
  busy.value = true;
  errorMsg.value = "";
  try {
    const caller = mode.value === "register" ? register : login;
    const data = await caller(username, password);
    emit("authed", { token: data.token, username: data.username });
  } catch (error) {
    errorMsg.value = error.message || (mode.value === "register" ? "注册失败" : "登录失败");
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="auth-overlay" @click.self="close">
      <div class="auth-card" role="dialog" aria-modal="true">
        <h2>{{ mode === "login" ? "登录" : "注册账号" }}</h2>
        <p class="sub">
          {{
            mode === "login"
              ? "登录后问答会按用户隔离,并沉淀个性化记忆"
              : "注册一个账号,用于保存跨会话的历史记忆"
          }}
        </p>

        <form @submit.prevent="submit">
          <label class="field-label" for="auth-username">用户名</label>
          <input
            id="auth-username"
            v-model="form.username"
            class="field"
            type="text"
            autocomplete="username"
            maxlength="32"
            placeholder="请输入用户名"
          />

          <label class="field-label" for="auth-password">密码</label>
          <input
            id="auth-password"
            v-model="form.password"
            class="field"
            type="password"
            :autocomplete="mode === 'register' ? 'new-password' : 'current-password'"
            :placeholder="mode === 'register' ? '至少 6 位密码' : '请输入密码'"
          />

          <div v-if="errorMsg" class="auth-error">{{ errorMsg }}</div>

          <div class="auth-actions">
            <button class="auth-link-btn" type="button" @click="toggleMode">
              {{ mode === "register" ? "已有账号?去登录" : "没有账号?去注册" }}
            </button>
            <button class="btn" type="submit" :disabled="busy">
              {{ busy ? "请稍候…" : mode === "login" ? "登录" : "注册" }}
            </button>
          </div>
        </form>
      </div>
    </div>
  </Teleport>
</template>
