# 记忆系统升级方案:短记忆(会话内)+ 长记忆(跨会话)

> 状态:§2「短记忆修复」已实施(history_window 喂 grade/generate/rewrite,
> 见 `app/memory.py::build_history_window` 与 `rag/prompts.py`)。§3 长记忆(Mem0)
> 仍未实施,决策点集中在文末「需要拍板的事」。
> Stage 0「登录 + user_id」已上线:HMAC bearer token 鉴权、开放注册、LangGraph
> thread 按 `f"{user_id}::{thread_id}"` 命名空间隔离,为 Mem0 的稳定 user_id 铺路
> (见 `app/auth.py`、`app/api.py` 的 `/auth/*` 与前端登录条)。
> 背景:现有记忆分两层——Postgres LangGraph checkpointer(原始消息)+
> `app/memory.py` 的规则记忆(实体/摘要/追问改写)。本文给出:
> ① 短记忆为何脆弱、如何低成本修复;② 是否引入 Mem0 做跨会话长记忆;
> ③ 三层各自边界;④ 落地清单与验证方式。

---

## 0. TL;DR(一页结论)

| 你要解决的 | 结论 | 成本 |
| --- | --- | --- |
| 会话内多轮太弱(只记得上一轮、指代靠关键词) | **不引新服务**——把 checkpointer 里已有的 messages 最近 K 轮喂给 LLM,规则层瘦身成"闸门" | 低,只改 graph 消费方式 |
| 跨会话长期记忆(用户下次来还记得) | **可选引入 Mem0**(自托管),读在 memory_node、写异步带外 | 中,新增一个服务与存储 |

三层各管一段,**谁也别越界**:

```text
① 原始对话       checkpointer(state["messages"])   无损、会话内        → 保留
② 会话内可用上下文  = messages 最近 K 轮 → 拼给 LLM(取代脆弱规则摘要)   → 本次改造
③ 跨会话语义记忆   Mem0(search 读 / 异步 add 写)     有损、按 user     → 可选新增
```

> 反直觉但关键的一点:**短记忆的修复原料你们早就有了**
> (`state["messages"]` 已由 checkpointer 持久化),却从没喂给 grade/generate,
> 反而用一个有损的单槽摘要去扛上下文——这是脆弱性的根因。

---

## 1. 现状盘点(以代码为准)

| 层 | 载体 | 作用域 | 干了什么 | 问题 |
| --- | --- | --- | --- | --- |
| 消息历史 | `app/checkpoint.py` PostgresSaver,按 `thread_id` | 单会话 | 原始 messages + state 持久化,跨 `/ask` 续跑 | 存了但下游基本没用 |
| 规则记忆 | `app/memory.py`(`state["memory_context"]`) | 单会话 | 关键词+正则抽实体、判追问、拼 `contextual_question` | ①只记得上一轮;②实体靠关键词表,库外词全断;③有损 |
| 知识库 | `rag/vectorstore.py` → Milvus | 全局 | 文档切片检索 | 与记忆无关,别混 |

Graph 现状(`app/graph.py`):

```text
START → memory_node → retrieve_node → grade_node → decide_node
              ↑                                     │ 相关→generate_node→END
              │            rewrite_node ←───────────┘ (不相关则改写重检索)
```

- `memory_node`(:61):只调 `prepare_memory_context`(规则),产出 `contextual_question` 同时当 `retrieval_query`。
- `generate_node`(:245):调 `update_memory_context` 写回 `memory_context`。
- **grade/generate 都看不到历史 messages**,只吃"当前问题 + contextual_question + 检索结果"。

---

## 2. 短记忆修复(会话内,必须做,低成本)

### 2.1 为什么脆弱

1. **只记得上一轮**:state 里是 `last_*` + 单行 ≤180 字 `summary`,三轮以上退化。
2. **实体抽取是规则**:`_extract_entities` 靠 `ENTITY_HINTS` 关键词表 + 正则,
   "二甲双胍"这类不在表里的实体直接断。
3. **有损替代无损**:checkpointer 明明存着逐字 messages,节点却不用,偏去用摘要。

### 2.2 改法(轻量、近零延迟、无损)

1. `OverAllState` 继承 `MessagesState`,`state["messages"]` 天然可用
   (含上一轮 H/A 与当前 H)。
2. 新增 state 字段 `history_window: str`:
   取最近 N 轮(默认 4),带角色拼成一段,按 token 预算截断。
3. `app/memory.py` **瘦身**:
   - **保留**:分区携带(`active_partition_id`)、`is_follow_up` 粗判(便宜闸门)。
   - **删掉**:用实体 + 单行摘要扛上下文的那部分——指代理解交给 LLM 看原话。
4. `contextual_question` 逻辑:
   - 非追问:仍用原问题(可顺带附分区)。
   - 追问:在问题前拼 `history_window`,让 LLM 自己消解"它/上面/那款药"。
5. `rag/prompts.py` 的 `build_generate_messages` 增加入参,把 `history_window`
   渲染成 prompt 里的「对话历史」段;检索证据仍只来自 Milvus。

### 2.3 明确不做的取舍

- **不上 LLM 滚动摘要**(每轮多一次模型调用 + 延迟)。
  先用无损历史;若日后评测发现历史太长拖质量,再升级为 LLM 压缩。
- 不改分区路由、不改 checkpointer、不动 Milvus。

---

## 3. 长记忆(跨会话,可选,用 Mem0)

> 只在你确认"用户会跨会话回来找连续性"的场景才有价值;单次会话即结束的
> demo,长记忆是空转成本。

### 3.1 为什么是 Mem0

- 两阶段记忆:**extract(抽取)→ update(合并/去重/过期)**,把每轮蒸馏成
  **可语义检索**、按 `user_id` 隔离的记忆,天然解决"上周那条 thread 问过,这周新 thread 追问"。
- 取代的是 `memory.py`,**不是 checkpointer**(Mem0 不实现 LangGraph checkpointer 协议)。

### 3.2 设计

1. **身份映射**:
   - 服务端只从 **HMAC bearer token** 解析 `user_id`(`/auth/login`、`/auth/register`
     签发;无 token → 匿名默认单用户),**不信任 body**。
   - LangGraph 内部 thread = `f"{user_id}::{resolved_thread_id}"`(匿名保持原名)。
   - Mem0:`user_id` = 稳定用户(跨会话),`run_id` = `thread_id`(追溯)。
2. **封装**:新增 `rag/mem0_service.py` → `create_memory(config)`;
   env 增 `MEM0_ENABLED` / `MEM0_STORE`(milvus | pgvector)/ 复用现有 LLM 与 embedding 配置。
   - LLM(抽取):`provider=openai`,base_url=LLM_BASE_URL,model=LLM_MODEL,key=LLM_API_KEY。
   - Embedder:DashScope OpenAI 兼容 `/embeddings`,dims=EMBED_DIMENSION(必须与 store 一致)。
   - Vector store:默认复用本地 Milvus(独立 collection);备选 pgvector(用 POSTGRES_URI)。
3. **读(在 `memory_node`)**:`memory.search(question, user_id, limit=3)` → `memory_hits`;
   追问时把命中片段拼进 `contextual_question`。
4. **写(带外,不堵主链路)**:`generate_node` 后**不在图里同步 add**(内含一次 LLM 抽取,
   会加重当前 ~18s 延迟);在 `RagStarter.ask` 拿到 answer 后异步触发
   (BackgroundTasks / 简单线程),写 `[user问, assistant答]` + `user_id/run_id`。
5. **写闸门**:仅 `is_answerable` 且命中 KB 证据的轮次才写;
   `missing_info` / 答非所问不沉淀。

### 3.3 医疗红线

- **记忆永远不是证据**:命中片段只当 generate 的背景(标注"历史对话,仅供参考,
  须以知识库为准"),**绝不进 grade/decide 的证据判据**;`citations` 仍只指向 KB chunk。
- **隐私**:多用户隔离已上线(登录后 thread 按 `user_id` 命名空间,互不可见);
  匿名仍是默认单用户。做 Mem0 前仍须先定授权范围、不落原始健康数据、
  提供 `delete_all`/过期接口。
- **错误记忆会传播**:Mem0 能合并去重,但改不了"模型抽错";用评测盯记忆是否把检索带偏。

---

## 4. 落地清单(改动文件)

| 文件 | 改什么 |
| --- | --- |
| `app/api.py`(注:`AskRequest` 定义于此,非 schemas.py) | 已从 token 取 `user_id` 透传;state/返回已透出 `user_id/history_window`;answer 后异步触发 Mem0 写(受 `MEM0_ENABLED` 控制) |
| `app/api.py` + `app/rag_starter.py` | 已按 token 的 `user_id` 命名空间隔离 thread;Mem0 写接入点仍在此 |
| `app/memory.py` | 瘦身:保留分区 + 追问闸门;新增 `build_history_window(messages)` |
| `app/graph.py` | `memory_node` 产出 `history_window` + mem0 命中;`generate_node` 传入并在返回点触发写钩子 |
| `rag/prompts.py` | `build_generate_messages` 支持「对话历史」与「长期记忆(参考)」两段 |
| 新增 `rag/mem0_service.py` | Mem0 初始化/关闭;config 从 `app/config.py` 读 |
| `app/config.py` | 加 `user_id` 默认、`MEM0_ENABLED`、`MEM0_STORE`、历史窗口轮数等 |
| `evaluation/` | 补多轮追问用例;开关做 A/B:关 / 仅短记忆 / 短+长记忆 |

---

## 5. 分阶段落地 + 验证(防止"加了一堆没人证明有用")

- **Stage 0 身份与开关**:加 `user_id`;`MEM0_ENABLED=false` 空转不动。
  **(已实施:登录 + token 派生 user_id + thread 命名空间,2026-09-06)**
- **Stage 1 只做短记忆**:喂 messages 历史,跑现有 eval + 一组**多轮脚本**,
  对比 关/开 的 `is_relevant` 与答案 —— 先证明短记忆修复有效。
- **Stage 2 长记忆只读**:Mem0 空库、只 search 不写,验证命中注入对追问检索有无增益。
- **Stage 3 长记忆异步写**:加闸门 + 幂等 key,盯记忆库增长与重复率。
- **Stage 4 A/B 收口**:与现状对照,量化延迟与准确率 delta;
  若多轮占比极低,把长记忆优先级让回检索本身。

> 现状 README 的 85% 准确率大概率是**单轮**评测;多轮是这套方案要新建立的度量。

---

## 6. 需要拍板的事(明天你要决定的)

1. **短记忆粒度**:推荐「最近 4 轮无损历史」起步(低成本);备选每轮 LLM 滚动压缩(准但贵)。
2. **长记忆是否上 Mem0**:取决于产品是否有"用户跨会话回来找连续性";没有就只做 2。
3. **Mem0 存储后端**:推荐复用本地 Milvus(独立 collection);备选 pgvector(用 POSTGRES_URI)。
4. **身份来源**:demo 无登录,先用可选 `user_id`(默认 thread_id)过渡,还是先只做会话内?
5. **改动范围**:是否接受一次动 7~8 个文件的增量改造,还是想先挑短记忆单独落地一轮?

> 依赖备注:Mem0 版本迭代快,config 键与返回结构不稳定;实施前以安装版本官方文档为准。
