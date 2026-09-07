# 教学：从文档到可检索知识库 —— 解析 / 切片 / 向量库 / 分区与索引 全链路

> 本文档把「一份原始 PDF 最终变成用户能按语义检索到、并按医疗领域分区命中」的整条
> 数据链路拆开讲。每一站都给出：它在哪个文件、入口函数是哪几行、关键代码做了什么、
> 为什么要这么做。请对照仓库代码边读边看，看完你就掌握了这条链路。
>
> 行号以当前 `master` 为准；代码演进后行号会漂移，认函数名、认目录结构比认死行号可靠。

---

## 0. 先建立一张全局地图

生产知识库（入库 / ingest）的路径：

```text
 source_documents/
 ├─ diagnosis/   xxx.pdf
 ├─ device/      xxx.doc、UDI.zip(噪音)、.xml(噪音)、.html(噪音)
 └─ medication/  xxx.pdf
      │  （A）MinerU 解析           rag/mineru_service.py
      ▼
 data/parsed/mineru_partitions/<分区>/<文档名>/<文档名>.md
      │  （B）加载成 Document       rag/loader.py
      ▼
   Document(page_content, metadata)
      │  （C）结构感知切片          rag/splitter.py
      ▼
   chunk 列表（每个 chunk 自己带全量 metadata + 上下文）
      │  （D）向量化                rag/embeddings.py（模型）
      ▼
   [0.12, -0.03, ...]  1024 维向量
      │  （E）写入 Milvus          rag/vectorstore.py
      ▼
 collection: medical_knowledge_base   (db: self_rag)
   ├─ partition: diagnosis     ← diagnosis 文档
   ├─ partition: device        ← device 文档
   └─ partition: medication    ← medication 文档
   每行 = { 主键 chunk_id, vector, text, source, partition_id, doc_title, ... }
   向量字段上有 AUTOINDEX + COSINE 索引
```

用户提问（检索 / query）时把上面的路反过来走，但会**先路由到分区**再缩小检索范围：

```text
 问题 → route_partition() 判定 diagnosis/device/medication/None(app/memory.py)
     → vectorstore.search(query, partition_ids=[分区])   (app/graph.py)
     → 只在这个 partition 里做 Top-K ANN 检索 → RetrievedChunk
```

两个**入口命令**，对应两种模式，用 `.env` 的 `PARTITION_CONFIG` 切换：

| 命令 | 模式 | 写入哪个 collection | 何时用 |
| --- | --- | --- | --- |
| `python -m scripts.ingest_partitioned --rebuild` | 分区主库 | `medical_knowledge_base`（三个分区） | `.env` 设了 `PARTITION_CONFIG=knowledge_partitions.json`（当前默认） |
| `python -m scripts.ingest --rebuild` | 平铺 | `RAG_DOCS`（或 `COL_NAME`） | 把 `PARTITION_CONFIG` 置空 / 设 `none` / `off` |

分区开关的真正判定点有两个，后面会反复看到：

- `rag/vectorstore.py` 用 `registry_from_config(config)` 是否为空来决定是否走分区建库/分区写入。
- `rag/partitions.py` 的 `registry_from_config()` 读的就是 `config.partition_config`，`app/config.py` 把环境变量解析成这个路径。

---

## 1. 第一站：文档解析（MinerU）

### 1.1 角色划分

仓库刻意把「解析」和「入库」拆成两步（见 `docs/MINERU.md`）：

- 解析只负责把 PDF / Office / 图片 / HTML **变成 Markdown**。
- 入库只负责把 Markdown 切片、向量化、写进 Milvus。

好处：未来换解析器（MinerU → 别的）不影响入库代码，只要它仍输出 `.md`。

### 1.2 入口

分区流程里，解析不是单独一个脚本，而是被 `scripts/ingest_partitioned.py` 按分区自动调用：

```bash
python -m scripts.ingest_partitioned --rebuild
```

它内部对 `source_documents/<分区>` 下每个子目录（`diagnosis/device/medication`）做三件事：
收集文件 → 需要的话调 MinerU 解析 → 读回 markdown。

独立可用的“只解析”脚本是 `scripts/parse_documents.py`：

```bash
python -m scripts.parse_documents .\source_documents --output-dir .\data\parsed\mineru
```

### 1.3 关键代码：`rag/mineru_service.py`

核心类 `MinerUDocumentParser`。两个静态方法最值得学：

**① 后缀白名单 —— `collect_parse_sources`（约 `rag/mineru_service.py:170`）**

```python
SUPPORTED_MINERU_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".webp", ".doc", ".docx",
    ".ppt", ".pptx", ".html", ".htm", ...
}
```

它 `rglob` 目录里所有文件，只放行后缀在白名单里的。注意 `.zip`、`.xml` **天然不在名单里**，
所以 device 目录下的 `UDI.zip`、`nmpa_udi_full_data_rss.xml` 直接就被忽略了；
但 `.html` 在名单里，所以 `scripts/ingest_partitioned.py` 还要**显式再滤掉 `.html/.htm`**（除非 `--include-html`）。
这就是“跳过噪音”的两层过滤，代码上分别在两个文件里体现：

```python
# scripts/ingest_partitioned.py
NOISE_SUFFIXES = {".zip", ".xml", ".html", ".htm"}

def _collect_partition_files(partition_dir, *, include_html):
    files = collect_parse_sources([partition_dir])   # 第一层：后缀白名单
    if include_html:
        return files
    return [f for f in files if f.suffix.lower() not in NOISE_SUFFIXES]  # 第二层：滤噪音
```

**② 远程解析与产物布局 —— `parse_files`（约 `rag/mineru_service.py:71`）**

```python
output_root = <-- 你传的 output_dir（分区模式下是 data/parsed/mineru_partitions/<分区>）
stem = _safe_stem(result.filename or source_path.name)
document_dir = _unique_output_dir(output_root, stem, index)   # output_root/<文档名>/
markdown_path = document_dir / f"{stem}.md"                    # <文档名>/<文档名>.md
result.save_markdown(str(markdown_path), with_images=with_images)
```

一个 pdf 解析后磁盘长这样（这是**中间产物**，不是最终知识）：

```text
data/parsed/mineru_partitions/diagnosis/
└─ hypertension_guideline_2025/
   ├─ hypertension_guideline_2025.md   ← 后续 loader 只吃它
   └─ images/...                        ← 图片（当前链路不使用）
```

`parse_files` 还会在 HTTP 429 时退避重试（`_extract_with_retry`），返回
`ParsedDocument(source_path, markdown_path, filename)` 列表。

> **记忆点**：一个原始 PDF → 一个 `<文档名>/<文档名>.md`。目录按文档名嵌套，
> 这就是为什么后面 loader 用 `rglob("*.md")` 能递归扫到。

---

## 2. 第二站：加载成 `Document`

### 2.1 `Document` 是什么

LangChain 的统一“带料”结构：`Document(page_content: str, metadata: dict)`。
后面整条链路（切片 → 向量 → 检索）都在这两个字段上做文章。

### 2.2 关键代码：`rag/loader.py` 的 `load_documents_from_path`

```python
SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown"}

for file_path in file_paths:          # 目录则 rglob 递归
    loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    loaded = loader.load()            # 一个文件 → 一个 Document(page_content=全文)
    for document in loaded:
        document.metadata = {**document.metadata, "source": str(file_path)}
    documents.extend(loaded)
```

只做两件朴素的事：

1. 把 `.md` 全文读成 `Document.page_content`；
2. 塞一条 `metadata["source"]`（该 md 的绝对路径），让每个切片都知道“我来自哪个文件”。

（`ingest_partitioned` 在 `_attach_partition_metadata` 里还会再补
`partition_id / doc_title / source_type`，见第 4 站。）

---

## 3. 第三站：切片（splitter）—— 链路里最有信息量的一站

### 3.1 为什么不能“按固定字符数硬切”

直接 `text[:200]` 会把一句话、一张表格、一个标题的层级关系切烂。医疗文档里表格、
多级标题很多，检索时“没有上下文的碎片”=噪音。所以仓库用的是**结构感知切分**
`HybridMarkdownSplitter`（`rag/splitter.py`）。分两步：**先分块，再切分**。

### 3.2 第一层：Markdown → “块 block”（`_parse_markdown_blocks`）

它逐行扫描 markdown，识别出不同类型，并把“当前所在的标题层级”记下来：

```python
heading → 更新 section_stack            # 维护“章节路径”
代码围栏 → block_type="code"
$$公式$$ → block_type="formula"
表格     → block_type="table"           # 见 3.3
图片     → block_type="image"
其它     → block_type="text"
```

每个块生成时自动带上**已累积的父级元数据**，这是“元数据长到每个片段”的第一跳：

```python
MarkdownBlock(metadata={**metadata, "document_index": document_index}, ...)
```

**这是整条链路的枢纽机制**，务必看透：`document.metadata → block.metadata → chunk.metadata`，
一层一层 `{**父元数据, ...}` 透传下去。所以你在**最顶层的 `Document` 上设一次**
`partition_id / doc_title / source_type`，最后**每一个 chunk 都自动携带**，不需要在切片器里改任何东西。

### 3.3 第二层：块 → “切块 chunk”

- 文本块：`_split_text_block` → 先按 `SENTENCE_RE` 切成“语义单元”（尽量不劈句子、列表项单独成单元），再按 `chunk_size` 打包 → `_pack_text_units`；超长单元 `_hard_split` 兜底。
- 表格块：`_split_table_block` → 用 `chunk_size * 2` 的上限（`self.table_chunk_size`），**保留表头**再按行分批，避免表格被切烂后表头丢失。

`chunk_size / chunk_overlap` 来自配置（`app/config.py`，默认 200 / 80）。

每产生一个 chunk，就地构造它的身份证 `chunk_id`：

```python
def _build_chunk_id(source, block_index, chunk_index, content):
    digest = sha1(f"{source}|{block_index}|{chunk_index}|{content}").hexdigest()[:10]
    return f"{source_name}:{block_index}:{chunk_index}:{digest}"
```

> **为什么 chunk_id 里有 sha1(content)？** 同源同位置，内容变了摘要就变 → id 跟着内容变化。
> 这让 id 成为“内容指纹”，是后面做**幂等 upsert** 的基础（同一片段重复写入，主键相同=覆盖而非新增）。

### 3.4 最后一层：给每个 chunk 补“邻片上下文 + 真正去算向量的文本”

`_attach_context_metadata`（`rag/splitter.py:103`）：

```python
context_before = 上一个同源 chunk 的末尾 snippet   # 长度 = chunk_overlap
context_after  = 下一个同源 chunk 的开头 snippet
chunk.metadata["embedding_text"] = _build_embedding_text(chunk)
```

而 `_build_embedding_text` 构造的字符串是**实际喂给 embedding 模型**的东西：

```text
章节路径：…
内容类型：text
前文上下文：…
正文：
<chunk.page_content>
后文上下文：…
```

> **这是本仓库最重要的一个 trick（记忆点）**：向量化时并不只用 chunk 正文，
> 而是先把“章节路径 + 上下文”拼在前面，让语义更完整；同时把这段“增强后的文本”
> 存在 `metadata["embedding_text"]`，供入库阶段读取，正文则原样保存在 `page_content`。
> 切片结束时，chunk 已经自带这一套字段：
> `chunk_id / block_type / block_index / chunk_index / chunk_count /
>  section_path / section_title / section_path_text / context_before / context_after /
>  embedding_text` + 从父级透传下来的所有 `document.metadata`（含分区字段）。

---

## 4. 第四站：向量化（Embedding）

### 4.1 模型封装：`rag/embeddings.py`

```python
def create_embedding_model(config=None):
    return DashScopeEmbeddings(
        model=config.embed_model_name,        # 例：qwen3.7-text-embedding
        dashscope_api_key=config.embed_api_key,
    )
```

只负责“造一个能调 DashScope 的 embedding 客户端”。

### 4.2 维度一致性（极易踩坑）

`config.embed_dimension`（默认 1024）**必须等于** embedding 模型实际输出的向量维数。
因为后面建 Milvus collection 时用的是 `dim=config.embed_dimension`，两者不一致会直接报错。
所以换 embedding 模型时，一定要同步改 `.env` 里的 `EMBED_DIMENSION`。

### 4.3 在哪真正调用、怎么批量（`rag/vectorstore.py:_embed_documents`）

```python
def _embed_documents(self, texts):
    vectors = []
    for batch in _batched(texts, self.config.embed_batch_size):  # 每批 20 条
        vectors.extend(self.embedding_model.embed_documents(batch))
    return vectors
```

对外接口再包一层“取哪段文本去向量化”：`_embedding_text(chunk)` 优先用
`metadata["embedding_text"]`，没有才退回 `page_content` —— 正是第 3.4 节埋的伏笔。

---

## 5. 第五站：写入 Milvus —— 分区与索引的核心战场

所有 Milvus 操作都收口在 `rag/vectorstore.py` 的 `MilvusVectorStore`。
请把这一节和第 6、7 节对照着读，这是你要的“分区/索引在代码上如何体现”的答案所在。

### 5.1 连接与“逻辑库”

```python
def _use_database(self):                 # rag/vectorstore.py:46
    databases = self.client.list_databases()
    if self.config.db_name not in databases:   # db_name = self_rag（来自 .env）
        self.client.create_database(db_name=self.config.db_name)
    self.client.use_database(db_name=self.config.db_name)
```

Milvus 允许在一个服务里建多个**逻辑数据库**做隔离；本仓库所有知识都放在 `self_rag` 库。
动手排查时千万别忘了先 `use_database('self_rag')`，否则会查到一个“空”的 default 库。

### 5.2 collection 名字从哪来 —— “分区唯一事实源”

构造时（`rag/vectorstore.py:28`）：

```python
self.registry = registry_from_config(self.config)          # 分区模式才非 None
self.collection_name = (
    self.registry.collection_name if self.registry else self.config.collection_name
)   # 分区模式 = medical_knowledge_base；平铺 = RAG_DOCS
```

**一个易错点（代码注释也标了）**：分区开启时，如果 `has_collection / upsert / search`
各自盯着不同的 collection 名（比如 config 里还留着 `COL_NAME=RAG_DOCS`）就会自相矛盾。
所以类里只认 `self.collection_name` 一个值，统一从 registry 取。

registry（`rag/partitions.py`）读的是仓库根目录的 `knowledge_partitions.json`，它才是
“有哪些分区、各自路由关键词、每个分区 metadata 模板”的唯一事实源：

```json
{
  "milvus_collection_name": "medical_knowledge_base",
  "default_partition": "diagnosis",
  "partitions": [
    { "id": "diagnosis",  "route_keywords": ["诊断","症状","指南", ...],
      "metadata_template": { "partition_id":"diagnosis", "source_type":"clinical_guideline", ... } },
    { "id": "device", ... }, { "id": "medication", ... }
  ]
}
```

`PartitionRegistry` 提供：

- `partition_ids`（按固定优先级排序：device → medication → diagnosis，`rag/partitions.py:58`）；
- `route(question)` 关键词路由（`:76`）；
- `label(partition_id)` 中文名（诊断=诊疗问题、device=设备问题、medication=用药问题）；
- `source_type(partition_id)`。

### 5.3 建 collection：平铺 / 分区两条分支

`ensure_collection(rebuild)` 根据 registry 是否存在，走 `create_collection`（平铺）或
`_ensure_partitioned_collection`（分区）。

分区分支是“分区 + 索引”的**代码正主**（`rag/vectorstore.py:77`），逐步拆解：

```python
# 1) 定义 Schema：显式声明“主键 + 向量”，其余字段全部走动态字段
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

# 2) 主键：跨分区唯一的稳定字符串
schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR,
                 is_primary=True, max_length=1024)

# 3) 向量字段：FLOAT_VECTOR，维数必须 = embedding 输出维数
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                 dim=self.config.embed_dimension)

# 4) 索引：向量字段 AUTOINDEX + COSINE
index_params = self.client.prepare_index_params()
index_params.add_index(field_name="vector",
                       index_type="AUTOINDEX", metric_type="COSINE")

self.client.create_collection(collection_name=..., schema=schema, index_params=index_params)

# 5) 建三个业务分区
for partition_id in self.registry.partition_ids:
    if not self.client.has_partition(collection_name=..., partition_name=partition_id):
        self.client.create_partition(collection_name=..., partition_name=partition_id)
```

对照理解四个设计点：

**主键 `chunk_id` VARCHAR(1024)，而不是平铺模式的整型 `id`。** 因为要支持
“跨分区唯一 + 幂等 upsert”。chunk_id 里可能含中文文档名，设 1024 是给中文多字节字符串留余量。

**为什么要 `enable_dynamic_field=True`。** 那 `text / source / partition_id / doc_title /
section_path_text / context_before / context_after …` 一大堆字段，都不需要预先声明在 schema 里，
Milvus 会把它们收进动态字段。这让 schema 极简：只要主键 + 向量两个显式字段。

**为什么只给 `vector` 建索引，不给 `partition_id` 建索引。** 向量检索需要的 ANN 索引
只作用于 `vector`。分区模式下 `partition_id` 是**动态字段**，给动态字段建标量 AUTOINDEX
会报错（`PARTITION_GUIDE.md` 里早年的设计草稿犯过这个错，正式代码已改正——这也是
那份文档在第九节特别标注“以代码为准”的原因）。

**`auto_id=False`。** 主键由我们（`partition_id:chunk_id`）生成而不是 Milvus 自增，
这是 upsert 幂等的前提。

### 5.4 写入：`_upsert_partitioned` —— 分区体现得最直白的地方

`upsert_documents` 拿到 `Document` 列表后：取 `_embedding_text` → 批量向量化 → 按分区分发。
分区写入分支 `_upsert_partitioned`（`rag/vectorstore.py:156`）：

```python
for chunk, vector in zip(chunks, vectors, strict=True):
    partition_id = chunk.metadata["partition_id"]        # 每个 chunk 都带着（第 3 节机制）
    ...
    row = {
        "chunk_id": f"{partition_id}:{chunk_id}",        # 跨分区全局唯一主键！
        "vector": vector,
        "text": chunk.page_content,
        "source": ...,
        "partition_id": partition_id,
    }
    row.update(_metadata_for_milvus(chunk.metadata))      # 挑选字段写入
    groups.setdefault(partition_id, []).append(row)

for partition_id, rows in groups.items():
    self.client.upsert(collection_name=..., data=rows, partition_name=partition_id)
    self.client.flush(collection_name=...)
```

逐条解释：

1. **主键 = `f"{partition_id}:{chunk_id}"`**。因为 Milvus 主键是**跨 partition 全局唯一**的；
   两个分区里同源同内容的 chunk 若不区分会撞主键互相覆盖。加分区前缀后，同一个文档重复
   导入（甚至跨分区搬移）都能靠同主键稳定 upsert——内容变了主键也变（sha1），于是新增一条；
   内容没变就原地覆盖，不会越导越多。
2. **`partition_name=partition_id`**：这就是“把这一批数据写进对应分区”的 API 本体。
   分组后再按分区批量 upsert，既收窄又是分区级幂等。
3. **未知 / 缺失 partition_id 直接抛错**，绝不静默塞进 diagnosis（避免把错误归类悄悄盖过）。
4. **`_metadata_for_milvus`** 只挑选 `METADATA_OUTPUT_FIELDS` 里的字段写出
   （`block_type/…/context_after/partition_id/doc_title` 等），列表会拼成 `a > b > c` 字符串。
   这些就是将来检索时能随命中文档一起返回的“旁证信息”。
5. `_metadata_for_milvus` 里并没有过滤 `vector`/`embedding_text`（embedding_text 体积大、又不回查），
   这正好说明：**写进 collection 的字段 = 检索要展示的字段**，两者要刻意对齐。

> 对比：平铺模式 `upsert_documents` 用 `{"id": index, "vector", "text", "chunk_id", "source"}`
> 写入，没有任何分区概念 —— 这就是“关掉 `PARTITION_CONFIG` 就完全回到旧行为”的原因。

### 5.5 “索引”汇总：索引到底建在哪、长什么样

| 对象 | 在哪一行体现 | 内容 |
| --- | --- | --- |
| 向量索引 | `rag/vectorstore.py:99` `index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")` | 在 collection 创建时随 schema 一起建好 |
| 索引类型 | AUTOINDEX | 不手填 HNSW/IVF 参数，交给 Milvus 自动挑选索引实现，最省事 |
| 距离度量 | COSINE | 相似度口径 = 余弦相似度；检索结果的距离字段（`hit["distance"]`）据此给出 |
| 单向量自动探测 | 无需写 `anns_field` | pymilvus 自动以唯一的 FLOAT_VECTOR 字段为向量字段做检索；若以后加第二个向量字段则必须显式指定 `anns_field` |
| 主键 | schema `is_primary=True` | 主键字段 Milvus 自带索引，无需我们显式 add_index |
| 动态字段 | `enable_dynamic_field=True` | 不建标量索引，检索按输出字段名取回即可 |

### 5.6 一行数据最终长什么样（写进 Milvus 之后）

对 schema 字段 + 动态字段一起看（示意）：

```jsonc
{
  "chunk_id": "diagnosis:hypertension_guideline_2025:12:3:9f8e2c1a07",  // 主键（schema 字段）
  "vector":   [0.12, -0.03, /* ...1024 维 */],                            // schema 字段，有索引
  "text":     "...正文切片...",                    // 动态
  "source":   "D:/.../hypertension_guideline_2025.md", // 动态
  "partition_id": "diagnosis",                     // 动态
  "doc_title": "hypertension_guideline_2025",      // 动态
  "section_title": "诊断", "context_before": "...", "context_after": "...",  // 动态
  "block_type": "text", "block_index": 12, "chunk_index": 3, "chunk_count": 20
}
```

---

## 6. 索引专题：向量索引到底在解决什么问题

1. **没有索引会怎样**：把 query 向量和库里**每一个**向量做内积找最近邻 = 全量扫描，
   几万行勉强、几百万行就不可用。向量索引 = 用近似最近邻（ANN）换速度，允许很小的召回损失。
2. **`AUTOINDEX`**：Milvus 提供一个“自动索引”，你不需要懂 HNSW 的 `M/efConstruction` 或
   IVF 的 `nlist`，创建 collection 时声明一次即可。适合入门与验证；追求极致召回/延迟时
   再改成手写参数的 `HNSW` 等索引类型（这属于调优，不影响本链路结构）。
3. **`COSINE`**：相似度的“尺子”。写入和查询必须用同一个 metric，否则分数没有可比性。
   仓库检索时把 Milvus 返回的 `distance` 当 score 用（`app/schemas.py` 的
   `RetrievedChunk.from_milvus_hit`），并以此排序、喂给后面的相关性打分。
4. **索引在代码里的“生活范围”**：它只活在 collection 的 schema 上，随 collection 创建/重建。
   也就是说：改 `chunk_size`、换 embedding 模型、改 metadata 字段，**都不需要动索引**；
   只有当你改了“向量字段本身”（比如换更粗/更细的字段、加第二个向量字段）才需要重建索引
   —— 重建 = `--rebuild`（drop + recreate collection）。

---

## 7. 检索：把链路反着走，并先路由到分区

### 7.1 路由：问题属于哪个分区（`app/memory.py:216`）

```python
def route_partition(question: str) -> str | None:
    registry = default_registry()          # 读 knowledge_partitions.json
    if registry is not None:
        return registry.route(question)    # 关键词路由（有优先级）
    # 无分区配置时回退到旧的关键词表 PARTITION_KEYWORDS
```

`PartitionRegistry.route`（`rag/partitions.py:76`）按固定优先级
device → medication → diagnosis 依次做关键词命中，返回第一个命中的分区；一个都不命中 → `None`
（表示“不限定分区”，见 7.2 的分支）。

### 7.2 检索节点如何决定查哪个分区（`app/graph.py` 的 `retrieve_node`，第 87 行起）

```python
retry_count = state.get("retry_count", 0)
partition_ids = (
    [state["partition_id"]]
    if (state.get("partition_id") and retry_count == 0)   # 首次检索：收窄到路由分区
    else None                                             # rewrite 重试：放开到全分区
)
documents = vectorstore.search(retrieval_query, top_k=top_k, partition_ids=partition_ids)
```

- `partition_id` 在 `memory_node`（`prepare_memory_context`）里已算好并放进 state。
- 首次检索在**单分区**做 Top-K → 相关度不够 → 重写查询重试时改成**全分区** Top-K，
  避免“高血压+用药”这类跨领域问题困死在第一个分区里。
- 路由结果为 `None`（没命中任何分区）时同样走全分区 → 兜底“全局召回”。

### 7.3 `vectorstore.search` 如何把“分区过滤”翻译成 Milvus 参数

（`rag/vectorstore.py:201`）

```python
search_kwargs = {}
if self.registry is not None and partition_ids:
    search_kwargs["partition_names"] = [
        pid for pid in partition_ids
        if self.registry.partition(pid) is not None   # 只留注册表里真实存在的分区
    ]

result_sets = self.client.search(
    collection_name=self.collection_name,
    data=[query_vector],          # query 先 embed_query 成向量
    limit=limit,
    output_fields=["text", "chunk_id", "source", *METADATA_OUTPUT_FIELDS],  # 想带回来的字段
    **search_kwargs,              # partition_names 若存在 → 检索被收窄到这些分区
)
```

- 平面模式（registry 为 None）时 `partition_ids` 被**忽略**，检索全 collection ——
  保持旧行为不变。
- 检索结果的原始 dict 再经 `app/schemas.py` 的 `from_milvus_hit` 清洗成 `RetrievedChunk`
  （统一读 `entity` 或顶层字段、取 `distance` 当 score、过滤掉 `id/vector/text/chunk_id/source`
  后把剩余字段塞进 `metadata`），上层（grade/generate）拿到的都是干净的 `RetrievedChunk`。

---

## 8. 一图流回顾“分区与索引在你仓库里写在哪”

| 你要找的东西 | 文件 | 函数 / 位置 |
| --- | --- | --- |
| 分区配置（3 个分区的定义、路由词、模板） | `knowledge_partitions.json` | 整个文件 |
| 分区的“唯一事实源”加载 | `rag/partitions.py` | `registry_from_config` / `default_registry` / `PartitionRegistry` |
| 分区开关的判定 | `rag/vectorstore.py` | `self.registry`（构造时） |
| 建分区 collection + 向量索引 | `rag/vectorstore.py` | `_ensure_partitioned_collection`（schema + index_params + create_partition） |
| 按分区写入 | `rag/vectorstore.py` | `_upsert_partitioned`（`partition_name=` + 分区前缀主键） |
| 检索按分区收窄 | `rag/vectorstore.py` | `search(partition_names=...)` |
| 首次分区检索 / 重试全库 | `app/graph.py` | `retrieve_node` |
| 问题 → 分区 的路由 | `app/memory.py` | `route_partition`（registry.route） |
| 向量化调用 + 批次 | `rag/vectorstore.py` | `_embed_documents`；模型在 `rag/embeddings.py` |
| 切片 + 上下文 + embedding_text | `rag/splitter.py` | `HybridMarkdownSplitter` / `_build_chunk_id` / `_build_embedding_text` |
| 文档解析 | `rag/mineru_service.py` | `collect_parse_sources` / `parse_files` |

---

## 9. 动手实验（照着做一遍才算真正掌握）

1. **先看“切片产物”**：临时写几行，把某个 chunk 的 metadata 打出来，观察
   `partition_id / doc_title / section_path_text / context_before / chunk_id` 是否都齐了、
   父级 metadata 是否真的透传到了每个 chunk。这是理解第 3 节的钥匙。

2. **看 schema 与分区**（先起 Milvus）：

   ```python
   from pymilvus import MilvusClient
   c = MilvusClient("http://localhost:19530"); c.use_database("self_rag")
   c.describe_collection("medical_knowledge_base")   # 看字段：主键 chunk_id + vector
   c.list_partitions("medical_knowledge_base")       # _default + diagnosis/device/medication
   ```

3. **改参数观察影响**：把 `.env` 的 `RAG_CHUNK_SIZE` 从 200 改成 80，重跑
   `python -m scripts.ingest_partitioned --rebuild`（会重新 MinerU 解析，或先备好 md 用
   `--no-parse`），观察 `chunks` 计数如何变化；再对比同一个问题检索命中的上下文长短。

4. **关掉分区看回落**：把 `.env` 的 `PARTITION_CONFIG` 设为 `none`，跑
   `python -m scripts.ingest --rebuild` → `describe_collection("RAG_DOCS")` 没有分区概念，
   主键是整型 `id`。再改回来，确认 `_ensure_partitioned_collection` 两条分支的行为差异。

5. **验证“跳过噪音”**：`python -m scripts.ingest_partitioned` 正常跑 device 时打印
   `source_files`，应为 1（只有那个 `.doc`）；加 `--include-html` 会变成 2。

6. **挑战题**：自己解释 —— 为什么一个文档重复导入两次（不 rebuild）数据不会翻倍？
   提示：主键 `partition_id:chunk_id` 的内容指纹特性 + Milvus upsert 语义。
