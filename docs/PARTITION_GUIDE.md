# Self-RAG 医疗知识库分区说明

当前先做三类分区：医疗诊断、医疗设备、药品用药。按照 Milvus 的做法，这里建议使用 **一个 Collection + 三个 Partition**：所有医疗知识放在同一个 `medical_knowledge_base` collection 里，再通过 `diagnosis`、`device`、`medication` 三个 partition 隔离不同领域的文档。

后续你拆文档并写入数据库时，只要把 chunk 插入对应 partition，并在 metadata 里保留 `partition_id`，检索时就能按分区缩小范围，减少无关内容干扰。

## 1. 三个分区怎么理解

| partition_id | 分区名 | 放什么内容 | Milvus 建议位置 |
| --- | --- | --- | --- |
| `diagnosis` | 医疗诊断分区 | 疾病诊断、治疗原则、临床指南、筛查、转诊、慢病管理 | `medical_knowledge_base` / `diagnosis` partition |
| `device` | 医疗设备分区 | 医疗器械、医用设备、UDI、设备分类、注册审评、操作/监管资料 | `medical_knowledge_base` / `device` partition |
| `medication` | 药品用药分区 | 药品目录、基本药物、剂型规格、处方、禁忌、不良反应、合理用药 | `medical_knowledge_base` / `medication` partition |

检验指标如果是在讲“如何诊断疾病”，先放 `diagnosis`；如果是在讲“检验设备/仪器”，放 `device`；如果是在讲“某药物监测或用药调整”，放 `medication`。

## 2. 推荐下载资料

### 医疗诊断分区

- 国家基层高血压防治管理指南2025版  
  下载页：https://hbp-office.nccd.org.cn/download.html  
  PDF直链：https://www.hnysfww.com/data/article/1758327573527813381.pdf

- 国家基层糖尿病防治管理指南（2022）  
  下载页：https://www.jctnb.org.cn/home/GuideDownload/index  
  PDF直链：https://www.hnysfww.com/data/article/1647557424677414104.pdf

### 医疗设备分区

- 国家药监局医疗器械唯一标识数据库  
  下载页：https://udi.nmpa.gov.cn/download.html  
  建议优先下载“全量发布数据”，后面做增量更新时再下载每日/每周/月度数据。

- 医疗器械分类目录动态调整工作程序（2026年第53号公告附件）  
  原文页：https://www.nmpa.gov.cn/xxgk/ggtg/ylqxggtg/ylqxqtggtg/20260601173941178.html  
  DOC直链：https://yjj.sh.gov.cn/cmsres/bb/bb5a3c9a7405472ab06dd0a97dd73530/35a7fd076e5c2be536b12a18019c9fb7.doc

### 药品用药分区

- 国家基本药物目录（2026年版）  
  原文页：https://www.nhc.gov.cn/yaozs/c100098/202607/0f4f2bdcfead449f8412453373522470.shtml  
  PDF直链：https://www.nhc.gov.cn/yaozs/c100098/202607/0f4f2bdcfead449f8412453373522470/files/%E5%9B%BD%E5%AE%B6%E5%9F%BA%E6%9C%AC%E8%8D%AF%E7%89%A9%E7%9B%AE%E5%BD%95%EF%BC%882026%E5%B9%B4%E7%89%88%EF%BC%89.pdf

- 国家基本药物目录（2018年版）  
  原文页：https://www.nhc.gov.cn/wjw/jbywml/201810/8b68d28bd3754898b339e06da8c7d907.shtml  
  PDF直链：http://www.nhc.gov.cn/ewebeditor/uploadfile/2018/10/20181025183346942.pdf

## 3. 创建分区的核心步骤

1. 创建一个 Milvus collection：`medical_knowledge_base`。
2. 在这个 collection 下创建三个 partition：`diagnosis`、`device`、`medication`。
3. 拆文档时判断每个 chunk 属于哪个分区。
4. 写入时使用 Milvus 的 `partition_name` 参数，把 chunk 插入对应 partition。
5. 每个 chunk 的字段或动态 metadata 至少保存：`partition_id`、`doc_title`、`source_url`、`source_type`、`section_title`。
6. 查询时先判断问题属于哪个分区，再用 Milvus 的 `partition_names` 参数只检索对应 partition。
7. 如果问题跨领域，例如“高血压患者能否使用某药”，先查 `diagnosis`，再查 `medication`。

## 4. Milvus 建库与创建分区

下面示例基于 Milvus 官方 `MilvusClient` 写法。`EMBEDDING_DIM` 必须和你实际使用的 embedding 模型维度一致，例如 768、1024 或 1536。

```python
from pymilvus import MilvusClient, DataType

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "medical_knowledge_base"
EMBEDDING_DIM = 768

client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
)

# 1. 创建 Collection Schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(
    field_name="id",
    datatype=DataType.VARCHAR,
    is_primary=True,
    max_length=128,
)
schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=EMBEDDING_DIM,
)
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
schema.add_field(field_name="partition_id", datatype=DataType.VARCHAR, max_length=32)
schema.add_field(field_name="doc_title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=64)
schema.add_field(field_name="section_title", datatype=DataType.VARCHAR, max_length=512)

# 2. 创建索引。向量字段建议先用 AUTOINDEX + COSINE。
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)
index_params.add_index(
    field_name="partition_id",
    index_type="AUTOINDEX",
)

# 3. 创建 Collection
if not client.has_collection(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

# 4. 创建三个业务 Partition
for partition_name in ["diagnosis", "device", "medication"]:
    if not client.has_partition(
        collection_name=COLLECTION_NAME,
        partition_name=partition_name,
    ):
        client.create_partition(
            collection_name=COLLECTION_NAME,
            partition_name=partition_name,
        )

print(client.list_partitions(collection_name=COLLECTION_NAME))
```

创建完成后，`medical_knowledge_base` 下面会有 Milvus 默认的 `_default` partition，以及我们手动创建的三个业务 partition。后续业务数据建议写入这三个业务 partition，不要写入 `_default`。

## 5. Milvus 写入示例

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

chunk = {
    "id": "diagnosis_hbp_2025_0001",
    "embedding": [0.01] * 768,
    "content": "这里放拆分后的文档片段",
    "partition_id": "diagnosis",
    "doc_title": "国家基层高血压防治管理指南2025版",
    "source_url": "https://hbp-office.nccd.org.cn/download.html",
    "source_type": "clinical_guideline",
    "section_title": "诊断",
}

client.insert(
    collection_name="medical_knowledge_base",
    partition_name="diagnosis",
    data=[chunk],
)
```

写入医疗设备文档时，把 `partition_name` 和 `partition_id` 都改成 `device`；写入药品文档时都改成 `medication`。

## 6. Milvus 分区检索示例

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

question = "糖尿病怎么诊断？"
query_embedding = [0.02] * 768
partition_id = "diagnosis"

results = client.search(
    collection_name="medical_knowledge_base",
    data=[query_embedding],
    anns_field="embedding",
    partition_names=[partition_id],
    limit=5,
    output_fields=[
        "content",
        "partition_id",
        "doc_title",
        "source_url",
        "source_type",
        "section_title",
    ],
)

print(results)
```

跨分区问题可以查多个 partition：

```python
results = client.search(
    collection_name="medical_knowledge_base",
    data=[query_embedding],
    anns_field="embedding",
    partition_names=["diagnosis", "medication"],
    limit=8,
    output_fields=["content", "partition_id", "doc_title", "source_url"],
)
```

## 7. 简单路由规则

刚开始不用上复杂模型，先用关键词路由：

```python
def route_partition(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["设备", "器械", "udi", "监护仪", "影像", "ct", "dr"]):
        return "device"
    if any(k in q for k in ["药", "用药", "剂型", "规格", "处方", "不良反应", "禁忌"]):
        return "medication"
    return "diagnosis"
```

这个版本足够支撑第一阶段优化。后面如果分区多了，再把路由升级成 LLM 分类或多路召回。

## 8. 参考 Milvus 官方文档

- 创建 Collection：https://milvus.io/docs/zh/create-collection.md
- 管理 Partition：https://milvus.io/docs/manage-partitions.md
- 插入实体到 Partition：https://milvus.io/docs/insert-update-delete.md
- 按 Partition 检索：https://milvus.io/docs/single-vector-search.md

## 9. 落地实现（本文档为设计稿，实际以代码为准）

上面的示例代码是设计时的草稿，正式实现已经落地为以下文件，以代码为准：

- `knowledge_partitions.json` —— 分区配置的唯一事实源（collection 名、三个分区的
  id / 路由关键词 / metadata 模板）。
- `rag/partitions.py` —— 读取配置并构建 `PartitionRegistry`，提供
  `route(question)`（优先级 device → medication → diagnosis）、label、source_type。
- `rag/vectorstore.py` —— 分区开关：`PARTITION_CONFIG` 开启后统一操作
  `medical_knowledge_base`，按 `partition_id` 分组建 partition 写入；查询支持
  `partition_ids → partition_names` 收窄检索。
- `scripts/ingest_partitioned.py` —— 新主流程：解析 `source_documents/<分区>` 下的
  文档（默认跳过 `.zip/.xml/.html` 噪音），按分区写入对应 partition。
- `app/memory.py` / `app/graph.py` —— 问题先路由到分区；首次检索只在该分区内做
  Top-K，路由未命中或重试时放开到全分区召回。

需要注意两点与上面草稿不同的实现细节：

1. 分区主键不是示例里的 `id`，而是 `f"{partition_id}:{chunk_id}"` 这种跨分区唯一的
   稳定字符串（VARCHAR，`max_length=1024`），否则跨分区会同 chunk 撞主键。
2. `partition_id` 等是 collection 的**动态字段**，不要再按草稿给标量 `partition_id`
   建 AUTOINDEX（分区模式下那样建会报错）。

