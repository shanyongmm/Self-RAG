# MinerU Document Parsing

这个项目把文档处理拆成两步：

1. `scripts.parse_documents` 只负责调用 MinerU，把 PDF、图片、Office 或 HTML 文件解析成 Markdown。
2. `scripts.ingest` 只负责读取 Markdown/Text，切片、Embedding，并写入 Milvus。

## 配置 Token

在 `.env` 中加入官网申请到的 Token：

```env
MINERU_TOKEN=your_mineru_token
MINERU_OUTPUT_DIR=data/parsed/mineru
```

不要把真实 Token 写进 Python 源码。

## 解析文档

```powershell
python -m scripts.parse_documents .\data --output-dir .\data\parsed\mineru
```

也可以指定单个文件，并额外保存 HTML 或 DOCX：

```powershell
python -m scripts.parse_documents `
  .\data\example.pdf `
  --output-dir .\data\parsed\mineru `
  --extra-format html `
  --extra-format docx
```

解析完成后，每个文档会生成一个独立目录，里面至少包含一个 `.md` 文件。

## 切片入库

把解析后的 Markdown 目录写入 Milvus：

```powershell
python -m scripts.ingest --source .\data\parsed\mineru --rebuild
```

如果不想重建 collection，去掉 `--rebuild`。

## 推荐流程

```powershell
python -m scripts.parse_documents .\data --output-dir .\data\parsed\mineru
python -m scripts.ingest --source .\data\parsed\mineru --rebuild
```

这样解析和入库互不耦合。以后如果换解析器，只要继续输出 Markdown，入库流程无需改动。
