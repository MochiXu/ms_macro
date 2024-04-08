## 说明

ms_macro 数据集下载
```bash
cd dataset_files
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -zxvf collectionandqueries.tar.gz
```

我们将会使用这些文件进行数据集生成
```bash
collection.tsv  原始入库文件, 表示 Bing Answers passages
qrels.dev.small.tsv dev-query, 表示 Bing 的 question-id answer-id 对应关系
queries.dev.small.tsv dev-query, 表示 Bing 的 question-id 对应的 question-text
```