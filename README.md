## 说明

### ms_macro hdf5 数据集制作流程

ms_macro 数据集下载

参考文章 [experiments-msmarco-passage](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)

```bash
cd dataset_files
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -zxvf collectionandqueries.tar.gz
```

`collectionandqueries.tar.gz` MD5 checksum 应该是 `31644046b18952c1386cd4564ba2ae69`

我们将会使用解压后的 `collectionandqueries.tar.gz` 进行数据集生成
```bash
collection.tsv  原始入库文件, 表示 Bing Answers passages
qrels.dev.small.tsv dev-query, 表示 Bing 的 question-id answer-id 对应关系
queries.dev.small.tsv dev-query, 表示 Bing 的 question-id 对应的 question-text
```

通过 `generate_test_dev.py` 生成 `ms-macro2-768-full-cosine-dev-query.hdf5` 数据集。
通过 `geerate_train.py` 生成 `ms-macro2-768-full-cosine.hdf5` 数据集。

### ms_macro hdf5 数据集结构介绍

可以通过 `hdf5_info.py` 文件来获得 hdf5 文件详细信息

#### 入库部分

```bash
HDF5 INFO ----- ms-macro2-768-full-cosine.hdf5
attrs keys:<KeysViewHDF5 ['extra_columns', 'extra_columns_type']>
datasets:<KeysViewHDF5 ['text', 'train']>
key:text	size:8841823	dtype:object	shape:(8841823,)
key:train	size:8841823	dtype:float64	shape:(8841823, 768)
```

#### Query 部分
```bash
HDF5 INFO ----- ms-macro2-768-full-cosine-dev-query.hdf5
attrs keys:<KeysViewHDF5 ['query_columns_in_hdf5', 'query_columns_in_table', 'query_columns_type']>
datasets:<KeysViewHDF5 ['distances', 'neighbors', 'query_text', 'test']>
key:distances	size:6980	dtype:object	shape:(6980,)
key:neighbors	size:6980	dtype:object	shape:(6980,)
key:query_text	size:6980	dtype:object	shape:(6980,)
key:test	size:6980	dtype:float64	shape:(6980, 768)
```

### ms_macro hdf5 数据集下载地址

[ms-macro2-768-full-cosine.hdf5 54GB](https://mqdb-release-1253802058.cos.ap-beijing.myqcloud.com/datasets/ms-macro2-768-full-cosine.hdf5)

[ms-macro2-768-full-cosine-dev-query.hdf5 42MB](https://mqdb-release-1253802058.cos.ap-beijing.myqcloud.com/datasets/ms-macro2-768-full-cosine-dev-query.hdf5)

