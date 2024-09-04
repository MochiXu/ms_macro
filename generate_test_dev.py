import concurrent.futures
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import h5py
from transformers import AutoModelForMaskedLM, AutoTokenizer

from common.config import sparse_vector_model_id
from common.utils import convert_bytes_to_str, texts_to_embeddings, gpu_compute


def get_query_dev_texts_and_vectors(
        query_dev_file_path: str,
        test_dev_file_path: str,
        batch_size: int,
        cuda_count: int,
        exist_answer_ids: List[int]
) -> Tuple[List[str], List[List[float]], List[List[int]], List[List[float]], List[List[int]], List[List[float]]]:
    df_queries = pd.read_csv(query_dev_file_path, sep='\t', header=None, names=['query-id', 'query-text'], )
    df_test = pd.read_csv(test_dev_file_path, sep='\t', header=None, names=['query-id', 'c1', 'answer-id', 'score'])

    queries_dict = dict(zip(df_queries['query-id'], df_queries['query-text']))

    # 基于 query-id 分组
    groups = df_test.groupby('query-id')

    neighbors_list = []
    distances_list = []
    query_ids = []
    for query_id, group in groups:
        # 根据 score 降序排序
        answer_id_and_score = sorted(zip(group['answer-id'], group['score']), key=lambda x: x[1], reverse=True)
        # 过滤掉 answer 不存在的结果
        answer_id_and_score_filtered = [(answer_id, score) for answer_id, score in answer_id_and_score if
                                        answer_id in exist_answer_ids]
        if len(answer_id_and_score_filtered) == 0:
            continue
        # 避免 queries 文件不存在该 query_id 对应的 text 文本
        if query_id not in set(queries_dict.keys()):
            print(f"query_id:{query_id} can't find query_text in queries file, skipped.")
            continue
        answer_ids, scores = zip(*answer_id_and_score_filtered)
        query_ids.append(query_id)
        neighbors_list.append(answer_ids)
        distances_list.append(scores)

    # 获取 query_ids 对应的 query_texts 文本数据
    print("Convert bytes to str from queries_dict")
    query_texts = [convert_bytes_to_str(raw_str) for raw_str in tqdm.tqdm([queries_dict[qid] for qid in query_ids])]
    print(f"""query_ids size:{len(query_ids)}
              query_texts size:{len(query_texts)}
              neighbors (candidates id) size:{len(neighbors_list)}
              scores (candidates score) size: {len(distances_list)}""")

    total_batches = (len(query_texts) + batch_size - 1) // batch_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 对 texts 文本数据分批
        text_batches = [query_texts[k:k + batch_size] for k in range(0, len(query_texts), batch_size)]
        # 每个 texts batch 将会在一个 GPU 上执行 embedding
        batch_in_text_model = [text_models[k % cuda_count] for k in range(len(text_batches))]
        # 用来生成 sparse vector 的 model
        batch_in_sparse_model = [sparse_models[i % cuda_count].module for i in range(len(text_batches))]
        # 用来生成 sparse vector 的 tokenizer
        batch_in_sparse_tokenizer = [sparse_tokenizers[i % cuda_count] for i in range(len(text_batches))]

        gpu_compute_results = list(
            tqdm.tqdm(
                executor.map(
                    gpu_compute,
                    text_batches,
                    batch_in_text_model,
                    batch_in_sparse_model,
                    batch_in_sparse_tokenizer
                ),
                total=total_batches,
                desc="Processing query(mc_macro question)"
            ))
        # 合并处理后的向量列表
        text_vectors = []
        sparse_vectors_dim_ids = []
        sparse_vectors_weights = []
        for batch in gpu_compute_results:
            text_vectors.extend(batch[0])
            sparse_vectors_dim_ids.extend(batch[1])
            sparse_vectors_weights.extend(batch[2])
    return query_texts, text_vectors, sparse_vectors_dim_ids, sparse_vectors_weights, neighbors_list, distances_list


if __name__ == '__main__':
    texts_batch_size = 256
    num_threads = 4
    limits = 10000000
    store_json = True
    store_hdf5 = False
    dataset_file_prefix = "/mnt/workspaces/mochix/datasets/ms_macro2"
    # dataset_file_prefix = "dataset_files"
    skip_gpus = [2]
    gpu_count = torch.cuda.device_count()
    gpu_devices = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in range(gpu_count) if
                   i not in skip_gpus]
    # 初始化 model, model 详细信息参考 hugging face:
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    text_models = [torch.nn.DataParallel(
        SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    ) for _ in range(0, len(gpu_devices))]

    # 生成 sparse vector 的模型
    sparse_models = [torch.nn.DataParallel(
        AutoModelForMaskedLM.from_pretrained(
            sparse_vector_model_id,
        )
    ) for _ in gpu_devices]

    # sparse vector 使用的 tokenizers
    sparse_tokenizers = [
        AutoTokenizer.from_pretrained(
            sparse_vector_model_id,
            device=gpu_device
        )
        for gpu_device in gpu_devices
    ]

    # move model to gpu
    for i in range(0, gpu_count):
        text_models[i].to(gpu_devices[i])
        sparse_models[i].to(gpu_devices[i])

    df_train = pd.read_csv(
        f'{dataset_file_prefix}/collection.tsv',
        sep='\t', header=None, names=['answer-id', 'answer-text'], nrows=limits
    )

    query_texts, query_text_vectors, query_text_dim_ids, query_text_weights, neighbors, distances = (
        get_query_dev_texts_and_vectors(
            query_dev_file_path=f'{dataset_file_prefix}/queries.dev.small.tsv',
            test_dev_file_path=f'{dataset_file_prefix}/qrels.dev.small.tsv',
            batch_size=texts_batch_size,
            cuda_count=gpu_count,
            exist_answer_ids=df_train['answer-id'].tolist()
        ))

    if store_json:
        data = [
            {
                "id": id_,
                "text": text,
                "dim_ids": dim_ids,
                "weights": weights,
                "neighbors": neighbors,
                "distances": distances
            }
            for id_, (text, dim_ids, weights, neighbors, distances) in
            enumerate(zip(query_texts, query_text_dim_ids, query_text_weights, neighbors, distances))
        ]
        with open(f"{dataset_file_prefix}/ms-macro-sparse-dev-query.json", "w") as f:
            json.dump(data, f)
    if store_hdf5:
        with h5py.File(f'{dataset_file_prefix}/ms-macro-sparse-768-full-cosine-dev-query.hdf5', 'w') as query_dev_hdf5:
            query_dev_hdf5.create_dataset('query_text', data=query_texts)
            query_dev_hdf5.create_dataset('test', data=query_text_vectors)
            dt = h5py.special_dtype(vlen=np.dtype('int32'))  # 存储变长数组
            neighbors_dataset = query_dev_hdf5.create_dataset('neighbors', (len(neighbors),), dtype=dt)
            neighbors_dataset[:] = neighbors
            distances_dataset = query_dev_hdf5.create_dataset('distances', (len(distances),), dtype=dt)
            distances_dataset[:] = distances

            # 创建 int32 变长数组的特殊数据类型
            dt_uint32 = h5py.special_dtype(vlen=np.dtype('uint32'))
            dt_float32 = h5py.special_dtype(vlen=np.dtype('float32'))

            sparse_ids_dataset = query_dev_hdf5.create_dataset(
                'sparse_ids',
                (len(query_text_dim_ids),),
                dtype=dt_uint32)
            sparse_ids_dataset[:] = query_text_dim_ids

            sparse_weights_dataset = query_dev_hdf5.create_dataset(
                'sparse_weights',
                (len(query_text_weights),),
                dtype=dt_float32)
            sparse_weights_dataset[:] = query_text_weights

            query_dev_hdf5.attrs["query_columns_in_hdf5"] = ["query_text", "sparse_dim_ids", "sparse_weights"]
            query_dev_hdf5.attrs["query_columns_type"] = ["string", "array(uint32)", "array(float32)"]
            query_dev_hdf5.attrs["query_columns_in_table"] = ["text", "sparse_vector"]
