import concurrent.futures
import json
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import h5py

from common.utils import convert_bytes_to_str, texts_to_embeddings, sparse_vector_model_id, gpu_compute
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_train_texts_and_vectors(
        train_file_path: str,
        rows_limit: int,
        batch_size: int,
        cuda_count: int
) -> Tuple[Any, List[Any], List[Any], List[Any], List[Any]]:
    df_train = pd.read_csv(train_file_path, sep='\t', header=None, names=['answer-id', 'answer-text'], nrows=rows_limit)

    ids = df_train['answer-id'].tolist()
    print("Convert bytes to str from df_train['answer-text']:")
    texts = [convert_bytes_to_str(raw_str) for raw_str in tqdm.tqdm(df_train['answer-text'].tolist())]
    print(f"""answer_ids size:{len(ids)}
              answer_texts size:{len(texts)}""")

    total_batches = (len(texts) + batch_size - 1) // batch_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 对 texts 文本数据分批
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        # 每个 texts batch 将会在一个 GPU 上执行 embedding
        batch_in_text_model = [text_models[i % cuda_count] for i in range(len(text_batches))]
        # 用来生成 sparse vector 的 model
        batch_in_sparse_model = [sparse_models[i % cuda_count].module for i in range(len(text_batches))]
        # 用来生成 sparse vector 的 tokenizer
        batch_in_sparse_tokenizer = [sparse_tokenizers[i % cuda_count] for i in range(len(text_batches))]

        text_vector_batches = list(
            tqdm.tqdm(
                executor.map(
                    gpu_compute,
                    text_batches,
                    batch_in_text_model,
                    batch_in_sparse_model,
                    batch_in_sparse_tokenizer
                ),
                total=total_batches,
                desc="Processing corpus(mc_macro answers)"
            ))
        # 合并处理后的向量列表
        text_vectors = []
        sparse_vectors_dim_ids = []
        sparse_vectors_weights = []
        for batch in text_vector_batches:
            text_vectors.extend(batch[0])
            sparse_vectors_dim_ids.extend(batch[1])
            sparse_vectors_weights.extend(batch[2])

    return ids, texts, text_vectors, sparse_vectors_dim_ids, sparse_vectors_weights


if __name__ == '__main__':
    texts_batch_size = 64
    num_threads = 2
    limits = 100000  # rows_limit
    store_json = True
    dataset_file_prefix = "/mnt/workspaces/mochix/datasets/ms_macro2"
    # dataset_file_prefix = "dataset_files"
    passages_file_path = f'{dataset_file_prefix}/collection.tsv'
    skip_gpus = [2]
    gpu_count = torch.cuda.device_count()
    gpu_devices = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu') for i in range(gpu_count) if
                   i not in skip_gpus]
    # 初始化 model, model 详细信息参考 hugging face:
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    text_models = [torch.nn.DataParallel(
        SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    ) for _ in range(0, gpu_count)]

    # 生成 sparse vector 的模型
    sparse_models = [torch.nn.DataParallel(
        AutoModelForMaskedLM.from_pretrained(
            sparse_vector_model_id,
            # device=torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu')
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
    for i in range(0, len(gpu_devices)):
        text_models[i].to(gpu_devices[i])
        sparse_models[i].to(gpu_devices[i])

    answer_ids, answer_texts, answer_vectors, answer_sparse_dim_ids, answer_sparse_weights = (
        get_train_texts_and_vectors(
            train_file_path=passages_file_path,
            rows_limit=limits,
            batch_size=texts_batch_size,
            cuda_count=len(gpu_devices)
        ))

    # 存储为 json
    if store_json:
        data = [
            {
                "row_id": row_id,
                "text": text,
                "dim_ids": dim_ids,
                "weights": weights
            }
            for row_id, text, dim_ids, weights in
            zip(answer_ids, answer_texts, answer_sparse_dim_ids, answer_sparse_weights)
        ]
        with open(f"{dataset_file_prefix}/ms-macro-sparse-train.json", "w") as f:
            json.dump(data, f)
    # 创建 train 数据集
    with h5py.File(f'{dataset_file_prefix}/ms-macro-sparse-768-full-cosine.hdf5', 'w') as train_hdf5:
        train_hdf5.create_dataset('text', data=answer_texts)
        train_hdf5.create_dataset('train', data=answer_vectors)

        # 创建 int32 变长数组的特殊数据类型
        dt_uint32 = h5py.special_dtype(vlen=np.dtype('uint32'))
        dt_float32 = h5py.special_dtype(vlen=np.dtype('float32'))

        sparse_ids_dataset = train_hdf5.create_dataset(
            'sparse_ids',
            (len(answer_sparse_dim_ids),),
            dtype=dt_uint32)
        sparse_ids_dataset[:] = answer_sparse_dim_ids

        sparse_weights_dataset = train_hdf5.create_dataset(
            'sparse_weights',
            (len(answer_sparse_weights),),
            dtype=dt_float32)
        sparse_weights_dataset[:] = answer_sparse_weights

        train_hdf5.attrs["extra_columns"] = ["text", "sparse_vector"]
        train_hdf5.attrs["extra_columns_type"] = ["string", "array(tuple)"]
