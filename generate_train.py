import concurrent.futures
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import h5py

from common.utils import convert_bytes_to_str, texts_to_embeddings


def get_train_texts_and_vectors(
        train_file_path: str,
        rows_limit: int,
        batch_size: int,
        cuda_count: int
) -> Tuple[List[int], List[str], List[List[float]]]:

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
        batch_in_model = [models[i % cuda_count] for i in range(len(text_batches))]
        text_vector_batches = list(
            tqdm.tqdm(
                executor.map(texts_to_embeddings, text_batches, batch_in_model),
                total=total_batches,
                desc="Processing corpus(mc_macro answers)"
            ))
        # 合并处理后的向量列表
        text_vectors = [vector for batch in text_vector_batches for vector in batch]

    return ids, texts, text_vectors


if __name__ == '__main__':
    texts_batch_size = 256
    num_threads = 4
    limits = 10000000  # rows_limit
    gpu_count = torch.cuda.device_count()
    dataset_file_prefix = "/mnt/workspaces/mochix/datasets/ms_macro2"
    passages_file_path = f'{dataset_file_prefix}/corpus.jsonl'

    # 初始化 model, model 详细信息参考 hugging face:
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    models = [torch.nn.DataParallel(
        SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    ) for _ in range(0, gpu_count)]

    # move model to gpu
    for i in range(0, gpu_count):
        models[i].to(torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu'))

    answer_ids, answer_texts, answer_vectors = get_train_texts_and_vectors(
        train_file_path=passages_file_path,
        rows_limit=limits,
        batch_size=texts_batch_size,
        cuda_count=gpu_count
    )

    # 创建 train 数据集
    with h5py.File(f'{dataset_file_prefix}/ms-macro2-768-full-cosine.hdf5', 'w') as train_hdf5:
        train_hdf5.create_dataset('text', data=answer_texts)
        train_hdf5.create_dataset('train', data=answer_vectors)
        train_hdf5.attrs["extra_columns"] = ["text"]
        train_hdf5.attrs["extra_columns_type"] = ["string"]

