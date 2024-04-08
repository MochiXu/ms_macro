from typing import List

import torch.nn.functional as F


def convert_bytes_to_str(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    if not isinstance(text, str):
        text = str(text)
    return text


def texts_to_embeddings(texts: List[str], model):
    embeddings = model.module.encode(texts, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().tolist()

