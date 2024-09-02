from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

sparse_vector_model_id = 'naver/splade-cocondenser-ensembledistil'


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


def compute_sparse_vector(texts: List[str], model, tokenizer=AutoTokenizer.from_pretrained(sparse_vector_model_id)):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.

    Args:
    logits (torch.Tensor): The logits output from a model.
    attention_mask (torch.Tensor): The attention mask corresponding to the input tokens.

    Returns:
    torch.Tensor: Computed vector.
    """
    tokens = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)

    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    dim_ids = []
    weights = []
    for vector in vec:
        cols = vector.nonzero().squeeze().cpu().tolist()
        dim_ids.append(cols)
        weights.append(vector[cols].cpu().tolist())

    return dim_ids, weights


def gpu_compute(texts: List[str], vector_model, sparse_model, sparse_tokenizer):
    vectors = texts_to_embeddings(texts, vector_model)
    sparse_dim_ids, sparse_weights = compute_sparse_vector(texts, sparse_model, sparse_tokenizer)
    print(f"vectors: {vectors}\nsparse_dim_ids: {sparse_dim_ids}\nsparse_weights: {sparse_weights}\n")
    return vectors, sparse_dim_ids, sparse_weights
