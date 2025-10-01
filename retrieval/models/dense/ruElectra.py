from typing import Dict
import torch
from rusBeIR.retrieval.models.dense.DenseHFModels import DenseHFModels

class ruElectraTransformers(DenseHFModels):
    def __init__(self, 
                 model_name: str = "ai-forever/ruElectra-large", 
                 maxlen: int = 512, 
                 batch_size: int = 128, 
                 device: str = 'cuda'):
        super().__init__(model_name, maxlen=maxlen, batch_size=batch_size, device=device)

    def encode_queries(self, queries: Dict[str, str], pooling_method: str = 'average'):
        """
        :param queries: Dict of query ids and corresponding queries.
        :param batch_size: Batch size for encoding.
        :param pooling_method: Method for pooling.
        :param prefix: Search prefix if needed
        :return: Query embeddings.
        """
        return super().encode_queries(queries, pooling_method=pooling_method)
    
    def encode_corpus(self, corpus: Dict[str, Dict[str, str]], pooling_method: str = 'average'):
        """
        :param passages: Dict of passages ids and corresponding passages.
        :param batch_size: Batch size for encoding.
        :param pooling_method: Method for pooling.
        :param prefix: Search prefix if needed
        :return: Passage embeddings.
        """
        return super().encode_corpus(corpus, pooling_method=pooling_method)
    
    def _average_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask