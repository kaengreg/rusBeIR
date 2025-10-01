from typing import Dict
from rusBeIR.retrieval.models.dense.DenseHFModels import DenseHFModels
import torch


class LaBSEModel(DenseHFModels):
    def __init__(self, model_name: str = 'cointegrated/LaBSE-en-ru', maxlen: int = 64, batch_size:int=128, device: str = 'cuda'):
        """
        :param model_name: Name of the pre-trained BGE model from HF.
        :param device: Where to run the model ('cuda' or 'cpu').
        :param maxlen: Models max_length
        :param batch_size: Size of batch that process 
        """
        super().__init__(model_name, maxlen=maxlen, batch_size=batch_size, device=device)
    
    def encode_queries(self, queries: Dict[str, str], pooling_method: str = 'cls'):
        """
        :param queries: Dict of query ids and corresponding queries.
        :param batch_size: Batch size for encoding.
        :param pooling_method: Method for pooling.
        :param prefix: Search prefix if needed
        :return: Query embeddings.
        """
        return super().encode_queries(queries, pooling_method=pooling_method)
    
    def encode_corpus(self, corpus: Dict[str, Dict[str, str]], pooling_method: str = 'cls'):
        """
        :param passages: Dict of passages ids and corresponding passages.
        :param batch_size: Batch size for encoding.
        :param pooling_method: Method for pooling.
        :param prefix: Search prefix if needed
        :return: Passage embeddings.
        """
        return super().encode_corpus(corpus, pooling_method=pooling_method)

    def _cls_pool(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output.pooler_output