from rusBeIR.beir.retrieval.search.base import BaseSearch
import bm25s
import re
from typing import Dict, List

class BM25s(BaseSearch):
    def __init__(self):
        self.bm25 = bm25s.BM25()
        self.doc_ids: List[str] = []       
        self.doc_id_to_index: Dict[str, int] = {}  
        self.indexed = False                 

    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Build the BM25 index for the given corpus using bm25s.
        
        Args:
            corpus: Dictionary of document entries {doc_id: {"title": ..., "text": ...}}
        """
        documents_tokens: List[List[str]] = []
        self.doc_ids = []
        self.doc_id_to_index = {}
        
        for doc_id, doc in corpus.items():
            title = doc.get("title", "")
            text = doc.get("text", "")
            content = ((title + " ") if title else "") + (text if text else "")
            content = content.strip().lower()  

            tokens = re.findall(r"\w+", content)
            documents_tokens.append(tokens)

            self.doc_id_to_index[doc_id] = len(self.doc_ids)
            self.doc_ids.append(doc_id)

        self.bm25.index(documents_tokens)
        self.indexed = True

    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], 
               top_k: int, score_function: str = None, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Execute BM25 search over the corpus for each query and return the top_k results.
        
        Args:
            corpus: Dictionary of documents to search.
            queries: Dictionary of queries {query_id: query_text}.
            top_k: Number of top results to return for each query.
            *args, **kwargs: Additional arguments (not used, for compatibility).
        
        Returns:
            A dict mapping each query_id to another dict of {doc_id: BM25_score} for the top_k results.
        """
        if (not self.indexed or len(corpus) != len(self.doc_ids) or any(doc_id not in self.doc_id_to_index for doc_id in corpus)):
            self.index(corpus)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        query_tokens_list: List[List[str]] = []
        for q_text in query_texts:
            q_tokens = re.findall(r"\w+", q_text.lower())
            query_tokens_list.append(q_tokens)

        docs_matrix, scores_matrix = self.bm25.retrieve(query_tokens_list, k=top_k)
        
        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(query_ids):
            hits: Dict[str, float] = {}
            num_returned = len(docs_matrix[i])
            for j in range(min(num_returned, top_k + 1)):
                doc_index = int(docs_matrix[i][j])
                doc_id = self.doc_ids[doc_index]
                if doc_id == qid:
                    continue
                score = float(scores_matrix[i][j])
                hits[doc_id] = score
            if len(hits) > top_k:
                last_doc_id = next(reversed(hits))
                hits.pop(last_doc_id)
            results[qid] = hits
        return results