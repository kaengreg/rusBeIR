## What is it?
<h1 align="center">
<img style="vertical-align:middle" width="450" height="200" src="https://github.com/kngrg/rusBeIR/blob/main/images/rusBeIR_logo.png" />
</h1>

RusBEIR is a Russian benchmark inspired by [BeIR](https://github.com/beir-cellar/beir), designed for zero-shot evaluation of Information Retrieval (IR) models. Adhering to the principles of BEIR, it offers a robust and diverse evaluation framework, enabling the assessment of IR models across a wide range of tasks and domains in the Russian language.
The datasets in the RusBEIR benchmark consist of available open-source datasets, datasets that have been translated from English, and newly created datasets.

## Baselines
NDCG@10 was choosen as a main metric, results could be found [here](https://docs.google.com/spreadsheets/d/19jUZigy-AolNOOhT0EzNggiRoEcvfqL7HRpq0bwHqXc/edit#gid=532683383).
Additional results for [MAP@10](https://docs.google.com/spreadsheets/d/19jUZigy-AolNOOhT0EzNggiRoEcvfqL7HRpq0bwHqXc/edit#gid=1808748240) and [Recall@10](https://docs.google.com/spreadsheets/d/19jUZigy-AolNOOhT0EzNggiRoEcvfqL7HRpq0bwHqXc/edit#gid=1398424942)

## Installation 
``` python
!git clone https://github.com/kngrg/rusBeIR.git
``` 

##  Available Datasets

| Source               | Task                          | Dataset                  | Origin                 | Relevancy | Train   | Dev   | Test   | Corpus    | Avg. Word Lengths (D/Q) |
|----------------------|-------------------------------|--------------------------|------------------------|-----------|---------|-------|--------|-----------|--------------------------|
| BEIR                | Bio-Medical IR                | rus-NFCorpus            | Translation           | Binary    | 2,590   | 324   | 323    | 3,633     | 216.6 / 3.5              |
| BEIR                | Argument Retrieval            | rus-ArguAna             | Translation           | Binary    | —       | —     | 1,406  | 8,674     | 147.8 / 173.8            |
| BEIR                | Fact Checking                 | rus-SciFact             | Translation           | Binary    | 809     | -     | 300    | 5,183     | 185.1 / 11.2             |
| BEIR                | Citation-Prediction           | rus-SCiDOCS             | Translation           | Binary    | —       | —     | 1000   | 25,657    | 153.1 / 9.8              |
| BEIR                | Bio-Medical IR                | rus-TREC-COVID          | Translation           | 3-level   | —       | -     | 50     | 171,332   | 138.9 / 8.5              |
| BEIR                | Question Answering (QA)       | rus-FiQA                | Translation           | Binary    | 5,500   | 500   | 648    | 57,638    | 122.1 / 9.9              |
| BEIR                | Duplicate Question Retrieval  | rus-Quora               | Translation           | Binary    | —       | 5,000 | 10,000 | 522,931   | 9.8 / 7.9                |
| BEIR                | Duplicate Question Retrieval  | rus-CQADupstack         | Translation           | Binary    | —       | —     | 13,145 | 457,199   | 117.6 / 7.6              |
| BEIR                | Argument Retrieval            | rus-Touche              | Translation           | Binary    | —       | —     | 49     | 382,545   | 252.5 / 6.8              |
| BEIR                | Information-Retrieval         | rus-MMARCO             | Part of multilingual  | Binary    | 502,939 | 6,980  | —      | 8,841,823 | 49.6 / 5.95              |
| Open-Source Dataset | Information-Retrieval         | rus-MIRACL             | Part of multilingual  | Binary    | 4,683   | 1,252 | —      | 9,543,918 | 43 / 6.2                 |
| Open-Source Dataset | Question Answering (QA)       | rus-XQuAD              | Part of multilingual  | Binary    | —       | 1,190 | —      | 240       | 112.9 / 8.6              |
| Open-Source Dataset | Question Answering (QA)       | rus-XQuAD-sentences    | Part of multilingual  | Binary    | —       | 1,190 | —      | 1,212     | 22.4 / 8.6               |
| Open-Source Dataset | Question Answering (QA)       | rus-TyDi QA            | Part of multilingual  | Binary    | —       | 1,162 | —      | 89,154    | 69.4 / 6.5               |
| Open-Source Dataset | Information-Retrieval         | SberQUAD-retrieval     | Originally Russian    | Binary    | 45,328  | 5,036 | 23,936 | 17,474    | 100.4 / 8.7              |
| Open-Source Dataset | Information-Retrieval         | rusSciBench-retrieval  | Originally Russian    | Binary    | -       | 345   | -      | 200,532   | 89.9 / 9.2               |
| Open-Source Dataset | Question Answering (QA)       | ru-facts              | Originally Russian    | Binary    | 2,241   | 753   | —      | 6,236     | 28.1 / 23.9              |
| RU-MTEB             | Information-Retrieval         | RuBQ                   | Originally Russian    | Binary    | —       | —     | 1,692  | 56,826    | 62.07 / 6.4              |
| RU-MTEB             | Information-Retrieval         | Ria-News               | Originally Russian    | Binary    | —       | —     | 10,000 | 704,344   | 155.2 / 8.8              |
| rusBEIR             | Information-Retrieval | wikifacts-articles       | Originally Russian  | 3-level   | —     | 540  | —    | 1,324    | 2,535.9 / 11.4          |
| rusBEIR             | Fact Checking         | wikifacts-para           | Originally Russian  | 3-level   | —     | 540  | —    | 15,317   | 219.2 / 11.4            |
| rusBEIR             | Information-Retrieval | wikifacts-sents          | Originally Russian  | 3-level   | —     | 540  | —    | 188,026  | 17.8 / 11.4             |
| rusBEIR             | Fact Checking         | wikifacts-sliding_para2  | Originally Russian  | 3-level   | —     | 540  | —    | 118,025  | 35.7 / 11.4             |
| rusBEIR             | Fact Checking         | wikifacts-sliding_para3  | Originally Russian  | 3-level   | —     | 540  | —    | 188,024  | 53.6 / 11.4             |
| rusBEIR             | Fact Checking         | wikifacts-sliding_para4  | Originally Russian  | 3-level   | —     | 540  | —    | 188,023  | 71.4 / 11.4             |
| rusBEIR             | Fact Checking         | wikifacts-sliding_para5  | Originally Russian  | 3-level   | —     | 540  | —    | 188,022  | 89.3 / 11.4             |
| rusBEIR             | Fact Checking         | wikifacts-sliding_para6  | Originally Russian  | 3-level   | —     | 540  | —    | 188,021  | 107.1 / 11.4            |      

All datasets are available at [HuggingFace](https://huggingface.co/collections/kngrg/rusbeir-66e28cb06e3e074be55ac0f3).

## Models supported now
Encoders:
- BM25
- E5
- BGE
- LaBSE
- [RoSBERTa](https://huggingface.co/ai-forever/ru-en-RoSBERTa)
- [rus-sci-tiny](https://huggingface.co/mlsa-iai-msu-lab/sci-rus-tiny)
- [FRIDA](https://huggingface.co/ai-forever/FRIDA)

Rerankers:
- [BGE-Reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

Any Transformers model can be added via describing class derived from HFTransformers

##  Examples 

### BM25 model
This example shows how to evaluate one dataset using BM25 model with ElasticSearch
```python
"""
This example shows how to evaluate ElasticSearch-BM25 in rusBeIR.
We advise you to use docker for running ElasticSearch. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull docker.elastic.co/elasticsearch/elasticsearch:7.5.2
2. docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.5.2
""" 


from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval

#### Load dataset via HF 
corpus, queries, qrels = HFDataLoader(hf_repo="kngrg/rus-scifact", hf_repo_qrels="kngrg/rus-scifact-qrels", streaming=False,
                                       keep_in_memory=False).load(split='test') # select necessary split train/test/dev

#### Provide parameters for elastic-search
hostname = "localhost:9200"
index_name = "scifact" 

#### Initialize BM25 model and retrieve documents 
model = BM25(index_name=index_name, hostname=hostname, initialize=True)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Evaluate your model with MRR@k where k = [1,3,5,10,100,1000]
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")


metrics = {"ndcg": ndcg, "_map": _map, "recall": recall, "precision": precision, "mrr": mrr}

for metric in metrics.keys():
    for it_num, it_val in zip(metrics[metric], metrics[metric].values()):
        print(it_num, it_val )
    print('\n')
```

This example shows how to evaluate all datasets using BM25 model with ElasticSearch
``` python
from rusBeIR.benchmarking.model_benchmark import DatasetEvaluator
from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25

bm25 = BM25(index_name="bm25", hostname="localhost:9200", initialize=True)
evaluator = DatasetEvaluator(model=bm25)

evaluator.retrieve(text_type='processed_text', results_path='rusBeIR-results)
evaluator.evaluate(results_path='rusBeIR-results')
evaluator.print_results()
```

### E5 model 
This example shows how to evaluate one dataset using E5 model
``` python
from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.retrieval.models.e5 import E5Model
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval

corpus, queries, qrels = HFDataLoader(hf_repo="kngrg/rus-scifact", hf_repo_qrels="kngrg/rus-scifact-qrels", streaming=False,
                                       keep_in_memory=False).load(split='test')

e5 = E5Model()
corpus_emb = e5.encode_passages(corpus)
results = e5.retrieve(queries, corpus_emb, corpus.keys())

retriever = EvaluateRetrieval(k_values=[1,3,5,10, 100, 1000])

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")

metrics = {"ndcg": ndcg, "_map": _map, "recall": recall, "precision": precision, "mrr": mrr}

for metric in metrics.keys():
    for it_num, it_val in zip(metrics[metric], metrics[metric].values()):
        print(it_num, it_val )
    print('\n')
```
This example shows how to evaluate all datasets using E5 model
``` python
from rusBeIR.benchmarking.model_benchmark import DatasetEvaluator
from rusBeIR.retrieval.models.E5Model import E5Model

e5 = E5Model()
evaluator = DatasetEvaluator(model=e5)

evaluator.retrieve(text_type='processed_text', results_path="rusBeIR-e5-results")
evaluator.evaluate(results_path="rusBeIR-e5-results")
evaluator.print_results()
```
