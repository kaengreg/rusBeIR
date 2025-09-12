import json
import requests
import re
import unicodedata
from collections import defaultdict
from tqdm import tqdm

# Creating qrels for the paragraph-level corpus requires existing sentence-level qrels and corpus on the sentence-level
# (corpus and qrels can be created with sents.py). 

def preprocess_text(text):
    exclude_chars = "йё"
    text = re.sub(r'а́', 'а', text)
    text = re.sub(r"==\s*(.*?)\s*==\s*", r"\1 ", text)
    text = text.replace('\xa0', ' ').strip()
    text = unicodedata.normalize('NFC', text)
    result = []
    for char in text:
        if char in exclude_chars:
            result.append(char)
        else:
            char_base = unicodedata.normalize('NFD', char)
            char_without_diacritics = ''.join(
                c for c in char_base if not unicodedata.combining(c)
            )
            result.append(char_without_diacritics)
    
    return ''.join(result)

def get_wikipedia_article(article_title):
    url = f"https://ru.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": article_title
    }

    response = requests.get(url, params=params)
    data = response.json()

    pages = data['query']['pages']
    page = next(iter(pages.values()))
    article_text = page['extract'] if 'extract' in page else "Текст не найден."

    return preprocess_text(article_text)


def extract_article_title(url):
    parts = url.split('/wiki/')
    return parts[1] if len(parts) > 1 else None

def is_sentence_in_paragraph(sentence, paragraph):
    """
    Checks whether the normalized text of a sentence is contained in a normalized paragraph.
    """
    return preprocess_text(sentence) in preprocess_text(paragraph)

with open('wiki-corpus.jsonl', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

data = [item for sublist in raw_data for item in sublist]

queries = {}
for idx, item in enumerate(data):
    query_id = f"bwq-{idx}"
    queries[query_id] = {"_id": query_id, "text": preprocess_text(item["fact"]), "title": ""}

with open('queries.jsonl', 'w', encoding='utf-8') as f:
    for record in queries.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Corpus creation - documents are full wikipedia articles
corpus_articles = {}      
doc_counter = 0
processed_links = {}     
query_interval = {}       
fact_counter = 0

for item in tqdm(data, total=len(data), desc="Processing facts"):
    fact_id = f"bwq-{fact_counter}"
    fact_counter += 1
    intervals_for_fact = []
    fact_links = set()  

    links = item.get("links", [])
    for link_item in links:
        link_url = link_item.get("link")
        if link_url in fact_links:
            continue
        fact_links.add(link_url)

        if link_url in processed_links:
            intervals_for_fact.append(processed_links[link_url])
            continue

        article_text = preprocess_text(link_item.get("link_data"))
        doc_id = f"bwc-{doc_counter}"
        corpus_articles[doc_id] = {"_id": doc_id, "title": "", "text": article_text}
        interval = (doc_counter, doc_counter)
        processed_links[link_url] = interval
        intervals_for_fact.append(interval)
        doc_counter += 1

    query_interval[fact_id] = intervals_for_fact

qrels_sents = defaultdict(dict)
with open("qrels.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        fact_id, sent_id, relevance = parts
        qrels_sents[fact_id][sent_id] = int(relevance)

corpus_sentences = {}
with open("corpus.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        corpus_sentences[record["_id"]] = record


# Creation of qrels for the article-level
updated_qrels_articles = {}

for fact_id, sent_rel_dict in tqdm(qrels_sents.items(), desc="Aggregating article-level qrels"):
    intervals = query_interval.get(fact_id, [])
    valid_article_ids = set()
    for (start, end) in intervals:
        for doc_id in corpus_articles.keys():
            try:
                doc_num = int(doc_id.split('-')[1])
            except Exception:
                continue
            if start <= doc_num <= end:
                valid_article_ids.add(doc_id)
    
    article_scores = defaultdict(int)
    for sent_id, sent_rel in sent_rel_dict.items():
        if sent_id not in corpus_sentences:
            continue
        sentence_text = corpus_sentences[sent_id]['text']
        for article_id in valid_article_ids:
            article_text = corpus_articles[article_id]['text']
            if is_sentence_in_paragraph(sentence_text, article_text):
                article_scores[article_id] = max(article_scores[article_id], sent_rel)
    updated_qrels_articles[fact_id] = article_scores


with open("qrels-articles.tsv", "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for fact_id, article_dict in updated_qrels_articles.items():
        for article_id, relevance in article_dict.items():
            f.write(f"{fact_id}\t{article_id}\t{relevance}\n")


with open('corpus-articles.jsonl', 'w', encoding='utf-8') as f:
    for record in corpus_articles.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
