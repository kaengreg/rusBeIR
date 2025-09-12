import json
import requests
import re
import unicodedata
from razdel import sentenize
from tqdm import tqdm

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


def find_id_by_text(corpus, text, threshold=0.7):
    target_words = set(preprocess_text(text).lower().split())
    
    best_match_id = None
    best_similarity = 0
    
    for doc_data in corpus.values():
        corpus_words = set(preprocess_text(doc_data['text']).lower().split())
        intersection = len(target_words & corpus_words)
        union = len(target_words | corpus_words)
        similarity = intersection / union if union != 0 else 0
        
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match_id = doc_data['_id']
    
    return best_match_id

# Loading data
with open('wiki-corpus.jsonl', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

data = [item for sublist in raw_data for item in sublist]

# Creation of queries
queries = {}
for idx, item in enumerate(data):
    query_id = f"bwq-{idx}"
    queries[query_id] = {"_id": query_id, "text": preprocess_text(item["fact"]), "title": ""}


with open('queries.jsonl', 'w', encoding='utf-8') as f:
    for record in queries.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Creation of corpus and link_intervals
corpus = {}
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
        sentences = [sent.text for sent in sentenize(article_text)]

        article_id_start = doc_counter
        for sentence in sentences:
            if sentence == "":
                continue
            doc_id = f"bwc-{doc_counter}"
            corpus[doc_id] = {"_id": doc_id, "title": "", "text": sentence}
            doc_counter += 1
        article_id_end = doc_counter - 1

        interval = (article_id_start, article_id_end)
        processed_links[link_url] = interval
        intervals_for_fact.append(interval)

    query_interval[fact_id] = intervals_for_fact


with open('queries_interval.json', 'w', encoding='utf-8') as f:
    json.dump(query_interval, f, ensure_ascii=False, indent=4)

with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for record in corpus.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open('link_intervals.jsonl', 'w', encoding='utf-8') as f:
    for record in processed_links.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Creation of qrels (relevance markup)
qrels_lines = []
not_found_lines = []

for idx, item in tqdm(enumerate(data), total=len(data), desc="Creating qrels"):
    query_id = f"bwq-{idx}"
    links = item.get("links", [])
    for link_item in links:
        article_url = link_item.get("link")
        article_id_start, article_id_end = processed_links.get(article_url, (None, None))

        if article_id_start is None or article_id_end is None:
            continue 

        for score_item in link_item.get("scores", []):
            score_text = score_item.get("text")
            relevance = score_item.get("score")

            if int(relevance) == 0:
                continue  

            doc_id = find_id_by_text({k: v for k, v in corpus.items() if article_id_start <= int(k.split('-')[1]) <= article_id_end}, score_text, threshold=0.2)
            
            if doc_id:
                qrels_lines.append(f"{query_id}\t{doc_id}\t{relevance}")
            else:
                not_found_lines.append({"query_id": query_id, "score_text": score_text, "relevance": relevance})


with open('qrels.tsv', 'w', encoding='utf-8') as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for line in qrels_lines:
        f.write(line + "\n")

with open('not_found.jsonl', 'w', encoding='utf-8') as f:
    for line in not_found_lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

