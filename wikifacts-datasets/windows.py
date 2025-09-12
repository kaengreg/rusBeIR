import json
import requests
import re
import unicodedata
from razdel import sentenize
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

def is_sentence_in_window(sentence, window_text):
    """
    Checks if the normalized text of a sentence is contained in a normalized window.
    """
    return preprocess_text(sentence) in preprocess_text(window_text)


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


# Creation of corpus - documents are sliding window chunks
global_sentences_list = []  
query_interval_sent = {}    
processed_links = {}       
global_sentence_counter = 0
fact_counter = 0

for fact in tqdm(data, desc="Processing facts (sentences)"):
    fact_id = f"bwq-{fact_counter}"
    fact_counter += 1
    intervals_for_fact = [] 
    fact_links = set()       
    for link_item in fact.get("links", []):
        link_url = link_item.get("link")
        if link_url in fact_links:
            continue
        fact_links.add(link_url)
        if link_url in processed_links:
            intervals_for_fact.append(processed_links[link_url])
            continue

        article_text = preprocess_text(link_item.get("link_data", ""))
        if not article_text:
            continue
        sentence_objs = list(sentenize(article_text))
        sentences = [s.text.strip() for s in sentence_objs if s.text.strip()]
        if not sentences:
            continue
        start_index = global_sentence_counter
        end_index = global_sentence_counter + len(sentences) - 1
        current_interval = (start_index, end_index)
        intervals_for_fact.append(current_interval)
        processed_links[link_url] = current_interval
        for s in sentences:
            sentence_id = f"bwc-{global_sentence_counter}"
            global_sentences_list.append((sentence_id, s, global_sentence_counter))
            global_sentence_counter += 1
    query_interval_sent[fact_id] = intervals_for_fact

with open("query-interval-sent.json", "w", encoding="utf-8") as f:
    json.dump(query_interval_sent, f, ensure_ascii=False, indent=2)

total_sentences = global_sentence_counter
print("Total sentences:", total_sentences)


window_size = 6  # Change this parameter depending on what window size is required

total_windows = total_sentences - window_size + 1

corpus_window = {}            
window_to_sentence_index = {}  

for i in range(total_windows):
    window_sentences = [global_sentences_list[j][1] for j in range(i, i + window_size)]
    window_text = " ".join(window_sentences)
    window_id = f"bw_window-{i}"
    corpus_window[window_id] = {"_id": window_id, "text": window_text, "title": ""}
    window_to_sentence_index[window_id] = global_sentences_list[i][2]


query_interval_window = {}

for fact_id, intervals in query_interval_sent.items():
    window_intervals = []
    for (start, end) in intervals:
        extended_end = end + (window_size - 1)
        if extended_end >= total_sentences:
            extended_end = total_sentences - 1
        window_start = start
        window_end = extended_end - window_size + 1
        if window_end < window_start:
            window_end = window_start
        window_intervals.append((window_start, window_end))
    query_interval_window[fact_id] = window_intervals


fact_ids = sorted(query_interval_window.keys(), key=lambda x: int(x.split('-')[1]))
extended_query_interval = {}

for i, fact_id in enumerate(fact_ids):
    intervals = query_interval_window[fact_id]
    extended_intervals = []
    for (L, R) in intervals:
        if i != 0:
            L = max(0, L - (window_size - 1))
        if i != len(fact_ids) - 1:
            R = R + (window_size - 1)
            if R >= total_windows:
                R = total_windows - 1
        extended_intervals.append((L, R))
    extended_query_interval[fact_id] = extended_intervals

with open('corpus-window6.jsonl', 'w', encoding='utf-8') as f:
    for record in corpus_window.values():
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

# Creation of qrels for the window-level
updated_qrels_window = {}

for fact_id, sent_rel_dict in tqdm(qrels_sents.items(), desc="Aggregating qrels for windows"):
    intervals = extended_query_interval.get(fact_id, [])
    valid_window_ids = set()
    for window_id, start_index in window_to_sentence_index.items():
        window_end = start_index + window_size - 1
        for (interval_start, interval_end) in intervals:
            if start_index <= interval_end and window_end >= interval_start:
                valid_window_ids.add(window_id)
                break

    window_scores = defaultdict(int)
    for sent_id, sent_rel in sent_rel_dict.items():
        if sent_id not in corpus_sentences:
            continue
        sentence_text = corpus_sentences[sent_id]['text']
        for window_id in valid_window_ids:
            window_text = corpus_window[window_id]['text']
            if is_sentence_in_window(sentence_text, window_text):
                window_scores[window_id] = max(window_scores[window_id], sent_rel)
    updated_qrels_window[fact_id] = window_scores

with open("qrels-window6.tsv", "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")
    for fact_id, win_scores in updated_qrels_window.items():
        for window_id, relevance in win_scores.items():
            f.write(f"{fact_id}\t{window_id}\t{relevance}\n")




