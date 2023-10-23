# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import glob

import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
import json

import subprocess
from sklearn.metrics import f1_score

from scipy.stats import spearmanr
from scipy.stats import kendalltau

# %% [markdown]
# # Search Engine Evaluation

# %%
def generate_random_ranking(N: int):
    ranked_list = list(range(N))
    random.shuffle(ranked_list)
    relevant_list = [1 if random.random() < 0.5 else 0 for _ in range(N)]
    return ranked_list, relevant_list

def run_experiments(num_experiments: int, num_iterations: int, N=0):
    experiment_data = []
    for _ in range(num_experiments):
        ranked_lists = []
        relevant_lists = []
        if N <= 0:
            N = random.randint(5, 20)
        # N is fixed for an experiment
        for _ in range(num_iterations):
            ranked_list, relevant_list = generate_random_ranking(N)
            ranked_lists.append(ranked_list)
            relevant_lists.append(relevant_list)
        experiment_data.append((N, ranked_lists, relevant_lists))
    return experiment_data

# %%
experiment_data = run_experiments(1, 100)

# %% [markdown]
# ## Task 1

# %%
def precision_recall(ranked_list: list, relevant_list: list, rank=0):
    num_relevant = sum(relevant_list)
    num_retrieved = 0
    num_relevant_and_retrieved = 0
    precision = []
    recall = []
    for i in range(len(ranked_list)):
        num_retrieved += 1
        if relevant_list[ranked_list[i]] == 1:
            num_relevant_and_retrieved += 1
        precision.append(num_relevant_and_retrieved / num_retrieved)
        recall.append(num_relevant_and_retrieved / num_relevant)
    if rank > 0 and rank <= len(ranked_list):
        return precision[rank-1], recall[rank-1]
    return precision, recall

# %% [markdown]
# ### Complete precison-recall

# %%
# Example to get entire precision and recall from a ranking in a single experiment
N, ranked_lists, relevant_lists = experiment_data[0]

idx = 0
ranked_list, relevant_list = ranked_lists[idx], relevant_lists[idx]
precision, recall = precision_recall(ranked_list, relevant_list)
print(f"Ranked List:\n{ranked_list}")
print(f"Relevant List:\n{relevant_list}")
print(f"\nPrecision:\n{precision}")
print(f"Recall:\n{recall}")

# %% [markdown]
# ### Precision and Recall @ Rank

# %%
# Example to get precision and recall at a particular rank from a ranking in a single experiment
N, ranked_lists, relevant_lists = experiment_data[0]

idx = 0
rank = 5
ranked_list, relevant_list = ranked_lists[idx], relevant_lists[idx]
precision, recall = precision_recall(ranked_list, relevant_list, rank=rank)
print(f"Ranked List:\n{ranked_list}")
print(f"Relevant List:\n{relevant_list}")
print(f"\nPrecision@{rank}: {precision}")
print(f"Recall@{rank}: {recall}")

# %% [markdown]
# ## Task 2

# %% [markdown]
# ### Random rankings P-R curve

# %%
# Plot the P-R curves for 2 ranking lists in a single experiment. Each experiment contains multiple ranking lists and each ranking list corresponds to 1 query.
N, ranked_lists, relevant_lists = experiment_data[0]

# Select first random ranking list from the experiment
rand_idx = random.randint(0, len(ranked_lists)-1)
ranked_list, relevant_list = ranked_lists[rand_idx], relevant_lists[rand_idx]
precision, recall = precision_recall(ranked_list, relevant_list)
print("Trial 1")
print(f"Ranked List:\n{ranked_list}")
print(f"Relevant List:\n{relevant_list}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Select second random ranking list from the experiment
rand_idx2 = random.randint(0, len(ranked_lists)-1)
while rand_idx2 == rand_idx:
    rand_idx2 = random.randint(0, len(ranked_lists)-1)
ranked_list2, relevant_list2 = ranked_lists[rand_idx2], relevant_lists[rand_idx2]
precision2, recall2 = precision_recall(ranked_list2, relevant_list2)
print("\nTrial 2")
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"\nPrecision: {precision2}")
print(f"Recall: {recall2}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f"Query {rand_idx}")
plt.scatter(recall, precision)
plt.plot(recall2, precision2, label=f"Query {rand_idx2}")
plt.scatter(recall2, precision2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ### Rankings for BM25

# %%
def remove_characters(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_text_files(path: str, N=0, shuffle=False) -> list:
    text_files = glob.glob(f"{path}/*.txt")
    if shuffle:
        random.shuffle(text_files)
    if N > 0:
        text_files = text_files[:N]
    return text_files

def get_text(text_files: list) -> list:
    text = []
    for text_file in text_files:
        with open(text_file, 'r', errors='ignore') as f:
            content = remove_characters(f.read())
            content = content.lower()
            text.append(content)
    return text

def tokenization(text):
    if type(text) == list:
        return [word_tokenize(t) for t in text]
    elif type(text) == str:
        return word_tokenize(text)
    return None

def stemmer(tokenized_text: list):
    ps = PorterStemmer()
    stemmed_text = []
    for doc in tokenized_text:
        stemmed_text.append([ps.stem(token) for token in doc])

    stemmed_dict = {}
    for doc in stemmed_text:
        for token in doc:
            if token in stemmed_dict:
                stemmed_dict[token] += 1
            else:
                stemmed_dict[token] = 1
    
    return stemmed_dict, stemmed_text

# %%
def get_terms_per_doc(tokenized_text: list):
    terms_per_doc = [set(doc) for doc in tokenized_text]
    return terms_per_doc

def get_terms(tokenized_text: list):
    terms = set()
    for doc in tokenized_text:
        for token in doc:
            terms.add(token)
    return list(terms)

# Term Frequency
def get_tf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    tf = {}
    if stemming:
        ps = PorterStemmer()
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                root = ps.stem(token)
                if root in freq_dict:
                    freq_dict[root] += 1
                else:
                    freq_dict[root] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    else:
        for i, doc in enumerate(tokenized_text):
            freq_dict = {}
            for token in doc:
                if token in freq_dict:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] = 1
            file = text_file_names[i]
            tf[file] = freq_dict
    return tf

# Inverse Document Frequency
def get_idf_dict(tokenized_text: list, text_file_names: list, stemming=False):
    if stemming:
        _, tokenized_text = stemmer(tokenized_text)
    terms_per_doc = get_terms_per_doc(tokenized_text)
    terms = get_terms(tokenized_text)
    idf = {}
    N = len(text_file_names)
    for term in terms:
        count = 0
        for doc in terms_per_doc:
            if term in doc:
                count += 1
        idf[term] = np.log2(N / count)
    return idf

# BM25
def get_BM25_matrix(tokenized_text: list, text_files: list, stemming=False, k=1.75, b=0.75):
    tf_dict = get_tf_dict(tokenized_text, text_files, stemming=stemming)
    idf_dict = get_idf_dict(tokenized_text, text_files, stemming=stemming)
    docLengths = [len(doc) for doc in tokenized_text]
    avgDocLength = np.mean(docLengths)
    bm25_matrix = pd.DataFrame.from_dict(tf_dict)
    bm25_matrix = bm25_matrix.fillna(0)
    terms = bm25_matrix.index.values.tolist()
    docs = bm25_matrix.columns.values.tolist()
    for i, doc in enumerate(docs):
        bm25_matrix[doc] = bm25_matrix[doc] * (k + 1) / (bm25_matrix[doc] + k * (1 - b + b * docLengths[i] / avgDocLength))
    for term in terms:
        bm25_matrix.loc[term] = bm25_matrix.loc[term] * idf_dict[term]
    return bm25_matrix

def generate_ranking(df: pd.DataFrame, query: str):
    if query not in df.index:
        return None
    return df.loc[query].sort_values(ascending=False)

# %%
# Generate bm25 matrix
directory = "./nasa"
text_files = get_text_files(directory, shuffle=False)
text_file_names = [text_file.split('/')[-1] for text_file in text_files]
text = get_text(text_files)
tokenized_text = tokenization(text)
bm25_matrix = get_BM25_matrix(tokenized_text, text_file_names, stemming=False)

queries = ["engine", "analysis"]

# %% [markdown]
# #### Query 1: "engine"

# %%
# Generate ranking for first query
query = queries[0]
ranking = generate_ranking(bm25_matrix, query)
print(f"Query: {query}")
print(f"Ranking:\n{ranking}")

# %%
# Get relevant documents from .key files
cmd = f"grep -rlE '\\b{query}\\b' {directory}/*.key"
output = !{cmd}
relevant_docs = output.grep(".*")
relevant_docs = [doc.split("/")[-1].split(".")[0] + ".txt" for doc in relevant_docs]
print(f"Relevant documents:\n{relevant_docs}")

# %%
# Calculate precision and recall for first query
ranked_list1 = [text_file_names.index(doc) for doc in ranking.index.values.tolist()]
relevant_list1 = [1 if text_file_name in relevant_docs else 0 for text_file_name in text_file_names]
precision, recall = precision_recall(ranked_list1, relevant_list1)
print(f"Ranked List:\n{ranked_list1}")
print(f"Relevant List:\n{relevant_list1}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f"Query = {query}")
plt.scatter(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for BM25")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# #### Query 2: "analysis"

# %%
# Generate ranking for second query
query = queries[1]
ranking = generate_ranking(bm25_matrix, query)
print(f"Query: {query}")
print(f"Ranking:\n{ranking}")

# %%
# Get relevant documents from .key files
cmd = f"grep -rlE '\\b{query}\\b' {directory}/*.key"
output = !{cmd}
relevant_docs = output.grep(".*")
relevant_docs = [doc.split("/")[-1].split(".")[0] + ".txt" for doc in relevant_docs]
print(f"Relevant documents:\n{relevant_docs}")

# %%
# Calculate precision and recall for second query
ranked_list2 = [text_file_names.index(doc) for doc in ranking.index.values.tolist()]
relevant_list2 = [1 if text_file_name in relevant_docs else 0 for text_file_name in text_file_names]
precision, recall = precision_recall(ranked_list2, relevant_list2)
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f"Query = {query}")
plt.scatter(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for BM25")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Task 3

# %%
def nDCG(ranked_list: list, relevant_list: list, rank=0):
    gains = []
    for i in range(len(ranked_list)):
        if relevant_list[ranked_list[i]] == 1:
            gains.append(1 / np.log2(i + 2))
        else:
            gains.append(0)
    dcg = [gains[0]]
    for i in range(1, len(gains)):
        dcg.append(dcg[i-1] + gains[i])
    print(f"DCG: {dcg}")

    ideal_ranked_list = sorted(range(len(relevant_list)), key=lambda k: relevant_list[k], reverse=True)
    ideal_gains = []
    for i in range(len(ideal_ranked_list)):
        if relevant_list[ideal_ranked_list[i]] == 1:
            ideal_gains.append(1 / np.log2(i + 2))
        else:
            ideal_gains.append(0)
    ideal_dcg = [ideal_gains[0]]
    for i in range(1, len(ideal_gains)):
        ideal_dcg.append(ideal_dcg[i-1] + ideal_gains[i])
    print(f"Ideal DCG: {ideal_dcg}")
    
    ndcg = [dcg[i]/ideal_dcg[i] for i in range(len(dcg))]
    if rank > 0 and rank <= len(ranked_list):
        return ndcg[rank-1]
    return ndcg

# %% [markdown]
# ### Random rankings for nDCG

# %%
# Compute nDCG for a ranking in a single experiment
N, ranked_lists, relevant_lists = experiment_data[0]
print(f"Ranked List:\n{ranked_list}")
print(f"Relevant List:\n{relevant_list}")
print(f"Text files in order:\n{text_file_names}\n")

ndcg = nDCG(ranked_lists[0], relevant_lists[0])
print(f"nDCG: {ndcg}")

# %% [markdown]
# ### Rankings for BM25

# %% [markdown]
# #### Query 1: "engine"

# %%
# Calculate nDCG for query1
query = queries[0]
print(f"Ranked List:\n{ranked_list1}")
print(f"Relevant List:\n{relevant_list1}")
print(f"Text files in order:\n{text_file_names}\n")

ndcg = nDCG(ranked_list1, relevant_list1)
print(f"nDCG: {ndcg}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ndcg)+1), ndcg, label=f"Query = {query}")
plt.scatter(range(1, len(ndcg)+1), ndcg)
plt.xlabel("Rank")
plt.ylabel("nDCG")
plt.title("nDCG Curve for BM25")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# #### Query 2: "analysis"

# %%
# Calculate nDCG for query2
query = queries[1]
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"Text files in order:\n{text_file_names}\n")

ndcg = nDCG(ranked_list2, relevant_list2)
print(f"nDCG: {ndcg}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ndcg)+1), ndcg, label=f"Query = {query}")
plt.scatter(range(1, len(ndcg)+1), ndcg)
plt.xlabel("Rank")
plt.ylabel("nDCG")
plt.title("nDCG Curve for BM25")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Task 4

# %% [markdown]
# #### Query 1: "engine"

# %%
query = queries[0]
precision, recall = precision_recall(ranked_list1, relevant_list1)
print(f"Ranked List:\n{ranked_list1}")
print(f"Relevant List:\n{relevant_list1}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")

# Average Precision till recall at 1
avg_precision1 = np.mean(precision[:np.argmax(recall)+1])
print(f"\nAverage Precision: {avg_precision1}")

# %% [markdown]
# #### Query 2: "analysis"

# %%
query = queries[1]
precision, recall = precision_recall(ranked_list2, relevant_list2)
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")

# Average Precision till recall at 1
avg_precision2 = np.mean(precision[:np.argmax(recall)+1])
print(f"\nAverage Precision: {avg_precision2}")

# %%
print(f"Mean Average Precision: {np.mean([avg_precision1, avg_precision2])}")

# %% [markdown]
# ## Task 5

# %%
def f1_PR(precision: list, recall: list):
    f1 = []
    for i in range(len(precision)):
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append((precision[i] * recall[i]) / ((precision[i]) + recall[i]))
        if recall[i] == 1:
            break
    return f1

def f1_MM(ranked_list: list, relevant_list: list):
    ranked_relevant_list = [relevant_list[i] for i in ranked_list]

    # Compute micro F1 score
    # Find the last position of 1 in top_k_ranked_rel
    last_pos_rel = len(ranked_relevant_list) - list(reversed(ranked_relevant_list)).index(1)

    last_pos_notrel = len(ranked_relevant_list) - list(reversed(ranked_relevant_list)).index(0)

    tp_rel = np.sum(ranked_relevant_list)
    tp_not_rel = len(ranked_relevant_list) - tp_rel
    
    p_rel = last_pos_rel
    p_not_rel = last_pos_notrel

    tp = tp_rel + tp_not_rel
    p = p_rel + p_not_rel

    micro_f1_score = (2 * tp) / (p + tp)

    # Compute macro F1 score
    # For relevant documents
    macro_precision_rel, macro_recall_rel = precision_recall(ranked_list, relevant_list)

    # Change all 0s to 1s and vice versa in ranked_relevant_list
    for i in range(len(ranked_relevant_list)):
        ranked_relevant_list[i] = 1 - ranked_relevant_list[i]

    # For non relevant documents
    macro_precision_not_rel, macro_recall_not_rel = precision_recall(ranked_list, relevant_list)

    macro_f1_scores_rel = f1_PR(macro_precision_rel, macro_recall_rel)
    macro_f1_scores_not_rel = f1_PR(macro_precision_not_rel, macro_recall_not_rel)

    macro_f1_score_rel = np.mean(macro_f1_scores_rel)
    macro_f1_score_not_rel = np.mean(macro_f1_scores_not_rel)

    return micro_f1_score, (macro_f1_score_rel + macro_f1_score_not_rel) / 2

# %% [markdown]
# ### Example

# %%
ranked_list = [3, 6, 2, 4, 7, 9, 0, 1, 5, 8]
relevant_list = [1, 0, 1, 1, 1, 1, 0, 0, 0, 0]

micro_f1_scores, macro_f1_scores = f1_MM(ranked_list, relevant_list)

print("Micro F1 Scores at Each Rank:", micro_f1_scores)
print("Macro F1 Scores at Each Rank:", macro_f1_scores)

# %% [markdown]
# #### Query 1: "engine"

# %%
query = queries[0]
precision, recall = precision_recall(ranked_list1, relevant_list1)
f1 = f1_PR(precision, recall)
print(f"Ranked List:\n{ranked_list1}")
print(f"Relevant List:\n{relevant_list1}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Average Precision till recall at 1
avg_precision1 = np.mean(precision[:np.argmax(recall)+1])
print(f"\nAverage Precision: {avg_precision1}")

# Average F1 Score till recall at 1
avg_f1 = np.mean(f1[:np.argmax(recall)+1])
print(f"\nAverage F1 Score: {avg_f1}")

# %%
micro_f1_scores, macro_f1_scores = f1_MM(ranked_list1, relevant_list1)

print("Micro F1 Scores at Each Rank:", micro_f1_scores)
print("Macro F1 Scores at Each Rank:", macro_f1_scores)

# %%
# plot the micro and macro F1 scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(micro_f1_scores)+1), micro_f1_scores, label="Micro F1")
# plt.scatter(range(1, len(micro_f1_scores)+1), micro_f1_scores)
plt.plot(range(1, len(macro_f1_scores)+1), macro_f1_scores, label="Macro F1")
# plt.scatter(range(1, len(macro_f1_scores)+1), macro_f1_scores)
plt.xlabel("Rank")
plt.ylabel("F1 Score")
plt.title(f"F1 Score Curve for BM25 for query = {query}")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# #### Query 2: "analysis"

# %%
query = queries[1]
precision, recall = precision_recall(ranked_list2, relevant_list2)
f1 = f1_PR(precision, recall)
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Average Precision till recall at 1
avg_precision2 = np.mean(precision[:np.argmax(recall)+1])
print(f"\nAverage Precision: {avg_precision2}")

# Average F1 Score till recall at 1
avg_f1 = np.mean(f1[:np.argmax(recall)+1])
print(f"\nAverage F1 Score: {avg_f1}")

# %%
micro_f1_scores, macro_f1_scores = f1_MM(ranked_list2, relevant_list2)

print("Micro F1 Scores at Each Rank:", micro_f1_scores)
print("Macro F1 Scores at Each Rank:", macro_f1_scores)

# %%
# plot the micro and macro F1 scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(micro_f1_scores)+1), micro_f1_scores, label="Micro F1")
# plt.scatter(range(1, len(micro_f1_scores)+1), micro_f1_scores)
plt.plot(range(1, len(macro_f1_scores)+1), macro_f1_scores, label="Macro F1")
# plt.scatter(range(1, len(macro_f1_scores)+1), macro_f1_scores)
plt.xlabel("Rank")
plt.ylabel("F1 Score")
plt.title(f"F1 Score Curve for BM25 for query = {query}")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Task 6

# %%
def spearman_correlation(ranked_list, relevance_list):
    ordered_relevance_list = [relevance_list[i] for i in ranked_list]
    correlation, _ = spearmanr(ranked_list, ordered_relevance_list)
    return correlation

def kendall_tau_correlation(ranked_list, relevance_list):
    ordered_relevance_list = [relevance_list[i] for i in ranked_list]
    correlation, _ = kendalltau(ranked_list, ordered_relevance_list)
    return correlation

# %% [markdown]
# ### Query 1: "engine"

# %%
query = queries[0]
print(f"Ranked List:\n{ranked_list1}")
print(f"Relevant List:\n{relevant_list1}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nSpearman Correlation: {spearman_correlation(ranked_list1, relevant_list1)}")
print(f"Kendall Tau Correlation: {kendall_tau_correlation(ranked_list1, relevant_list1)}")

# %% [markdown]
# ### Query 2: "analysis"

# %%
query = queries[1]
print(f"Ranked List:\n{ranked_list2}")
print(f"Relevant List:\n{relevant_list2}")
print(f"Text files in order:\n{text_file_names}")
print(f"\nSpearman Correlation: {spearman_correlation(ranked_list2, relevant_list2)}")
print(f"Kendall Tau Correlation: {kendall_tau_correlation(ranked_list2, relevant_list2)}")

# %% [markdown]
# # Zipf's Law

# %% [markdown]
# ### Identify all unique words

# %%
french_book = "./lhomme_qui_rit.txt"
with open(french_book, 'r', errors='ignore') as f:
    french_text = f.read()

# Tokenize the text using regular expressions
words = re.findall(r'\b\w+\b', french_text, flags=re.UNICODE)
words_lower = [word.lower() for word in words]

# Convert to lowercase and create a set to get unique words
unique_words = list(set(words_lower))
print(f"Unique words: {unique_words}")
print(f"Number of unique words: {len(unique_words)}")

# %% [markdown]
# ### Get document frequency and sort

# %%
doc_freq = {}
for word in unique_words:
    doc_freq[word] = words_lower.count(word)

# Sort the dictionary by descending order of frequency
sorted_doc_freq = {k: v for k, v in sorted(doc_freq.items(), key=lambda item: item[1], reverse=True)}
print("Top 10 words with highest frequency:")
for i, (k, v) in enumerate(sorted_doc_freq.items()):
    if i == 10:
        break
    print(f"{k}: {v}")

ranked_words_by_freq = list(sorted_doc_freq.keys())

# %%
k = 50
df = pd.DataFrame.from_dict(sorted_doc_freq, orient='index', columns=['Frequency'])
df.index.name = 'Word'
df.head(k)

# %%
sns.barplot(x=df.index[:k], y=df['Frequency'][:k])
plt.xticks(rotation=90)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Frequency of Words")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Transform frequencies into probabilities

# %%
n = len(words_lower)

# Divide by document length to get probability of each word
sorted_doc_freq_prob = {k: v/n for k, v in sorted_doc_freq.items()}

# %%
# Take log2 of the probabilities for better visualization
probabilities = np.log2(np.array(list(sorted_doc_freq_prob.values())))
ranks = list(range(1, len(probabilities)+1))

# Zipf's Law
x = np.linspace(1, len(probabilities), len(probabilities))
y = np.log2(np.array([0.1/r for r in x]))

plt.figure(figsize=(10, 6))
plt.plot(ranks, probabilities, label="Actual Probabilities")
# plt.scatter(ranks, probabilities)
plt.plot(x, y, label="Zipf's Law")

# plot the linear regression line
plt.plot(np.unique(ranks), np.poly1d(np.polyfit(ranks, probabilities, 1))(np.unique(ranks)), label="Linear Regression")
# print the values of R-squared and p for the linear regression line
print(f"R-squared: {np.corrcoef(ranks, probabilities)[0,1] ** 2}")
print(f"p: {np.polyfit(ranks, probabilities, 1)[0]}")

plt.xlabel("Rank")
plt.ylabel("log(Probability)")
plt.title("Probability - Rank Curve 1")
plt.legend()
plt.grid()
plt.show()

# %%
# Take log2 of the ranks to compare with log2 of probabilities
probabilities = np.array(list(sorted_doc_freq_prob.values()))
ranks = np.log2(np.array(list(range(1, len(probabilities)+1))))

# Zipf's Law
x = np.linspace(1, len(probabilities), len(probabilities))
y = np.array([0.1/r for r in x])
x = np.log2(x)

plt.figure(figsize=(10, 6))
plt.plot(ranks, probabilities, label="Actual Probabilities")
# plt.scatter(ranks, probabilities)
plt.plot(x, y, label="Zipf's Law")

# plot the linear regression line
plt.plot(np.unique(ranks), np.poly1d(np.polyfit(ranks, probabilities, 1))(np.unique(ranks)), label="Linear Regression")
# print the values of R-squared and p for the linear regression line
print(f"R-squared: {np.corrcoef(ranks, probabilities)[0,1] ** 2}")
print(f"p: {np.polyfit(ranks, probabilities, 1)[0]}")

plt.xlabel("log(Rank)")
plt.ylabel("Probability")
plt.title("Probability - Rank Curve 2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# From the above 2 probability - rank curves, it is clear that the Zipf's law curve seems to approximate the actual probabilities at each rank, with good accuracy for a majority of the curve. The linear regression model does a better job of fitting the 2nd curve than the first. This is most likely due to the fact that the first curve follows Zipf's law well, since probability (and log(probability)) are inversely proportional to rank, and the second curve is more evened out compared to the first, since we are plotting the log of rank on the x-axis.

# %% [markdown]
# ### Word categories

# %%
df = pd.DataFrame(columns=['Very Frequent Words', 'Averagely Frequent Words', 'Median Frequent Words', 'Very Rare Words'])
items = list(sorted_doc_freq.items())

print("Very Frequent Words:")
col = []
for i in range(10):
    print(f"{items[i][0]}: {items[i][1]}")
    col.append(items[i][0])
df['Very Frequent Words'] = col

avg_freq = np.mean([items[i][1] for i in range(len(items))])
print(f"\nAverage Frequency: {avg_freq}")
start = -1
end = -1
for i in range(len(items)):
    if items[i][1] <= avg_freq:
        start = i
        break
end = start
while end < len(items) and items[end][1] == avg_freq:
    end += 1
idx = (start + end) // 2
print("Averagely Frequent Words:")
col = []
for i in range(idx - 5, idx + 5):
    print(f"{items[i][0]}: {items[i][1]}")
    col.append(items[i][0])
df['Averagely Frequent Words'] = col

print("\nMedian Frequent Words:")
col = []
for i in range(len(items)//2 - 5, len(items)//2 + 5):
    print(f"{items[i][0]}: {items[i][1]}")
    col.append(items[i][0])
df['Median Frequent Words'] = col

print("\nVery Rare Words:") 
col = []
for i in range(len(items)-10, len(items)):
    print(f"{items[i][0]}: {items[i][1]}")
    col.append(items[i][0])
df['Very Rare Words'] = col

# %%
df

# %% [markdown]
# Intuitively, the last category of 'very rare words' would seem to be more useful in information retrieval since these rare words contain more overall information. Also, since they are sparsely located, the location or position of these words could be useful for analysis, like word association or named-entity-recognition tasks. This last category is also likely to have higher tf-idf score, since idf for these words will be high in a large corpus. In smaller corpuses, the very frequent words will have high tf-idf value. 


