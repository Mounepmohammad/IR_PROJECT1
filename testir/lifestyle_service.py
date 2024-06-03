import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
# إعدادات NLTK

stop_words1 = set(stopwords.words('english'))
# print("this is stop words en",stop_words1)

file = open("common_words", "r")
fileData = file.read()
file.close()
stopwords = re.findall("[a-z]+", fileData)

stop_words = set(stopwords).union(stop_words1)
# print(stop_words)
 

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    # tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def load_data(output_dir):
    docs_dict = joblib.load(os.path.join(output_dir, 'docs_dict.joblib'))
    tfidf_matrix = joblib.load(os.path.join(output_dir, 'tfidf_matrix.joblib'))
    vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer.joblib'))
    return docs_dict, tfidf_matrix, vectorizer

def represent_query_as_vector(query, vectorizer):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector

def retrieve_top_docs(query_vector, tfidf_matrix, top_n):
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_doc_indices = similarities.argsort()[-top_n:][::-1]
    return top_doc_indices, similarities[top_doc_indices]

def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_relevant_docs = [doc for doc in retrieved_docs[:k] if doc in relevant_docs]
    return len(retrieved_relevant_docs) / k

def mean_average_precision(relevant_docs, retrieved_docs, k):
    avg_precision = 0.0
    relevant_count = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            relevant_count += 1
            avg_precision += relevant_count / (i + 1)
    return avg_precision / min(len(relevant_docs), k)

# def recall(relevant_docs, retrieved_docs, k):
#     retrieved_relevant_docs = [doc for doc in retrieved_docs[:k] if doc in relevant_docs]
#     return len(retrieved_relevant_docs) / len(relevant_docs)

def calculate_recall(relevant_docs, retrieved_docs):
    num_relevant_docs = len(relevant_docs)
    # Extract document IDs from relevant_docs
    
    TP = sum(1 for doc in retrieved_docs if doc in relevant_docs)
    FN = num_relevant_docs - TP
    # Calculate the recall
    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    return recall
def evaluate_system(query, relevant_docs, docs_dict, tfidf_matrix, vectorizer, top_n):
    query_vector = represent_query_as_vector(query, vectorizer)
    top_doc_indices, _ = retrieve_top_docs(query_vector, tfidf_matrix, top_n)
    retrieved_docs = [list(docs_dict.keys())[i] for i in top_doc_indices]
    
    prec = precision_at_k(relevant_docs, retrieved_docs, top_n)
    rec =  calculate_recall(relevant_docs, retrieved_docs)
    map_score = mean_average_precision(relevant_docs, retrieved_docs, top_n)
    
    return prec, rec, map_score


output_dir = 'output'
# queries_file = 'C:/Users/DELL/Desktop/IR/lotte/lotte/lifestyle/dev/qas.search.jsonl'  # قم بتحديث المسار إلى ملف الاستعلامات

    # Load stored data using joblib
docs_dict, tfidf_matrix, vectorizer = load_data(output_dir)
    
    # Load queries

# Load queries
questions = []
with open('C:/Users/DELL/Desktop/IR/lotte/lotte/lifestyle/dev/qas.search.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

questions_df = pd.DataFrame(questions)
    # Evaluate the system on each query
     
mapfile = "C:/Users/DELL/Desktop/IR/testir/evaluation1/MAP.txt"
recallfile = "C:/Users/DELL/Desktop/IR/testir/evaluation1/recall.txt"
pricisionfile = "C:/Users/DELL/Desktop/IR/testir/evaluation1/pricision.txt"
total_precision = 0.0
total_recall = 0.0
total_map = 0.0
i = 0
for index, row in questions_df.iterrows():
    query = row['query']
    relevant_docs = list(row['answer_pids'])
    i += 1
    # print(f"Query: {row['query']}")
    # print(f"url: {row['url']}")
    # print(f"pids: {row['answer_pids']}")
    # print()
    precision,recall,map_score = evaluate_system(query, relevant_docs, docs_dict, tfidf_matrix, vectorizer,10)        
    print(f"Query: {query}")
    print(f"Precision@10: {precision:.4f}")
    total_precision +=precision
    with open(pricisionfile , 'a') as P:
        P.write(f"Query: {query}\tprecision: {precision}\n")
    print(f"Recall@10: {recall:.4f}")
    total_recall += recall
    with open(recallfile , 'a') as R:
        R.write(f"Query: {query}\trecall: {recall}\n")
    print(f"MAP@10: {map_score:.4f}")
    total_map += map_score
    with open(mapfile , 'a') as M:
        M.write(f"Query: {query}\tmap_score: {map_score}\n")
    print("-" * 30)
    
                     
print(f"total precision : {total_precision}\n ")
print(f"total recall : {total_recall}\n ")
print(f"total map : {total_map}\n ")
with open(pricisionfile , 'a') as P:
    AP= total_precision / i
    P.write(f"AP: {AP}\n")
with open(recallfile , 'a') as R:
    MRR = total_recall /i
    R.write(f"MRR: {MRR}\n") 
   
with open(mapfile , 'a') as M:
    map = total_map /i
    M.write(f"map: {map}\n")



