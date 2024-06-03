import os
import json
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

# إعدادات NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def load_queries(file_path):
    questions = []
    with open(file_path , 'r', encoding='utf-8') as f:
      for line in f:
         
        questions.append(json.loads(line))
    questions_df = pd.DataFrame(questions)
    return questions_df

def extract_important_words(queries, vectorizer):
    questions = queries['query'].apply(preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(questions)
    feature_names = vectorizer.get_feature_names_out()
    
    important_words = set()
    for i in range(tfidf_matrix.shape[0]):
        for j in tfidf_matrix[i].nonzero()[1]:
            important_words.add(feature_names[j])
    
    return important_words

def process_term(term, term_index, tfidf_matrix, doc_ids, important_words):
    term_postings = []
    if term in important_words:
        for doc_index in range(tfidf_matrix.shape[0]):
            tfidf_value = tfidf_matrix[doc_index, term_index]
            if tfidf_value > 0:
                doc_id = doc_ids[doc_index]
                term_postings.append((doc_id, tfidf_value))
    return term, term_postings

def build_inverted_index(tfidf_matrix, vectorizer, doc_ids, important_words, n_jobs=-1):
    inverted_index = defaultdict(list)
    terms = vectorizer.get_feature_names_out()
    
    num_terms = len(terms)
    print(f"Number of terms: {num_terms}")
    
    # استخدام joblib لتوازي معالجة المصطلحات مع tqdm لإظهار شريط التقدم
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_term)(term, term_index, tfidf_matrix, doc_ids, important_words) 
        for term_index, term in tqdm(enumerate(terms), total=num_terms, desc="Processing terms")
    )
    
    for term, term_postings in results:
        if term_postings:
            inverted_index[term].extend(term_postings)
    
    return inverted_index

def save_inverted_index_to_json(inverted_index, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)


output_dir = 'output'
queries_file = 'C:/Users/DELL/Desktop/IR/lotte/lotte/lifestyle/dev/qas.search.jsonl' # قم بتحديث المسار إلى ملف الاستعلامات
    
    # Load stored data using joblib
docs_dict = joblib.load(os.path.join(output_dir, 'docs_dict.joblib'))
tfidf_matrix = joblib.load(os.path.join(output_dir, 'tfidf_matrix.joblib'))
vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer.joblib'))
    
    # Ensure tfidf_matrix is a csr_matrix
if not isinstance(tfidf_matrix, csr_matrix):
    tfidf_matrix = csr_matrix(tfidf_matrix)
    
    # Get the list of document IDs from docs_dict
doc_ids = list(docs_dict.keys())
    
    # Load queries and extract important words
queries = load_queries(queries_file)
important_words = extract_important_words(queries, vectorizer)
    
    # Build the inverted index
inverted_index = build_inverted_index(tfidf_matrix, vectorizer, doc_ids, important_words, n_jobs=-1)
    
    # Save the inverted index to a JSON file
save_inverted_index_to_json(inverted_index, os.path.join(output_dir, 'inverted_index.json'))    