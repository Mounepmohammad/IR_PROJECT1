import os
import joblib 
import time

import joblib
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

output_dir = 'output'
docs_dict2 = joblib.load(os.path.join(output_dir, 'docs_dict.joblib'))  

all_text =[]
for x in range(268893):
    if (x < 119594 or x > 119598)and (x < 132633 or x > 132640) :
        all_text += [docs_dict2[x]]
    else :
        continue
     
def build_tfidf_matrix(docs_dict):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_dict.values())
    print('build_tfidf_matrix')
    return vectorizer, tfidf_matrix
        

def process_term(term_index, term, tfidf_matrix, doc_ids):
    term_postings = []
    for doc_index in range(tfidf_matrix.shape[0]):
        tfidf_value = tfidf_matrix[doc_index, term_index]
        if tfidf_value > 0:
            doc_id = doc_ids[doc_index]
            term_postings.append((doc_id, tfidf_value))
    return term, term_postings

def build_inverted_index(vectorizer,tfidf_matrix, doc_ids, n_jobs=-1):
    inverted_index = defaultdict(list)
    terms = vectorizer.get_feature_names_out()
    
    num_terms = len(terms)
    print(f"Number of terms: {num_terms}")
    
    # استخدام joblib لتوازي معالجة المصطلحات مع tqdm لإظهار شريط التقدم
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_term)(term_index, term, tfidf_matrix, doc_ids) 
        for term_index, term in tqdm(enumerate(terms), total=num_terms, desc="Processing terms")
    )
    
    for term, term_postings in results:
        if term_postings:
            inverted_index[term].extend(term_postings)
    
    return inverted_index


start = time.time()

vectorizer, tfidf_matrix = build_tfidf_matrix(all_text)
print(vectorizer)
print(pd.DataFrame(tfidf_matrix))

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
start =time.time()

joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.joblib'))
joblib.dump(tfidf_matrix, os.path.join(output_dir, 'tfidf_matrix.joblib'))
print("file loaded ! this operation take",time.time()-start,"seconds")
