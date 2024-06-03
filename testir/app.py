import os
import joblib
import json
import nltk
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def load_data(dataset):
    output_dir = 'output'
    # if(dataset == 'dataset1'):
       
    #    docs_dict = joblib.load(os.path.join(output_dir, 'docs_dict.joblib'))
    #    tfidf_matrix = joblib.load(os.path.join(output_dir, 'tfidf_matrix.joblib'))
    #    vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer.joblib'))
    #    return docs_dict, tfidf_matrix, vectorizer
    # else:
    docs_dict = joblib.load(os.path.join(output_dir, 'docs_dict_writing.joblib'))
    print(docs_dict[0])
    tfidf_matrix = joblib.load(os.path.join(output_dir, 'tfidf_matrix_writing.joblib'))
    vectorizer = joblib.load(os.path.join(output_dir, 'vectorizer_writing.joblib'))
    return docs_dict, tfidf_matrix, vectorizer


def represent_query_as_vector(query, vectorizer):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector

def retrieve_top_docs(query_vector, tfidf_matrix, top_n=10):
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_doc_indices = similarities.argsort()[-top_n:][::-1]
    return top_doc_indices

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    dataset = request.form['dataset']
    
    docs_dict, tfidf_matrix, vectorizer = load_data(dataset)
    query_vector = represent_query_as_vector(query, vectorizer)
    top_doc_indices = retrieve_top_docs(query_vector, tfidf_matrix)
    
    results = [(list(docs_dict.keys())[i], docs_dict[list(docs_dict.keys())[i]]) for i in top_doc_indices]
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)