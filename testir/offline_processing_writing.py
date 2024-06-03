import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
from nltk.stem import WordNetLemmatizer
import re
import time

# تحميل الموارد اللازمة من nltk
# nltk.download('punkt')
# nltk.download('stopwords')

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


start = time.time()

collection_df = pd.read_csv(r'C:/Users/DELL/Desktop/IR/lotte/lotte/writing/test/collection.tsv', sep='\t', header=None, names=['pid', 'text'])
print("file loaded ! this operation take",time.time()-start,"seconds")
# print("data shape" ,collection_df.shape)
docs_dict1 = collection_df.set_index('pid')['text'].to_dict()
docs_dict2 = collection_df.set_index('pid')['text'].to_dict()
for i in range(100000):
    prossecd = preprocess_text(docs_dict1[i])
    docs_dict2[i] = prossecd
    
print(docs_dict1[0],"this is2 \n",docs_dict1[1])    
print(docs_dict2[0],"this is2 \n",docs_dict2[1])

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
start =time.time()

joblib.dump(docs_dict2, os.path.join(output_dir, 'docs_dict_writing.joblib'))
print("file loaded ! this operation take",time.time()-start,"seconds")