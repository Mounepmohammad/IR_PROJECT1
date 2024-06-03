APP.py  : is a file connect with interfase and recive a query to matching it with datasets and diseplay docs retrived.
offline_processing_lifestyle : is a file to processing all docs in dataset lifestyle and store it in a new dectenay by joblib library to use it later.
offline_indexing_lifestyle : is a file use lifestyle dict to build a  TF_IDF matrix and vectorizer and store them in joblib library to use it later.
offline_processing_writing : is a file to processing all docs in dataset writing and store it in a new dectenay by joblib library to use it later.
offline_indexing_writing : is a file use writing dict to build a  TF_IDF matrix and vectorizer and store them in joblib library to use it later.
inverted_index :is a file to build an inverted index using dict , TF_IDF matrix , vectorizer to all terms in dataset dict.
lifestyle_service : is a file to take all query in qas.search and bild evaluating for all query by compare the retrived docs with relevant docs to calculate (precision,recall,MAP).
writing_service : is a file to take all query in qas.search and bild evaluating for all query by compare the retrived docs with relevant docs to calculate (precision,recall,MAP).
static,template : contain interface files (index.html,style.css).
evaluating folder : contain result for (precision , recall ,MAP) for tow datasets and for all querys.
