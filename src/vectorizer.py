# I. f) Create a tf-idf matrix from scratch given a corpus as the input parameter
# (do not use any pre-exisiting packages that offer functionality to do this directly).
import math # Require for taking log

import pandas as pd
import numpy as np

from preprocess import *

stop_words_file = '../data/Stopwords/Basic Stopwords List.txt'

# Util function to add an element in a python dictionary
def add_in_dict(_dict, _key):
    if _key in _dict.keys():
        _dict[_key] +=1
    else:
        _dict[_key] = 1
    return _dict


# Util function to normalize, lognormalize, and inverse normalize
def normalize(_dict, total):
    _dict = {k: v / total for k, v in _dict.items()}
    return _dict


def log_normalize(_dict):
    _dict = {k: math.log10(v) for k, v in _dict.items()}
    return _dict


def inverse_normalize(_dict, total):
    _dict = {k: total/v for k, v in _dict.items()}
    return _dict


# Util function to create term frequency and inverse document frequency given a list of docs
def create_TF_IDF_dict(docs):
    doc_map = {}
    TF_dict = {}
    IDF_dict = {}
    if len(docs)!=0:
        doc_counter = 1
        for doc in docs:
            if len(doc)!=0:
                doc_map[doc_counter] = doc   # storing <doc_id-doc_text> mapping
                word_freq_in_doc = {}   # dictionary to store term freq in doc
                for word in doc.split(' '):
                    IDF_dict = add_in_dict(IDF_dict, word)    # store in global list of terms
                    word_freq_in_doc = add_in_dict(word_freq_in_doc, word)
                # print("doc wise:", doc)
                TF_dict[doc_counter] = normalize(word_freq_in_doc, len(doc)) # update TF Dictionary
            doc_counter +=1
        IDF_dict = inverse_normalize(IDF_dict, len(docs))  # normalize idf doctionary
        IDF_dict = log_normalize(IDF_dict)  # log normalize idf doctionary
    return doc_map, TF_dict, IDF_dict


# Function to create tf-ifd matrix given term frequencies, inverse document frequency and document map.
# This is to be consumed along with method `create_TF_IDF_dict`
def calculate_TF_IDF_matrix(doc_map, TF_dict, IDF_dict):
    mat_tf_idf = pd.DataFrame(index=doc_map.keys(), columns=IDF_dict.keys())
    mat_tf_idf = mat_tf_idf.fillna(0)

    for doc in doc_map.keys():
        for word in IDF_dict.keys():
            tf = 0
            tf_vals = TF_dict[doc]
            if word in tf_vals.keys():
                tf = tf_vals[word]
            idf = IDF_dict[word]
            mat_tf_idf.loc[doc, word] = tf * idf
    print(mat_tf_idf.head())
    return np.array(mat_tf_idf)


#  Clean the documents
def clean_docs(docs):
    new_docs = []
    for doc in docs:
        text = clean_text(doc)
        no_stop_text = remove_stopwords(stop_words_file, text)
        stem_text = apply_stemming(no_stop_text)
        lemma_text = apply_lemmatization(stem_text)
        new_docs.append(lemma_text)
    return new_docs


# function to create tf-idf matrix given a corpus; doClean if set true cleans the documents as well.
def create_TF_IDF_matrix(corpus, doClean=False, verbose=False):
    corp_lines = get_file_lines(corpus)
    if doClean:
        corp_lines = clean_docs(corp_lines)
    if verbose:
        print(corp_lines)
    # Assuming each line is a doc
    doc_map, TF_dict, IDF_dict = create_TF_IDF_dict(corp_lines)
    # print(doc_map, TF_dict, IDF_dict)
    # generate TF*IDF values
    mat = calculate_TF_IDF_matrix(doc_map, TF_dict, IDF_dict)
    return doc_map, mat