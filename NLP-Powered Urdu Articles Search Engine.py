#!/usr/bin/env python
# coding: utf-8

# # NLP-Powered Urdu Articles Search Engine

import os
import re
from bs4 import BeautifulSoup
import numpy as np
from nltk.tokenize import word_tokenize

# Compile regex pattern to remove non-alphanumeric characters
patterns = re.compile(r'[^\w\s]+')

# Query list for testing the search engine
query_list = [
    'مجرم کو پھانسی دے دی گئی', 'احتجاجی دھرنے, جلوس اور ریلیاں',
    'انسداد تمباکو نوشی کے اقدامات اور آگاہی', 'وکلاء کی ہڑتال اور ریلیاں',
    'خفیہ ایجنسیاں اور جاسوس', 'احتساب عدالت کے مقدمات',
    'بچوں کے خلاف تشدد اور اغوا', 'عورتوں پر ظلم و تشدد , قتل اور اغوا',
    'انسانوں کی اسمگلنگ اور انسداد انسانی سمگلنگ کے اقدامات', 'مزدور طبقے کی خوشحالی کے اقدامات'
]

# Function to process a query and return relevant documents
def process_query(query):
    # Load stopwords
    with open('Urdu stopwords.txt', encoding='utf-8') as file:
        stop_words = file.read().split('\n')
    
    # Remove non-alphanumeric characters from the query
    query = re.sub(patterns, '', query)
    tokens = word_tokenize(query)
    keywords = [word for word in tokens if word not in stop_words]
    
    document_set = set()
    
    for word in keywords:
        term_id = term_IDS[word]
        with open('term_info.txt') as term_info_file:
            lines = term_info_file.read().split('\n')
            lines.pop()
            term_info = lines[term_id - 1].split('\t')
            with open('term_index.txt') as term_index_file:
                term_index_file.seek(int(term_info[1]), 0)
                index_line = term_index_file.readline().split('\t')
                index_line.pop()
                postings = [list(map(int, posting.split(':'))) for posting in index_line[1:]]
                
                doc_id_sum = 0
                for doc_id, freq in postings:
                    doc_id_sum += doc_id
                    document_set.add(doc_id_sum)
    
    document_list = sorted(list(document_set))
    return document_list, keywords

# Function to calculate term frequency (TF)
def calculate_tf(term, document):
    document_id = doc_IDS[document]
    term_id = term_IDS[term]
    
    with open('term_info.txt') as term_info_file:
        lines = term_info_file.read().split('\n')
        lines.pop()
        term_info = lines[term_id - 1].split('\t')
        
        with open('term_index.txt') as term_index_file:
            term_index_file.seek(int(term_info[1]), 0)
            index_line = term_index_file.readline().split('\t')
            index_line.pop()
            postings = [list(map(int, posting.split(':'))) for posting in index_line[1:]]
            
            doc_id_sum = 0
            for i, (doc_id, freq) in enumerate(postings):
                doc_id_sum += doc_id
                if doc_id_sum == document_id:
                    postings = postings[i:]
                    break
            
            term_positions = [postings[0][1]]
            cumulative_freq = postings[0][1]
            for doc_id, freq in postings[1:]:
                cumulative_freq += freq
                if doc_id == 0:
                    term_positions.append(cumulative_freq)
                else:
                    break
    
    return len(term_positions)

# Function to calculate inverse document frequency (IDF)
def calculate_idf(term):
    term_id = term_IDS[term]
    
    with open('term_info.txt') as term_info_file:
        lines = term_info_file.read().split('\n')
        lines.pop()
        term_info = lines[term_id - 1].split('\t')
    
    return np.log10(len(doc_IDS) / int(term_info[3]))

# Calculate TF-IDF scores for the queries
for query_id, query_text in enumerate(query_list, start=1):
    documents, keywords = process_query(query_text)
    document_scores = {}
    
    for document in documents:
        filename = [doc_name for doc_name in doc_IDS if doc_IDS[doc_name] == document][0]
        tf_idf_score = 0
        query_tf = 0
        tf_query_product = 0
        
        for keyword in keywords:
            tf_idf = (1 + np.log10(calculate_tf(keyword, filename))) * calculate_idf(keyword)
            query_tf = (1 + np.log10(keywords.count(keyword)))
            tf_query_product += tf_idf * query_tf
        
        document_scores[filename] = tf_query_product
    
    sorted_tf_idf_scores = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
    
    with open("tf_idf_score.txt", 'a') as output_file:
        rank = 1
        for doc_name, score in sorted_tf_idf_scores:
            output_file.write(f"{query_id}  {doc_name}  {rank}  {score}  run 1\n")
            rank += 1

# Function to calculate BM25 parameter p1
def calculate_p1(term):
    term_id = term_IDS[term]
    
    with open('term_info.txt') as term_info_file:
        lines = term_info_file.read().split('\n')
        lines.pop()
        term_info = lines[term_id - 1].split('\t')
    
    return np.log10((len(doc_IDS) + 0.5) / (int(term_info[3]) + 0.5))

# Load stopwords
with open('Urdu stopwords.txt', encoding='utf-8') as file:
    stop_words = file.read().split('\n')

# Calculate average document length
total_length = 0  
for doc_index, doc_filename in enumerate(os.listdir("C:/Users/"), start=1):
    with open(f'Documents/{doc_filename}', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    text = re.sub(patterns, '', soup.get_text())
    doc_IDS[doc_filename] = doc_index
    tokens = word_tokenize(text)
    words = [token for token in tokens if token not in stop_words]
    total_length += len(words)
average_length = total_length / len(doc_IDS)

# Function to calculate BM25 parameter K
def calculate_K(doc_filename):
    with open('Urdu stopwords.txt', encoding='utf-8') as file:
        stop_words = file.read().split('\n')
    
    with open(f'Documents/{doc_filename}', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    text = re.sub(patterns, '', soup.get_text())
    tokens = word_tokenize(text)
    words = [token for token in tokens if token not in stop_words]
    
    return 1.2 * ((1 - 0.75) + 0.75 * (len(words) / average_length))

# Function to calculate Okapi BM25 scores
def calculate_okapi_bm25(keywords, document_list):
    k2 = 10
    document_scores = {}
    
    for document in document_list:
        filename = [doc_name for doc_name in doc_IDS if doc_IDS[doc_name] == document][0]
        bm25_score = 0
        
        for keyword in keywords:
            bm25_score += calculate_p1(keyword) * (
                ((1 + 1.2) * calculate_tf(keyword, filename)) / (calculate_K(filename) + calculate_tf(keyword, filename))
            ) * (
                ((1 + k2) * (1 + np.log10(keywords.count(keyword)))) / (k2 + (1 + np.log10(keywords.count(keyword))))
            )
        
        document_scores[filename] = bm25_score
    
    return document_scores

# Calculate Okapi BM25 scores for the queries
for query_id, query_text in enumerate(query_list, start=1):
    documents, keywords = process_query(query_text)
    bm25_scores = calculate_okapi_bm25(keywords, documents)
    sorted_bm25_scores = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
    
    with open("okapi_bm25_score.txt", 'a') as output_file:
        rank = 1
        for doc_name, score in sorted_bm25_scores:
            output_file.write(f"{query_id}  {doc_name}  {rank}  {score}  run 1\n")
            rank += 1

# Collect background words for language model with Dirichlet Smoothing
background_words = []
with open('Urdu stopwords.txt', encoding='utf-8') as file:
    stop_words = file.read().split('\n')

total_words = []
total_word_count = 0
for doc_filename in os.listdir("C:/Users/"):
    with open(f'Documents/{doc_filename}', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    text = re.sub(patterns, '', soup.get_text())
    tokens = word_tokenize(text)
    words = [token for token in tokens if token not in stop_words]
    total_word_count += len(words)
    background_words.extend(words)

# Calculate Dirichlet smoothing parameter mu
mu = total_word_count / len(doc_IDS)

# Calculate Dirichlet smoothing scores for the queries
for query_id, query_text in enumerate(query_list, start=1):
    documents, keywords = process_query(query_text)
    dirichlet_scores = {}
    
    for document in documents:
        filename = [doc_name for doc_name in doc_IDS if doc_IDS[doc_name] == document][0]
        
        with open(f'Documents/{filename}', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, "html.parser")
        text = re.sub(patterns, '', soup.get_text())
        tokens = word_tokenize(text)
        words = [token for token in tokens if token not in stop_words]
        
        document_score = 1
        lambda_param = len(words) / (len(words) + mu)
        
        for keyword in keywords:
            document_score *= (
                (lambda_param * (words.count(keyword) / len(words))) +
                ((1 - lambda_param) * (background_words.count(keyword) / len(background_words)))
            )
        
        dirichlet_scores[filename] = document_score
    
    sorted_dirichlet_scores = sorted(dirichlet_scores.items(), key=lambda item: item[1], reverse=True)
    
    with open("dirichlet_smoothing_score.txt", 'a') as output_file:
        rank = 1
        for doc_name, score in sorted_dirichlet_scores:
            output_file.write(f"{query_id}  {doc_name}  {rank}  {score}  run 1\n")
            rank += 1

# # THE END
