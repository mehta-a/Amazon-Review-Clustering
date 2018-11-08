from preprocess import get_file_lines, clean_text, remove_stopwords
from preprocess import apply_lemmatization, apply_stemming, extract_n_grams
from vectorizer import create_TF_IDF_matrix
from cluster import k_mean

test_file_name = '../data/Test Corpus/Test Corpus.txt'
stopwords_file = '../data/Stopwords/Basic Stopwords List.txt'

# 1. a
lines = get_file_lines(test_file_name)
print("***********Test-1***********")
print(lines)
print("****************************")

# 1. b
for i in range(len(lines)):
    lines[i] = clean_text(lines[i])
print("***********Test-2***********")
print(lines)
print("****************************")

# 1.c
for i in range(len(lines)):
    lines[i] = remove_stopwords(stopwords_file, lines[i], do_clean=True)
print("***********Test-3***********")
print(lines)
print("****************************")

# 1.d
for i in range(len(lines)):
    lines[i] = apply_stemming(lines[i])
    lines[i] = apply_lemmatization(lines[i])
print("***********Test-4***********")
print(lines)
print("****************************")

# 1.e
print("***********Test-5***********")
for line in lines:
    n_grams = extract_n_grams(line, 3)
    print(n_grams)
print("****************************")


# 1.f
print("***********Test-6***********")
_, mat = create_TF_IDF_matrix(test_file_name, doClean=True)
print(mat)
print("****************************")

# 1.g
print("***********Test-6***********")
_, mat = create_TF_IDF_matrix(test_file_name, doClean=True)
k = 2
n = 5
print("created tf-idf")
clusters, cluster_centroids = k_mean(mat, k, n)
print(clusters)
print("****************************")