# Necessary imports
import string # Require for clean text


from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# I. a) Read a text file using the path to the text file as the input parameter.
def get_file_lines(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        return lines


# I. b) Remove punctuations, numbers and special characters given a document as input.
def clean_text(text):
    text = ''.join([x for x in text if x not in [*string.punctuation, *string.digits, *['\n','\r','\t']]])
    return text


# I. c) Remove stopwords given a document as the input parameter -
# a basic list of stopwords is provided, you can use more extensive lists from the
# internet (please cite the link in your comments if you do).
def remove_stopwords(stopwords_file, text_to_clean, do_clean=False):
    stp_words = get_file_lines(stopwords_file)
    # Remove punctuation, digits, or extra special characters from stopwords file.
    if do_clean:
        stp_words = list(map(clean_text, stp_words))
    resp = ' '.join([x for x in text_to_clean.split(' ') if x.lower() not in stp_words])
    return resp


# I. d) Apply stemming or lemmatisation given a document as the input parameter
# (functionality from any pre-existing package can be used to do this).
# Stemming:
def apply_stemming(doc):
    stemmed_list = [stemmer.stem(x) for x in doc.split()]
    stemmed_doc = ' '.join([x for x in stemmed_list])
    return stemmed_doc


# Lemmatize:
def apply_lemmatization(doc):
    lemma_list = [lemmatizer.lemmatize(x, pos='v') for x in doc.split()]
    lemma_doc = ' '.join([x for x in lemma_list])
    return lemma_doc


# I. e) Extract n-grams given a document as the input parameter
# (functionality from any pre-existing package can be used to do this).
def extract_n_grams(doc, n):
    n_grams = []
    word_list = doc.split()
    if n > 0:
        if len(word_list) < n:
            n_grams = ' '.join(word_list)
        for i in range(len(word_list)-n):
            gram = ' '.join(word_list[i:i+n])
            n_grams.append(gram)
    return n_grams