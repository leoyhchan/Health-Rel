import os
import re
import pickle
import argparse
import svmlight
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn import preprocessing
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

STOPWORDS = set(stopwords.words("english"))
NEW_WORDS = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","http","layout","cell","www","dwlayoutemptycell","div"]
STOPWORDS = STOPWORDS.union(STOPWORDS)

def svm_parse(filename):
    features,target = load_svmlight_file(filename)
    _,nfeatures = features.shape
    
    it = 0
    for cl in target:
        doc_features = []
        for i in range(nfeatures):
            doc_features.append((float(i+1),features[it,i]))        
        it+=1
        yield (int(cl),doc_features)

def word_features(doc, vectorizer):
    vector = vectorizer.transform([doc])
    doc_to_list = list(vector.toarray()[0])
    maximum = max(doc_to_list)
    if maximum:
        for val in doc_to_list:
                index = doc_to_list.index(val)
                doc_to_list[index] = val/maximum
    return doc_to_list

def features_calc(corpus, vectorizer):
    for doc in corpus:
        doc_features = word_features(doc, vectorizer)
        yield doc_features

def generate_vocabulary(corpus, min_df, dataset):
    vectorizer = None

    if dataset != "CLEF":
        vectorizer = CountVectorizer(min_df=min_df)
        vectorizer.fit(corpus)
    
    return vectorizer

def normalize_text(line, stop_words, dataset):
    # Remove punctuations
    line = re.sub('[^a-zA-Z]', ' ', line)
    # Convert to lowercase
    line = line.lower()
    # remove tags
    line = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",line)
    # remove special characters and digits
    line = re.sub("(\\d|\\W)+"," ",line)
    ## Convert to list from string
    line = line.split()
    
    ## Removing stopwords
    if dataset != "Sondhi":
        line = [word for word in line if not word in stop_words]
    
    line = " ".join(line)
    return line

def preprocess_text(filename, stop_words, dataset):
    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(),'html5lib') # requests.get(url), when the service is implemented
        text = soup.get_text()
        output = text.split("\n")
        lines = []
        for line in output:
                line = normalize_text(line, stop_words, dataset)
                lines.append(line)
        doc= " ".join(lines)
        return doc

def generate_corpus(docs, stop_words, dataset):
    corpus = []
    for doc in docs:
            doc = preprocess_text(doc, stop_words, dataset)
            corpus.append(doc)
    return corpus

def train_test_split_sondhi():
    path1 = './reliable'
    os.chdir(path1)
    arr1 = os.listdir('.')
    path2 = '../unreliable'
    os.chdir(path2)
    arr2 = os.listdir('.')
    X = []
    Y = []
    for rel,unrel in zip(arr1,arr2):
            os.chdir('../reliable')
            X.append('./reliable/'+rel)
            Y.append(-1)
            os.chdir('../unreliable')
            X.append('./unreliable/'+unrel)
            Y.append(1)
    os.chdir("../")
    return np.array(X), np.array(Y)

def train(dataset):
    if dataset == "CLEF":
        pass
    
    elif dataset == "Sondhi":
        X, Y = train_test_split_sondhi()
        min_df = 1
        cost_factor = 3
    
    elif dataset == "Schwarz":
        pass
        min_df = 0.5
    
    else:
        print("No such dataset. Expected: CLEF, Sondhi or Schwarz")

    data = X
    corpus = generate_corpus(data, STOPWORDS, dataset)
    vectorizer = generate_vocabulary(corpus, min_df, dataset)
    
    if vectorizer:
        pickle.dump(vectorizer, open("models/"+dataset+"/vocabulary.pkl", "wb"))

    data = features_calc(corpus, vectorizer)
    target = Y
    scaler_x = preprocessing.StandardScaler().fit(list(data)) # Standardisation
    data = scaler_x.transform(list(data))
    pickle.dump(scaler_x, open("models/"+dataset+"/scaler.pkl", "wb"))

    if dataset != "CLEF":
        dump_svmlight_file(data, target, 'train.txt')
        train = svm_parse('train.txt')
        model = svmlight.learn(list(train), type='classification', verbosity=0, costratio=cost_factor)
        svmlight.write_model(model, "models/"+dataset+"/model.dat")

    else: 
        pass

    os.remove('train.txt')

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
args = parser.parse_args()
dataset = args.dataset

train(dataset)