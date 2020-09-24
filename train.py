import os
import re
import sys
import time
import pickle
import argparse
import svmlight
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file


STOPWORDS = set(stopwords.words("english"))
NEW_WORDS = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", \
             "http","layout","cell","www","dwlayoutemptycell","div"]
STOPWORDS = STOPWORDS.union(STOPWORDS)
COMM_LIST = ["buy", "sell", "cheap", "deal" ,"free", "guarantee", "shop", "price", "business", "trade", "interests", \
             "expensive", "commerce", "ecommerce", "advertising", "sale", "market", "profitable", "lucrative", "ad", \
             "advertisement", "seller", "vendor", "tradesman", "mercature", "vendor", "production", "transaction", \
             "buying", "selling", "leasing", "industry", "company", "companies", "deal", "goods", "exchangeable", \
             "retail price", "sold out", "in storage", "warehouse", "warrant", "value", "terms", "terms of payment"]   

def weighted_accuracy(bias, tn, tp, fn, fp):
    return (bias*tp+tn)/(bias*(tp+fn)+tn+fp) 

def evaluate(predictions, val, cost_factor):
    tp, tn, fp, fn = 0, 0, 0, 0

    for a, b in zip(val, predictions):
        if np.sign(a) == np.sign(b): # true
            if np.sign(a) == -1:
                    tn +=1
            else:
                    tp += 1
        else: # false
            if np.sign(a) == 1:
                    fn += 1
            else: 
                    fp += 1

    acc = weighted_accuracy(cost_factor, tn, tp, fn, fp)*100

    predictions = np.array(predictions)
    predictions[predictions<0] = -1
    predictions[predictions>0] = 1
    f1_micro = f1_score(val, predictions, average='micro')
    cl = f1_score(val, predictions, average=None)

    return acc, f1_micro, cl[0], cl[1]


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


def count_commercial_keywords(filename, doc):
    commercial_words = 0

    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(), 'html5lib')
        text = soup.get_text()
        output = text.split(" ")
        
        for line in output:
            for term in COMM_LIST:
                if term in line:
                    commercial_words += 1
        
        doc = doc.split(" ")
    
    return commercial_words/len(doc)

def count_commercial_links(filename, z1):
    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(),'html5lib')
        links = Counter([x.get('href') for x in soup.findAll('a')])
        links = links.most_common()
        commercial = 0
        
        for item in links:
            if item[0]: 
                if any(ext in item[0] for ext in COMM_LIST):
                    commercial += item[1]
    
    return commercial/z1

def count_links(filename, z1):
    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(),'html5lib')
        links = Counter([x.get('href') for x in soup.findAll('a')])
        links = links.most_common()
        total = 0
        external = 0
        contact = 0
        privacy = 0
        c_list = ["ContactUs", "ContactJudy", "Contact Us", "Contact Judy"]
        p_list = ["PrivacyInformation", "PrivacyPolicy", "PrivacyStatement", "PrivacySecured", "Privacy Information", \
                  "Privacy Policy", "Privacy Statement", "Privacy Secured"]
        
        for item in links:
            total += item[1]
            if item[0]: 
                if item[0].startswith(('http','ftp','www')):
                    external += item[1]
                if any(ext in item[0] for ext in c_list):
                    contact = 1
                if any(ext in item[0] for ext in p_list):
                    privacy = 1
        
        internal = total - external
    
    return total/z1, external/z1, internal/z1, contact, privacy

def features_calc(docs, corpus, vectorizer, features):
    z1 = 200 # empirical observed standardisation value

    for filename, doc in zip(docs, corpus):
        doc_features = []
        
        if features == "link" or features == "comm" or features == "all1" or features == "all2":
            links_counts = count_links(filename, z1)
            doc_features.extend(links_counts)

        if features == "comm" or features == "all1" or features == "all2":
            commercial_links = count_commercial_links(filename, z1)
            commercial_words = count_commercial_keywords(filename, doc)
            doc_features.extend([commercial_links, commercial_words])

        if features == "words1" or features == "words2" or features == "all1" or features == "all2":
            words = word_features(doc, vectorizer)
            doc_features.extend(words)

        yield doc_features

def generate_vocabulary(corpus, min_df, dataset):
    vectorizer = None

    if dataset != "CLEF":
        vectorizer = CountVectorizer(min_df=min_df)
        vectorizer.fit(corpus)
    
    return vectorizer

def normalize_text(line, stop_words, features):
    line = re.sub('[^a-zA-Z]', ' ', line) # remove punctuations
    line = line.lower() # convert to lowercase
    line = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", line) # remove tags
    line = re.sub("(\\d|\\W)+"," ",line) # remove special characters and digits
    line = line.split() # convert to list from string
    
    if features != "words2" and features != "all2":
        line = [word for word in line if not word in stop_words] # remove stopwords
    
    line = " ".join(line)
    return line

def preprocess_text(filename, stop_words, dataset):
    with open(filename,encoding="utf-8",errors="ignore") as reader:
        soup = BeautifulSoup(reader.read(),'html5lib')
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

def data_sondhi():
    root = os.getcwd()
    path1 = './datasets/Sondhi/reliable'
    os.chdir(path1)
    arr1 = os.listdir('.')
    path2 = '../unreliable'
    os.chdir(path2)
    arr2 = os.listdir('.')
    X = []
    Y = []
    
    for rel,unrel in zip(arr1,arr2):
        os.chdir('../reliable')
        X.append('./datasets/Sondhi/reliable/'+rel)
        Y.append(-1)
        os.chdir('../unreliable')
        X.append('./datasets/Sondhi/unreliable/'+unrel)
        Y.append(1)
    
    os.chdir(root)
    return np.array(X), np.array(Y)

def feature_set():
    ext = False
    option = 0
    
    while not ext:
        print ("1. Link-based")
        print ("2. Link + Commercial")
        print ("3. Word-based (removing stopwords)")
        print ("4. Word-based (keeping stopwords)")
        print ("5. All (removing stopwords)")
        print ("6. All (keeping stopwords)")
        print ("7. Exit experiment")
        option = int(input("Choose a feature set: "))
    
        if option == 1:
            return "link"
        elif option == 2:
            return "comm"
        elif option == 3:
            return "words1"
        elif option == 4:
            return "words2"
        elif option == 5:
            return "all1"
        elif option == 6:
            return "all2"
        elif option == 7:
            sys.exit()
        else:
            print ("Not valid option")

def adapt_test_to_svmlight(aux):
    test = []
    val = []
    
    for element in aux:
        lst = list(element)
        val.append(lst[0])
        lst[0] = 0
        element = tuple(lst)
        test.append(element)
    
    return test, val


def train(dataset, dump, cost_factor):
    if dataset == "CLEF":
        n = 5
    
    elif dataset == "Sondhi":
        X, Y = data_sondhi()
        min_df = 1
        n = 5
    
    elif dataset == "Schwarz":
        min_df = 0.5
        n = 2

    skf = StratifiedKFold(n_splits=n)
    it = 1

    features = feature_set()
    accuracies, f1_l, f1_rel_l, f1_unrel_l = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        ts = str(time.time())

        data_train = X[train_index]
        corpus_train = generate_corpus(data_train, STOPWORDS, features)
        
        vectorizer = generate_vocabulary(corpus_train, min_df, dataset)
        if dump == "yes":
            
            if not os.path.exists('./models/'+dataset):
                os.makedirs('./models/'+dataset)
            
            pickle.dump(vectorizer, open("models/"+dataset+"/vocabulary_"+ts+".pkl", "wb"))

        data_train = features_calc(data_train, corpus_train, vectorizer, features)
        target_train = Y[train_index]
        list_data_train = list(data_train)
        scaler_x = preprocessing.StandardScaler().fit(list_data_train) # standardisation
        data_train = scaler_x.transform(list_data_train)
        if dump == "yes":
            pickle.dump(scaler_x, open("models/"+dataset+"/scaler_"+ts+".pkl", "wb"))

        data_test = X[test_index]
        corpus_test = generate_corpus(data_test, STOPWORDS, features)
        data_test = features_calc(data_test, corpus_test, vectorizer, features)
        target_test  = Y[test_index]
        data_test = scaler_x.transform(list(data_test))

        if not os.path.exists('./aux'):
            os.makedirs('./aux')

        dump_svmlight_file(data_train, target_train, './aux/train_'+ts+'.txt')
        dump_svmlight_file(data_test,target_test,'./aux/test_'+ts+'.txt')

        train = svm_parse('./aux/train_'+ts+'.txt')
        aux = svm_parse( './aux/test_'+ts+'.txt')
        test, labels = adapt_test_to_svmlight(aux)

        print("Training for iteration =", it)

        model = svmlight.learn(list(train), type='classification', verbosity=0, costratio=cost_factor)
        if dump == "yes":
            svmlight.write_model(model, "models/"+dataset+"/model_"+ts+".dat")
        
        predictions = svmlight.classify(model, test)
        print("Predicted for iteration =", it)
        acc, f1_micro, f1_rel, f1_unrel = evaluate(predictions, labels, cost_factor)
        accuracies.append(acc)
        f1_l.append(f1_micro)
        f1_rel_l.append(f1_rel)
        f1_unrel_l.append(f1_unrel)
        it += 1
        os.remove('./aux/train_'+ts+'.txt')
        os.remove('./aux/test_'+ts+'.txt')

    print("The mean accuracy is", np.mean(accuracies))
    print("The f1-score is", np.mean(f1_l)) # micro: calculates metrics totally by counting the total true positives, false negatives and false positives
    print("The credible f1-score is", np.mean(f1_rel_l)) # None: returns scores for each class
    print("The non-credible f1-score is", np.mean(f1_unrel_l))

np.random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["CLEF", "Sondhi", "Schwarz"]) # CLEF, SONDHI, SCHWARZ
parser.add_argument("cost_factor", nargs='?', choices=["1", "2", "3"], default = "1") # 1, 2, 3
parser.add_argument("dump", nargs='?', choices=["yes", "no"], default = 'yes') # YES, NO 
args = parser.parse_args()
dataset = args.dataset
cost_factor = int(args.cost_factor)
dump = args.dump

train(dataset, dump, cost_factor)

