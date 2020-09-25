import numpy as np
import pandas as pd
import requests
import csv
import os
import svmlight
import random
import re
import nltk
import time 
import matplotlib.pyplot as plt
import pickle
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
from w3lib.html import remove_tags
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.decomposition import PCA
from pandas import DataFrame, read_csv

COMM_LIST = ["buy", "sell", "cheap", "deal" ,"free", "guarantee", "shop", "price", "business", "trade", "interests", \
             "expensive", "commerce", "ecommerce", "advertising", "sale", "market", "profitable", "lucrative", "ad", \
             "advertisement", "seller", "vendor", "tradesman", "mercature", "vendor", "production", "transaction", \
             "buying", "selling", "leasing", "industry", "company", "companies", "deal", "goods", "exchangeable", \
             "retail price", "sold out", "in storage", "warehouse", "warrant", "value", "terms", "terms of payment"]

def weighted_accuracy(bias,tn,tp,fn,fp):
        return (bias*tp+tn)/(bias*(tp+fn)+tn+fp) 

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

def transform_to_svm_light_format(data_train_res,target_train_res,data_test_res,target_test_res):
        train = []
        test = []

        for data,target in zip(data_train_res,target_train_res):
                mod_data = []
                list_of_groups = zip(*(iter(data),)*2)

                for element in list_of_groups:
                        mod_data.append(element)
                train.append((target,mod_data))

        for data,target in zip(data_test_res,target_test_res):
                mod_data = []
                list_of_groups = zip(*(iter(data),)*2)

                for element in list_of_groups:
                        mod_data.append(element)
                test.append((target,mod_data))
        
        return train,test 

def append_word_features(doc_features,words):
        for val in words:
                doc_features.append(val)

def word_features(doc,vectorizer):
        vector = vectorizer.transform([doc])
        doc_to_list = list(vector.toarray()[0])
        maximum = max(doc_to_list)
        if maximum:
                for val in doc_to_list:
                        index = doc_to_list.index(val)
                        doc_to_list[index] = val/maximum
        return doc_to_list

def count_commercial_keywords(filename,comm_list,doc):
        commercial_words = 0
        with open(filename,encoding="utf-8",errors="ignore") as reader:
                soup = BeautifulSoup(reader.read(),'html5lib') # requests.get(url), when the service is implemented
                text = soup.get_text()
                output = text.split(" ")
                for line in output:
                        for term in comm_list:
                                if term in line:
                                        commercial_words += 1
                doc = doc.split(" ")
        # print("%s: %d" %(filename,commercial_words))
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

def count_links(filename,z1,comm_list):
        with open(filename,encoding="utf-8",errors="ignore") as reader:
            soup = BeautifulSoup(reader.read(),'html5lib') # requests.get(url), when the service is implemented
            links = Counter([x.get('href') for x in soup.findAll('a')])
            links = links.most_common()
            total = 0
            external = 0
            contact = 0
            privacy = 0
            commercial = 0
            c_list = ["ContactUs", "ContactJudy", "Contact Us", "Contact Judy"]
            p_list = ["PrivacyInformation", "PrivacyPolicy", "PrivacyStatement", "PrivacySecured", "Privacy Information", "Privacy Policy", "Privacy Statement", "Privacy Secured"]
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


def features_calc(docs,corpus,vectorizer,stop_words):
        count = 0
        z1 = 200
        comm_list = ["buy","sell","cheap","deal","free","guarantee","shop","price"]
        print_flag = True
        for filename,doc in zip(docs,corpus):
                count = count_links(filename,z1,comm_list)
                commercial_words = count_commercial_keywords(filename,comm_list,doc)
                commercial_links = count_commercial_links(filename, z1)
                # words = word_features(doc,vectorizer)
                # doc_features = []
                doc_features = [count[0],count[1],count[2],count[3],count[4], commercial_links ,commercial_words]
                # append_word_features(doc_features, words)
                
                if print_flag:
                        print("Longitud de las features->",len(doc_features))
                        print_flag = False

                yield doc_features

def generate_vocabulary(corpus):
        vectorizer = CountVectorizer()
        vectorizer.fit(corpus)
        return vectorizer

def __normalize_text(line,stop_words):
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

        ########################################################
        ######## COMMENT THIS LINE FOR KEEPING STOPWORDS #######
        ########################################################

        line = [word for word in line if not word in stop_words]
        line = " ".join(line)
        return line

def preprocess_text(filename,stop_words):
        with open(filename,encoding="utf-8",errors="ignore") as reader:
                soup = BeautifulSoup(reader.read(),'html5lib') # requests.get(url), when the service is implemented
                text = soup.get_text()
                output = text.split("\n")
                lines = []
                for line in output:
                      line = __normalize_text(line,stop_words)
                      lines.append(line)
                doc= " ".join(lines)
                return doc

def generate_corpus(docs,stop_words):
        corpus = []
        for doc in docs:
                doc = preprocess_text(doc,stop_words)
                corpus.append(doc)
        return corpus


def generate_train_and_test_clef():
        X = []
        Y = []

        with open('CLEF2018_qtrust_20180914_cleaned.txt',newline='') as assestments:
                reader = csv.reader(assestments,delimiter=' ')
                for row in reader:
                        web = row[2]
                        rating = float(row[3])
                        rating = int(rating)

                        if rating == 0 or rating == 1 or rating == 2 or rating == 3: # Trustworthiness
                                for filename in Path('./clef2018collection').rglob(web):
                                        if filename not in X:
                                                X.append(filename)
                                        break
                                
                                if len(Y) == len(X)-1:
                                        Y.append(1)

                        elif rating == 7 or rating == 8 or rating == 9 or rating == 10: 
                                for filename in Path('./clef2018collection').rglob(web):
                                        if filename not in X:
                                                X.append(filename)
                                        break
                                
                                if len(Y) == len(X)-1:  
                                        Y.append(-1)

        print("END")
        print("======================================")        
        return np.array(X), np.array(Y)

def generate_train_and_test_morris():
        df = pd.read_excel("web_credibility_relabeled.xlsx")
        ratings = df['Likert Rating']
        urls = df['URL']
        root = os.getcwd()
        path = './CachedPages'
        os.chdir(path)
        cached_pages_dir = os.getcwd()
        X = []
        Y = []
        
        for url,rating in zip(urls,ratings):
                try:
                        url = url.replace('http://','')
                        url = url.split('/')
                        if url[-1]: # urls del estilo 'www.adamofficial.com/us/home'
                                url = '/'.join(url[:-1])
                                os.chdir(url)
                                f = [f for f in os.listdir() if re.match(url[-1]+'*',f) and os.path.isfile(f)]
                        else:
                                url = '/'.join(url)
                                os.chdir(url)
                                f = [f for f in os.listdir() if re.match('index*',f) and os.path.isfile(f)]
                        
                        X.append(os.path.join(os.getcwd(),f[0]))
                        Y.append(rating)
                        os.chdir(cached_pages_dir)

                except:
                        pass
        
        os.chdir(root)
        return np.array(X), np.array(Y)

def generate_train_and_test():
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

X, Y = generate_train_and_test() # ECIR
# X_train, Y_train = generate_train_and_test_morris() # MORRIS
# X,Y = generate_train_and_test_clef() # CLEF E HEALTH
# print(len(X))
# print(len(Y))
stop_words = set(stopwords.words("english"))
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","http","layout","cell","www","dwlayoutemptycell","div"]
stop_words = stop_words.union(new_words)
skf = StratifiedKFold(n_splits=5) # stratified k-fold preserves the percentage of samples for each class
np.random.seed(1) # reproducibility

for bias in range(3): # bias: cost-factor, by which training errors on positive examples (unreliable) outweight errors on negative ones (reliable). Missing an unreliable one is more dangerous

        accuracies = []
        it = 1
        f1_micro = []
        f1_rel = []
        f1_unrel = []

        for train_index, test_index in skf.split(X,Y):

                data_train   = X[train_index]
                corpus_train = generate_corpus(data_train, stop_words)
                vectorizer = generate_vocabulary(corpus_train) # for each fold we reset vocabulary associated to training set
                # os.chdir("/opt/catenae/model_data")
                # pickle.dump(vectorizer, open("vocabulary.pkl","wb"))
                data_train = features_calc(data_train, corpus_train, vectorizer, stop_words)
                target_train = Y[train_index]
                unique,counts = np.unique(target_train,return_counts=True)
                dictionary = dict(zip(unique, counts))
                print(dict(zip(unique, counts)))
                # data_train_res,target_train_res = smote_tomek(np.array(list(data_train)),np.array(target_train)) # Hybrid sampling

                ## Scaling
                # list_data_train = list(data_train)
                # scaler_x = preprocessing.StandardScaler().fit(list_data_train)
                # pickle.dump(scaler_x, open("scaler.pkl","wb"))
                # data_train = scaler_x.transform(list_data_train)
                
                data_train = np.array(list(data_train))
                nsamples,nx = data_train.shape
                data_train = data_train.reshape((nsamples,nx))
                dump_svmlight_file(data_train, target_train, 'train.txt')

                unique,counts = np.unique(target_train, return_counts=True)
                print(dict(zip(unique, counts)))

                data_test = X[test_index]
                corpus_test = generate_corpus(data_test, stop_words)
                data_test = features_calc(data_test, corpus_test, vectorizer, stop_words)
                target_test  = Y[test_index]
                unique,counts = np.unique(target_test,return_counts=True)
                print(dict(zip(unique, counts)))

                ##Scaling
                # data_test = scaler_x.transform(list(data_test))
                
                data_test = np.array(list(data_test))
                nsamples,nx = data_test.shape
                data_test = data_test.reshape((nsamples,nx))
                dump_svmlight_file(data_test, target_test, 'test.txt')
                        
                train = svm_parse('train.txt')
                aux = svm_parse('test.txt')

                test = []
                val = []
                for element in aux:
                        lst = list(element)
                        val.append(lst[0])
                        lst[0] = 0
                        element = tuple(lst)
                        test.append(element)

                print("Training it=", it,"cost-factor=",bias+1) 

                ## Costratio = cost-learning
                model = svmlight.learn(list(train), type='classification', verbosity=0, costratio=bias+1) ## CAMBIAR BIAS//COSTRATIO !!!
                # svmlight.write_model(model,'clef_model.dat')

                predictions = svmlight.classify(model, test)
                print("Predicting it=", it,"cost-factor=",bias+1) 

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                for a, b in zip(val,predictions):
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

                accuracies.append(weighted_accuracy(bias+1,tn,tp,fn,fp)*100) ## CAMBIAR BIAS//COSTRATIO !!!

                predictions = np.array(predictions)
                predictions[predictions<0] = -1
                predictions[predictions>0] = 1
                f1_micro.append(f1_score(val,predictions,average='micro'))
                cl = f1_score(val, predictions, average=None)
                f1_rel.append(cl[0])
                f1_unrel.append(cl[1])
                it+=1

        print("The accuracy is", np.mean(accuracies))
        print("The f1-score is", np.mean(f1_micro)) # micro: calculates metrics totally by counting the total true positives, false negatives and false positives
        print("The f1-score per class is", np.mean(f1_rel), np.mean(f1_unrel)) # None: returns scores for each class

