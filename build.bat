#!/bin/bash
conda env create -f environment.yml
cd python3-svmlight/
python3 setup.py install
cd ..
python -c "import nltk;nltk.download('punkt');nltk.download('stopwords')"
