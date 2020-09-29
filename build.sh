#!/bin/bash
pip install -r requirements.txt
git clone https://github.com/jeremija/python3-svmlight.git
cd python3-svmlight/
chmod +x setup.py
python3 setup.py build
python3 setup.py install
cd ..
python -c 'import nltk;nltk.download("'punkt'");nltk.download("'stopwords'")'
