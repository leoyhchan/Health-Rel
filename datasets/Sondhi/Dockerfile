FROM catenae/link:develop

RUN  \
    apt-get update -qqy && apt-get install -qqy libopenblas-dev gfortran && apt-get install libsqlite3-0 \
    && pip install bs4 \
     torch \
     ernie \
     numpy \
     pandas \
     nltk \
     html5lib \
     matplotlib \
     w3lib \
     requests \
     imblearn \
     scipy \
     xlrd \
     scikit-learn \
     && python -c 'import nltk;nltk.download("'punkt'");nltk.download("'stopwords'")' \
     && rm -f /etc/protocols \
     && apt-get update \
     && apt-get -qqy install git \
     && git clone https://github.com/jeremija/python3-svmlight.git \
     && cd python3-svmlight/ \
     && chmod +x setup.py \
     && python3 setup.py build \
     && python3 setup.py install \
     && cd /opt/catenae \
     && mkdir -p synced-data/features

COPY ./ /opt/catenae

ENTRYPOINT ["python", "train.py"]