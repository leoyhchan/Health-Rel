#!/bin/bash
docker build -t model_test .
docker rm -f model_words
docker run -tid --name m1 -v ~/Projects/credibility/ReliabilityDataset/Classification/clef2018collection:/opt/catenae/clef2018collection model_test 
