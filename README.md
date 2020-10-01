# Health-Rel
This is the official repository for **Reliability Prediction for Health-related Content: A Replicability Study** [1]. It contains all the necessary **code** and **files** to reproduce the experiments shown in our study.

## Repo structure

The repo contains the following files and folders:
```
.
+-- train.py
+-- requirements.txt
+-- build.sh
+-- Readme.md
+-- LICENSE
+-- .gitignore
+-- .Rhistory
+-- lexicon
|   +-- comm_list.txt
|   +-- contact.txt
|   +-- privacy.txt
|   +-- stopwords.txt
+-- datasets
|   +-- CLEF
    |   +-- CLEF2018_qtrust_20180914.txt
|   +-- Schwarz
    |   +-- web_credibility_relabeled.xlsx
    |   +-- CachedPages
|   +-- Sondhi
    |   +-- reliable
    |   +-- unreliable
```

As can be seen, there is a main file called ```train.py``` which performs the experiments (in the following sections we will explain in more detail how to run it). There are also two files ```requirements.txt``` and ```build.sh``` which main goal is to prepare the environment for running the experiments.

On the other hand, there are two main directories in this repository. The first one is called ```lexicon``` and it contains all lexicon files used by our code. The second one contains the three ```datasets``` used.  

```Sondhi``` folder [2] contains two subfolders which divide the given webpages in reliable and unreliable examples. This dataset was facilitated to us by original study authors. 

```Schwarz``` folder [3] is formed by a set of cached webpages. This dataset was created during research and it was extracted from the following link...
It must be noticed that a relabelled ground truth is also provided, following the relabelling process described in [1].

```CLEF``` folder [4] only contains the gorund truth file with the trustworthiness assessments. In this case, it was not possible to place the full dataset in this repo due to size problems. Anyone that would like to reproduce CLEF experiments, can download the dataset from  [here](https://www.dropbox.com/s/ixnqt33u5xeelth/clef2018collection.tar.gz?dl=0). This will return a ```tar.gz``` file, which needs to be extracted as a subfolder of CLEF in order experiments to properly work. 

## Environment creation

To re-run this experiments, there are certain prerequisites that need to be fulfilled. To that end, we recomend the creation of an ```Anaconda 4.8.0``` environment with ```Python 3.7.3``` on it, as described in [1]. 

Once this environment is created, the following command needs to be executed:

```
bash build.sh
```
This will install all necessary dependencies inside the environment. 

## Re-running experiments

As said before, the main 

## Output files and expected results

# References

[1] Fernández-Pichel, M., Losada, D.. Pichel, J., Elsweiler, D. 2020. Reliability Prediction for Health-related Content: A Replicability Study. ECIR 2021 Reproducibility Track (submitted).

[2] Parikshit Sondhi, V. G. Vinod Vydiswaran, and Cheng Xiang Zhai. 2012. Reliability prediction of webpages in the medical domain. In Proceedings of the 34th European conference on Advances in Information Retrieval (ECIR'12), Ricardo Baeza-Yates, Arjen P. Vries, Hugo Zaragoza, B. Barla Cambazoglu, and Vanessa Murdock (Eds.). Springer-Verlag, Berlin, Heidelberg, 219-231. DOI=10.1007/978-3-642-28997-2_19 http://dx.doi.org/10.1007/978-3-642-28997-2_19

[3] Schwarz, J., Morris, M.: Augmenting Web Pages and Search Results to Support reliability Assessment. In: Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 1245-1254, Association for Computing Machinery (ACM), Vancouver, BC, Canada (2011)

[4] Jimmy, J., Zuccon, G., Palotti, J., Goeuriot, L., Kelly, L.: Overview of the clef 2018 consumer health search task. In: International Conference of the Cross-Language Evaluation Forum for European Languages (2018)
