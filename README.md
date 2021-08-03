# Bewerbung_NLP

## General
This project contains machine learning models from the nlp area in order to build a binary classifier for a chinese blogpost dataset. The goal is to predict whether or not a blogpost is going to be censored or not purely on its content
The original dataset can be downloaded from: https://gitlab.com/NLP4IF/nlp4-if-censorship-detection

## Notebooks
This project contains a notebook for both BERT and DistilBERT for the dataset in its original form as well as an english version. Each of those notebooks can be easily configured by adjusting the parameters in the so called "notebook_parameters" dictionary at the beginning of each notebook.
In case you want to save a trained model, feel free to do so by using the code presented in the "save model" section of each notebook. This will allow you to specifiy a foldername and save the models state dict as well as the state of the notebook_parameters and train/validation evaluation scores in 2 sperate files within that specified folder.

## Other files
The EDA_toolkit is a selfwritten object-oriented data exploration tool wrapepd around an EDA course on Kaggle to offer a nice and quick-to-use API. It is used to explore the correlation between syntax and the posts likelihood of being cencored in the data exploration notebook. 
The translator.py was used to translate the original dataset from chinese to english.
