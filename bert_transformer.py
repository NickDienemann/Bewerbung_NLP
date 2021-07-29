"""
This file holds a bert model to classify a given text into censored or not censored
"""

#imports 
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from blogpost_dataset import Blogpost_dataset



if __name__=="__main__":

    #create a dataset
    blogpost_ds= Blogpost_dataset("C:\Users\nick\Code\MachineLearning_Projects\Bewerbung_NLP\data\english_datasets\en_train.csv")

    #create a Data
