"""
This file is used to create a custom pytorch dataset from the given file
"""

#imports
import torch
from torch.utils.data import Dataset,DataLoader,Subset
import pandas as pd
from collections import namedtuple
from transformers import BertTokenizer
import numpy as np

class Blogpost_dataset(Dataset):
    """
    this class serves as a custom dataset for the given blog posts
    """

    def __init__(self,src_path,transform=None):
        """
        task: inits the dataset and sets optional transforms \n
        parameters: src_path(str(path to the underlying source data)), transform(optional transformation that may be applied to each sample ) \n
        return value:
        """

        self.src_df = pd.read_csv(src_path)
        self.transform= transform
        self.Dataset_item= namedtuple("Dataset_item",["text","label"]) 

    def __len__(self):
        """
        task: return the length of the underlying source DataFrame \n
        parameters:\n
        return value:
        """

        return len(self.src_df)

    def __getitem__(self, index):
        """
        task: return the item at the given index \n
        parameters:\n
        return value:
        """

        #transform the index to a list in case it is a tensor
        if torch.is_tensor(index):
            index= index.tolist()

        #fetch item from source df
        item=self.Dataset_item(self.src_df.iloc[index]["text"],self.src_df.iloc[index]["label"])

        #apply transform if available
        if self.transform:
            item= self.transform(item)

        return item

class Bert_compatible_dataset(Dataset):
    """
    this class holds a dataset that was transformed using the BertTransform
    """

    def __init__(self,input_ids_list,attention_mask_list,token_type_ids_list,label_list,text_list=None,transform=None):
        """
        task: create a dataset from the given lists of tokens. if text_list is given that column will hold the original text \n
        parameters:input_ids_list(list(token id)), attention_mask_list(list(attention mask)),token_type_ids_list(list(token_type_id)) ,label_list(list(label)), text_list(list(optional: orignal text))\n
        return value:
        """

        #create a class that will hold one element of data
        self.Dataset_item= namedtuple("Bert_dataset_item",["input_ids","attention_mask","token_type_ids","label","text"])

        #this list will store all data 
        self.data= []
        
        if text_list:
            #assert that all of those lists are of same length
            assert len(input_ids_list)==len(attention_mask_list)==len(token_type_ids_list)==len(label_list)==len(text_list),"length of lists has to match"
        
            #zip the lists together
            for input_ids,attention_mask,token_type_ids,label,text in zip(input_ids_list,attention_mask_list,token_type_ids_list,label_list,text_list):

                #create a namedtuple storing that data and append it to self.data
                item= self.Dataset_item(input_ids,attention_mask,token_type_ids,label,text)
                self.data.append(item)

        else:
            assert len(input_ids_list)==len(attention_mask_list)==len(token_type_ids_list)==len(label_list),"length of lists has to match"

            #zip the lists together
            for input_ids,attention_mask,token_type_ids,label in zip(input_ids_list,attention_mask_list,token_type_ids_list,label_list):

                #create a namedtuple storing that data and append it to self.data
                item= self.Dataset_item(input_ids,attention_mask,token_type_ids,label,0)
                self.data.append(item)

        self.transform = transform


    def __len__(self):
        """
        task: return the length of self.data field \n
        parameters:\n
        return value:
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        task: return the item at index \n
        parameters:\n
        return value:
        """

        #transform the index to a list in case it is a tensor
        if torch.is_tensor(index):
            index= index.tolist()

        data=self.data[index]

        if self.transform:
            data= self.transform(data)

        return data


class BertToTensor(object):
    """
    This class serves as a transform to transfer the elements of namedtuple into tensors on the given device
    """

    def __init__(self,device="cpu"):
        """
        task:  \n
        parameters:\n
        return value:
        """

        self.device=device

    def __call__(self,named_tuple):
        """
        task: transform the elements of the named tuple into tensors and ship them over to self.device \n
        parameters: named_tuple("Bert_dataset_item",["input_ids","attention_mask","token_type_ids","label","text"]) \n
        return value: transformed elements
        """

        #unpack the named tuple
        input_ids,attention_mask,token_type_ids,label,text = named_tuple

        #transform to tensor
        input_ids= torch.IntTensor(input_ids).to(device=self.device)
        attention_mask= torch.IntTensor(attention_mask).to(device=self.device)
        token_type_ids= torch.IntTensor(token_type_ids).to(device=self.device)
        label= torch.IntTensor(label).to(device=self.device)
        text= torch.IntTensor(text).to(device=self.device)

        return input_ids,attention_mask,token_type_ids,label,text


class BertTransform(object):
    """
    this class will serve as a transform that tokenizes a given text using the bert tokenizer
    """

    def __init__(self,max_length):
        """
        task: init the transform and creates a bert tokenizer\n
        parameters:\n
        return value:
        """

        self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        self.max_length= max_length

    def __call__(self,item):
        """
        task: when called transform the given items' text field by applying bertTokenization  \n
        parameters:\n
        return value:
        """

        #transform the 
        transformed_text= self.tokenizer.encode_plus(
            item.text,
            add_special_tokens=True, #adds beginning(CLS)) and end(SEP) tokens of sequence)
            max_length= self.max_length,
            pad_to_max_length=True, # makes the tokenizer fill the token vectors with padding tokens if the sequence is smaller than max_length
            return_attention_mask = True
        )

        Dataset_item=namedtuple("Dataset_item",["text","label"]) 
        return Dataset_item(transformed_text,item.label)

def transform_original_dataset_2_bert_compatible(src_path,max_length=512,limit=-1,device="cpu"):
    """
    task: use the BertTokenize transform to tokenize the given original Dataset and thus create a dataset that is bert compatible \n
    parameters: src_path(path to original data), max_length(int(max length allowed for transformer, 512 for bert)),limit(number of entries to use) \n
    return value: torch.utils.data.Dataset
    """

    #create BertTransform
    transform=BertTransform(max_length)

    #create the original Blogpost_dataset
    blogpost_ds=Blogpost_dataset(src_path,transform=transform)

    #pull a subset of the dataset if a limit was given
    if limit>0:
        indices= np.random.choice(range(len(blogpost_ds)),size=limit,replace=False) #choose random indices
        blogpost_ds= Subset(blogpost_ds,indices)

    #lists that will stored the transformed/tokenized text
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    #iterate over the dataset with transform and catch the thus transformed texts by putting them into a new dataset
    for transformed_text,label in blogpost_ds:

        #apppend the contents to the corresponding list
        input_ids_list.append(transformed_text['input_ids'])
        token_type_ids_list.append(transformed_text['token_type_ids'])
        attention_mask_list.append(transformed_text['attention_mask'])
        label_list.append([label])

    #return the Bert_compatible_dataset derived from those lists
    return Bert_compatible_dataset(input_ids_list,attention_mask_list,token_type_ids_list,label_list,transform=BertToTensor(device))




        



if __name__=="__main__":

    #create a bert compatible dataset
    bert_ds=transform_original_dataset_2_bert_compatible(r"C:\Users\nick\Code\MachineLearning_Projects\Bewerbung_NLP\data\english_datasets\en_train.csv",limit=4)

    #for item in bert_ds:
    #    print(item)
    #    break

    #create a dataloader
    bert_dl= DataLoader(bert_ds,batch_size=4)
    for batch in bert_dl:
        print(batch)
        print(batch[0].device)
        break


    """
    #create a dataset and get one item from it
    blogpost_ds=Blogpost_dataset()
    blogpost_ds= Subset(blogpost_ds,np.arange(3))

    for text,label in blogpost_ds:

        print(text)
        print(label)
        
        
    
    #create a dataloader for that dataset
    blogpost_dl = DataLoader(blogpost_ds,batch_size=4,shuffle=True)

    #iterate by using dataloader
    for batch_id,item_batch in enumerate(blogpost_dl):

        print(f"{batch_id}: has {len(item_batch.text)} items  ")
        print(item_batch.text[0])
        print(item_batch.label)

        if batch_id==3:
            break

    """
        