"""
This file holds a bert model to classify a given text into censored or not censored
"""

#imports 
from transformers import BertForSequenceClassification
from blogpost_dataset import transform_original_dataset_2_bert_compatible
from torch.optim import Adam
from torch.nn import CrossEntropyLoss    
from torchmetrics import Accuracy
from torch.utils.data import DataLoader 
import torch

if __name__=="__main__":

    #create a bert compatible dataset
    bert_ds= transform_original_dataset_2_bert_compatible(r"C:\Users\nick\Code\MachineLearning_Projects\Bewerbung_NLP\data\english_datasets\en_train.csv",limit=100)

    #create a corresponding dataloader
    bert_dl= DataLoader(bert_ds,batch_size=6,shuffle=True)

    #learning rate and number of params
    lr=2e-5
    number_of_epochs= 4
    epsilon= 1e-8

    #init the model
    model= BertForSequenceClassification.from_pretrained('bert-base-uncased')

    #use adam optimizer
    optimizer= Adam(model.parameters(),lr=lr)

    #use Crossentropy loss
    criterion = CrossEntropyLoss()
    metric= Accuracy()
    """"
    #model compile
    model.compile(optimizer=optimizer,loss=loss,metrics=[metric])

    #train
    bert_history= model.fit(bert_dl,epochs=number_of_epochs)
    """

    for _ in range(number_of_epochs):

        for batch in bert_dl:

            #unpack the batch
            input_ids_list,attention_mask_list,token_type_ids_list,label_list,*_=batch

            #zero gradients
            optimizer.zero_grad()

            #forward_pass
            output=model(input_ids_list,attention_mask_list,token_type_ids_list)

            #compute loss
            batch_loss=criterion(output.logits,label_list.flatten().to(dtype=torch.long))
            print(batch_loss)
            batch_loss.backward()

            #optimize
            optimizer.step()

