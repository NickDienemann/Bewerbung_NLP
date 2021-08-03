"""
this file will be used to transform data sets into english
"""

#imports 
from googletrans import Translator
import pandas as pd


def translate_df(df,col_name,src_lan,dest_lan, full_dest_path=None):
    """
    task: translate the column of the given dataframe from src_lan into dest_lan \n
    parameters: df(pd.DataFrame), col_name(str), src_lan(str(language code according to googletrans.Language)), dest_lan(str(language code according to googletrans.Language)), full_dest_path(str(optional: path+file name(no file ending) to store the resulting dataframe)) \n
    return value: pd.DataFrame(translated df)
    """

    #initialize the translator
    translator= Translator()

    #define a translation function that returns the translated text
    translator_func = lambda sentence,dest_lan,src_lan: translator.translate(sentence,dest_lan,src_lan).text

    #apply the translation function to the dataframe
    df[col_name] = df[col_name].apply(translator_func,args=[dest_lan,src_lan])

    #if a path is given store the df thus created
    if full_dest_path:
        df.to_csv(full_dest_path+".csv")

    return df

if __name__=="__main__":
    #original_df=pd.DataFrame()
    original_df= pd.read_csv(r"C:\Users\nick\Code\MachineLearning_Projects\Bewerbung_NLP\data\v1\v1\dev.tsv", sep= "\t")
    mod_df=translate_df(original_df,"text",src_lan="zh-cn",dest_lan="en", full_dest_path="data/selfmade/en_dev_2")
