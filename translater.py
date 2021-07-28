"""
this file will be used to transform data sets into english
"""

#imports 
from googletrans import Translator
import pandas as pd

#dataformat
"""
pd.DataFrame
columns: index, text, label
"""


def translate_df(df,col_name,src_lan,dest_lan, full_dest_path=None):
    """
    task: translate the column of the given dataframe from src_lan into dest_lan \n
    parameters: df(pd.DataFrame), col_name(str), src_lan(str(language code according to googletrans.Language)), dest_lan(str(language code according to googletrans.Language)), full_dest_path(str(optional: path+file name(no file ending) to store the resulting dataframe)) \n
    return value: pd.DataFrame(translated df)
    """

    #initialize the translator
    translator= Translator()

    #define an internal function that takes a pd.Series and translates each element
    #def translator_func():

    translator_func = lambda sentence,dest_lan,src_lan: translator.translate(sentence,dest_lan,src_lan).text

    df[col_name] = df[col_name].apply(translator_func,args=[dest_lan,src_lan])

    #if a path is given store the df thus created
    if full_dest_path:
        df.to_csv(full_dest_path+".csv")

    return df

#initialize the translator
"""
translator=Translator()

text= "特写 ： 特朗普 在 联合国 的 90 分钟 记者会"  #input("enter your text: ")
source_lan= "zh-cn"
translated_to ="en"

translated_text = translator.translate(text,src=source_lan,dest=translated_to)

print(text)
print(translated_text.text)
"""


if __name__=="__main__":
    #original_df=pd.DataFrame()
    original_df= pd.read_csv(r"C:\Users\nick\Code\MachineLearning_Projects\Bewerbung_NLP\data\v1\v1\dev.tsv", sep= "\t")
    mod_df=translate_df(original_df,"text",src_lan="zh-cn",dest_lan="en", full_dest_path="data/selfmade/en_dev_2")
    print(mod_df.head())