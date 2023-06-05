import re
import pandas as pd
from unidecode import unidecode

# Read the CSV files
df_alay = pd.read_csv("dataset/new_kamusalay.csv", names=['alay', 'normal'], encoding='latin-1')
dict_alay = dict(zip(df_alay['alay'], df_alay['normal']))

df_abusive = pd.read_csv('dataset/abusive.csv', names=['label'], encoding='latin-1')
set_abusive = set(df_abusive['label'].to_list())

df_stopwords = pd.read_csv('dataset/stopwords-id.txt', sep=" " ,names=['nirmakna'])
set_stopwords = set(df_stopwords['nirmakna'].to_list())

#Function clean
def clean_text(text):
    #Lowercasing all the letters
    text = text.lower()
    #Remove ascii2
    text = re.sub(r'\\x[A-Za-z0-9./]+','', unidecode(text))
    #Remover every \\n
    text = re.sub(r'\\n',' ', text)
    #Remove punctuations only show alphabets
    text = re.sub(r"[^\w\d\s]+", " ",text)
    #Remove every new line
    text = re.sub(r'\n', ' ',text)
    #Remove space
    text = re.sub('  +', ' ',text)
    #Remover every text url
    text = re.sub('url','', text)
    #remove web url
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    #Remove rt (retweet)
    text = re.sub('rt ',' ', text)
    #remove word user
    text = text.replace('user', '')
    #remove space in and out
    text = text.strip()
    return text

#Function Change Alay to Normal
def change_alay(text):
    for word in dict_alay:
        change_word = ' '.join([dict_alay[word] if word in dict_alay else word for word in text.split(' ')])
        return change_word

#Function remove abusive
def remove_abusive(text):
    text = text.split(" ")
    text = [i for i in text if i not in set_abusive]
    text = ' '.join(text)
    return text

#Remove stopword
def remove_stopwords(text):
    text = text.split(" ")
    text = [i for i in text if i not in set_stopwords]
    text = ' '.join(text)
    return text

#Function Cleansing
def clean(text):
    
    text = clean_text(text)
    text = change_alay(text)
    text = remove_abusive(text)
    text = remove_stopwords(text)

    return text