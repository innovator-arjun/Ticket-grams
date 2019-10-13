import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation
from nltk.tokenize import RegexpTokenizer
import re
import collections
from past.builtins import xrange

import spacy
nlp=spacy.load('en_core_web_sm')
#pointing the source and listing the file names under that path
path = os.chdir('C://Users//ar393556//Documents//Ticket grams//sample')
files = os.listdir(path)
print(files)


#Fetching only xlsx files
files_xls = [f for f in files if f[-4:] == 'xlsx']
print(files_xls)

df = pd.DataFrame()

#Reading the files and appending the data in the DataFrame format
for f in files_xls:
    data = pd.read_excel(f,sheet_name='CI-Data')
    df = df.append(data)
print(df.shape)




#Taking only TicketID and Summary which are essential

ticket_df_full = pd.DataFrame(df, columns= ['TicketID','Summary'])

ticket_df_trim=ticket_df_full[['TicketID','Summary']]

#no. of records and columns
ticket_df_trim.shape

#Dropping null/blank records
ticket_df_trim=ticket_df_trim.dropna()

ticket_df = ticket_df_trim[ticket_df_trim['Summary'].notnull()]


print(ticket_df.shape)
ticket_df['unclean']=ticket_df['Summary']


# To remove punchuation, numbers and converting the string to lower
ticket_df['Summary'] = ticket_df['Summary'].str.lower().str.replace(r'[^a-z\s]', '')


nlp.Defaults.stop_words.add('thanks')
nlp.Defaults.stop_words.add('please')
nlp.Defaults.stop_words.add('team')
nlp.Defaults.stop_words.add('dear')
nlp.Defaults.stop_words.add('hi')

nlp.Defaults.stop_words.remove('not')
nlp.Defaults.stop_words.remove('cannot')
nlp.Defaults.stop_words.remove('nothing')

stop_words=nlp.Defaults.stop_words

ticket_df=ticket_df.dropna()
vari=ticket_df.isnull().sum()
print(vari)

print('Lemmatization Started')

ticket_df['Summary'] = ticket_df['Summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
ticket_df['Summary'] = ticket_df['Summary'].apply(lambda x: " ".join(x for x in x.split() if len(x)>=2 and len(x)<15))


ticket_df['Summary']= ticket_df['Summary'].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))

print('Lemmatization Done')



def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]



ui=collections.Counter()

for i in ticket_df['Summary']:
    x = i.rstrip().split(" ")
    ui.update(ngrams(x, n=1))

ui_counter = pd.DataFrame.from_dict(ui, orient='index')
ui_counter.to_csv('uigrams.csv')
ui_df=pd.read_csv('uigrams.csv',index_col=False)
ui_df['Unnamed: 0'] = ui_df['Unnamed: 0'].str.lower().str.replace(r'[^a-z\s]', '')
dict_ui=ui_df.set_index('Unnamed: 0')['0'].to_dict()



bi=collections.Counter()
for i in ticket_df['Summary']:
    x = i.rstrip().split(" ")
    bi.update(set(zip(x[:-1],x[1:])))

bigram_counter = pd.DataFrame.from_dict(bi, orient='index')
bigram_counter.to_csv('bigrams.csv')
bigram_df=pd.read_csv('bigrams.csv',index_col=False)
bigram_df['Unnamed: 0'] = bigram_df['Unnamed: 0'].str.lower().str.replace(r'[^a-z\s]', '')
dict_bigram=bigram_df.set_index('Unnamed: 0')['0'].to_dict()



tri=collections.Counter()
for i in ticket_df['Summary']:
    x = i.rstrip().split(" ")
    tri.update(set(zip(x[:-2],x[1:-1],x[2:])))

trigram_counter = pd.DataFrame.from_dict(tri, orient='index')
trigram_counter.to_csv('trigram.csv')
trigram_df=pd.read_csv('trigram.csv',index_col=False)
trigram_df['Unnamed: 0'] = trigram_df['Unnamed: 0'].str.lower().str.replace(r'[^a-z\s]', '')
dict_trigram=trigram_df.set_index('Unnamed: 0')['0'].to_dict()


four=collections.Counter()

for i in ticket_df['Summary']:
    x = i.rstrip().split(" ")
    four.update(ngrams(x, n=4))

fourgram_counter = pd.DataFrame.from_dict(four, orient='index')
fourgram_counter.to_csv('fourgram.csv')
fourgram_df=pd.read_csv('fourgram.csv',index_col=False)
fourgram_df['Unnamed: 0'] = fourgram_df['Unnamed: 0'].str.lower().str.replace(r'[^a-z\s]', '')
dict_four=fourgram_df.set_index('Unnamed: 0')['0'].to_dict()


def top_grams(dicti, N):
    res_topgram = []
    for text in ticket_df['Summary']:

        comp_list = get_ngrams(text, N)
        n = {k: dicti[k] for k in comp_list if k in dicti}
        flag = bool(n)
        res = ''
        if flag == True:
            res = max(n, key=n.get)
        else:
            res = 'No Pattern'
        res_topgram.append(res)

    return res_topgram



res_unigram=top_grams(dict_ui,1)
ticket_df['Top1Grams']=res_unigram


res_bigram=top_grams(dict_bigram,2)
ticket_df['Top2Grams']=res_bigram


res_trigram=top_grams(dict_trigram, 3)
ticket_df['Top3Grams']=res_trigram


res_fourgram=top_grams(dict_four,4)
ticket_df['Top4Grams']=res_fourgram

print('TOp grams are done')
print(ticket_df.head())


ticket_df.to_csv('ticket_df_with_topgrams.csv')

path = os.chdir('C://Users//ar393556//Documents//Ticket grams//CIKeyword')
files = os.listdir(path)

automation_df=pd.read_excel('CIKeywords.xlsx')

automation_df['Keyword']=automation_df['Keyword'].apply(lambda x: x.strip(punctuation))
automation_df['Keyword']=automation_df['Keyword'].str.replace('*',' ')


# To remove punchuation, numbers and converting the string to lower
automation_df['Keyword']=automation_df['Keyword'].str.lower()

automation_df['Keyword']=automation_df['Keyword'].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))

dict_keyword=automation_df.set_index('Keyword')['Weight'].to_dict()

dict_automation_text=automation_df.set_index('Keyword')['AutomationCategory'].to_dict()

print('Automation imported')
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

automation_col = []
# for i in columns:
# for record in range(0,len(ticket_df)):
for index, row in ticket_df.iterrows():

    #     for index, row in df.iterrows():
    #     print(index)

    grams_list_record = []
    grams_list_record.append(str(ticket_df['Top1Grams'][index]))
    grams_list_record.append(str(ticket_df['Top2Grams'][index]))
    grams_list_record.append(str(ticket_df['Top3Grams'][index]))
    li = []
    #     li=[i for grams_words in grams_list_record for i in dict_keyword if fuzz.token_sort_ratio(grams_words,i)>80]
    for grams_words in grams_list_record:
        highest = process.extractOne(grams_words, dict_keyword.keys(), scorer=fuzz.token_set_ratio)
        # scorer=fuzz.QRatio ,  scorer=fuzz.ratio
        li.append(highest[0])
    #         for dict_words in dict_keyword:
    #             if fuzz.token_sort_ratio(grams_words,dict_words)>80:
    #                 li.append(dict_words)
    #                 break

    n = {k: dict_keyword[k] for k in li if k in dict_keyword}
    #     print(grams_list_record)
    #     print(n)
    print(index)
    flag = bool(n)
    res = ''
    if flag == True:
        res = max(n, key=n.get)
    else:
        res = ''
    # print(res)
    automation_col.append(res)


ticket_df.to_csv('checking.csv')
automation_text = [dict_automation_text[k] for k in automation_col if k in dict_automation_text]
ticket_df['Automation Category']=automation_text
ticket_df.to_csv('ticket_df_with_topgrams_automation_category_50K.csv')

print('Finish')
