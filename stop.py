import nltk
from rake_nltk import Rake
from nltk.corpus import stopwords

from konlpy.tag import Okt
from konlpy.tag import Kkma

import os
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
stop_words = None
with open("stop_words.txt", 'r', encoding='utf-8') as file:
    stop_words = file.readline().split(' ')
    
stop_words = list(set(stop_words))

with open("stop_words.txt", 'w', encoding='utf-8') as file:
    file.write(" ".join(stop_words))

'''
stop_words = list(set(stop_words))

with open("stop_words.txt", 'w', encoding='utf-8') as file:
    file.write(" ".join(stop_words))
'''


file_paths = [
    "resource/data/일상대화요약_dev.json",
    "resource/data/일상대화요약_test.json",
    "resource/data/일상대화요약_train.json"
]

all_data = []

for file_path in file_paths:
    if os.path.exists(file_path):
        data = load_json(file_path)
        all_data.extend(data)
    else:
        print(f"File {file_path} not found.")

okt = Okt()
kkma = Kkma()
text = ""
for i in all_data:
    '''
    if i['id'] not in "nikluge-2024-일상 대화 요약-dev-000078":
        continue
    '''
    for j in i["input"]["conversation"]:
        text += j["utterance"]
        text += " "

word_frequency, total = dict(), 0
for noun in okt.nouns(text):
    if noun not in word_frequency:
        word_frequency[noun] = 1
    else:
        word_frequency[noun] += 1

l = []
for i in word_frequency:
    l.append((word_frequency[i], i))

l = sorted(l, key=lambda x: x[0], reverse=True)
global_l = dict()
for i in range(len(l)):
    global_l[l[i][1]] = i
global_len = len(global_l)

okt = Okt()
kkma = Kkma()
for i in all_data:
    if i['id'] not in "nikluge-2024-일상 대화 요약-dev-000078":
        continue
    text = ""
    for j in i["input"]["conversation"]:
        text += j["utterance"]
        text += " "

    loc_word_frequency = dict()
    for noun in okt.nouns(text):
        if noun not in loc_word_frequency:
            loc_word_frequency[noun] = 1
        else:
            loc_word_frequency[noun] += 1

    loc = []
    for i in loc_word_frequency:
        loc.append((loc_word_frequency[i], i))

    loc = sorted(loc, key=lambda x: x[0], reverse=True)
    li = []
    for ii in range(len(loc)):
        i = loc[ii]
        deci = round(((global_l[i[1]]+1)/global_len)/((ii+1)/len(loc)), 2)
        if i[1] not in stop_words:
            li.append((i, deci))
            print(i, (ii+1)/len(loc), (global_l[i[1]]+1)/global_len, deci)
    li = sorted(li, key=lambda x: x[1], reverse=True)

    final_list, ratio = [], 0.75
    for ii in range(len(loc)):
        jj = 0
        for j in li:
            if j[0][1] in loc[ii][1]:
                break
            else:
                jj+=1
        print(loc[ii][1], ii, jj)
        final_list.append((ii*ratio+jj, loc[ii][1]))

    for i in sorted(final_list, key=lambda x: x[0])[:10]:
        print(i)


    print()
    print()