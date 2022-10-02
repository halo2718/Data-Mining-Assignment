import numpy as np
import bs4
from urllib.request import Request, urlopen
import pandas as pd
import json
import argparse
from tqdm import tqdm
from urllib.error import HTTPError, URLError
import copy
import os

src = ""
df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df_url = df.loc[:,"url"]

for i in range(0, 10):
    trg_dir = os.path.join("./crawl", str(i), "data.json")
    files   = open(trg_dir)
    cur_json = json.load(files)
    wrong_idx = []
    for j in range(i * 4000, min((i+1) * 4000, 39644)):
        key = str(j)
        if not key in cur_json.keys():
            wrong_idx.append(j)
    if len(wrong_idx) != 0:
        for iidx in wrong_idx:
            url = df_url.iloc[iidx]
            print(url)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}  
            try:
                header_request = Request(url, headers=headers)
                html = urlopen(header_request).read().decode('utf8', 'ignore')
            except HTTPError as e:
                print(i)
                print(e)
                continue        
            html = urlopen(header_request).read().decode('utf8', 'ignore')    
            soup = bs4.BeautifulSoup(html, "html.parser")
            header = soup.find("h1").text
            article = soup.find("article")
            paragraphs = article.find_all("p")
            articles = " ".join([para.text for para in paragraphs])
            author = soup.find("div", class_= r'w-full subtitle-2 mt-8 text-left md:flex md:flex-wrap md:items-baseline md:space-x-8').find('a').text
            content  = {'title':header, 'author':author, 'content':articles}
            cur_json[iidx] = content
        with open("./crawl/"+str(i)+"/data.json", "w") as F:
            json.dump(cur_json, F)
