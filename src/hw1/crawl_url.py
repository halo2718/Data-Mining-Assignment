import numpy as np
import bs4
from urllib.request import Request, urlopen
import pandas as pd
import json
import argparse
from tqdm import tqdm
from urllib.error import HTTPError, URLError
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

src = ""
df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df_url = df.loc[:,"url"]

json_info = {}

wrong_idx = []

for i in tqdm(range(args.start * 4000, min(df_url.shape[0], (args.start + 1) * 4000))):
    url = df_url.iloc[i]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}  
    try:
        header_request = Request(url, headers=headers)
        html = urlopen(header_request).read().decode('utf8', 'ignore')
    except:
        print(i)
        wrong_idx.append(i)
        continue        
    soup = bs4.BeautifulSoup(html, "html.parser")
    header = soup.find("h1").text
    article = soup.find("article")
    paragraphs = article.find_all("p")
    articles = " ".join([para.text for para in paragraphs])
    author = soup.find("div", class_= r'w-full subtitle-2 mt-8 text-left md:flex md:flex-wrap md:items-baseline md:space-x-8').find('a').text
    content  = {'title':header, 'author':author, 'content':articles}
    json_info[i] = content
    # time.sleep(0.1)

# print(json_info)
print(wrong_idx)
# while len(wrong_idx) != 0:
#     tmp_idx = []
#     for i in wrong_idx:
#         url = df_url.iloc[i]
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}  
#         try:
#             header_request = Request(url, headers=headers)
#             html = urlopen(header_request).read().decode('utf8', 'ignore')
#         except:
#             # print(i)
#             tmp_idx.append(i)
#             continue        
#         soup = bs4.BeautifulSoup(html, "html.parser")
#         header = soup.find("h1").text
#         article = soup.find("article")
#         paragraphs = article.find_all("p")
#         articles = " ".join([para.text for para in paragraphs])
#         author = soup.find("div", class_= r'w-full subtitle-2 mt-8 text-left md:flex md:flex-wrap md:items-baseline md:space-x-8').find('a').text
#         content  = {'title':header, 'author':author, 'content':articles}
#         json_info[i] = content
#     wrong_idx = copy.deepcopy(tmp_idx)

with open("./crawl/"+str(args.start)+"/data.json", "w") as F:
    json.dump(json_info, F)

print(wrong_idx)
# print(html) 
