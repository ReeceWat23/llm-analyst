import requests
from pymilvus import MilvusClient
import helper
import SearchEng
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
import langchain.agents as agent

client = MilvusClient(
    uri="https://in03-5d7a6e0d5ff229c.api.gcp-us-west1.zillizcloud.com",  # Cluster endpoint obtained from the console
    token="db3f305a3164011c78212dffbb378983bd31bc209e0722d35623447320de893808c6873c5fc33a484ebd8f526e13c01aba81d033"
)

collection = "Llm_analyst_data"

client.drop_collection(collection_name=collection)
#
# client.create_collection(collection_name=collection,dimension=1536)
#
# reports = ["GoldmansachsQ2-10Q.pdf","GoldmansachsQ1-10Q.pdf","GoldmansachsQ4-22-10Q.pdf"]
# index = 0
# # for loop to go through each and insert into our db
# print("..processing data")
# for i in reports:
#     text = extract_text(i)
#     helper.insert_chunk_2(index=index,source=i,content=helper.clean_text(text),collection=collection,client=client)
#     index += 1

# def find_docs(links):
#     x = None
#     for link in links:
#         if str(link.get('href')).__contains__("medium"):
#             x = link.get('href')
#     return x

twitterscraper = requests.get("https://twitter.com/elonmusk")
content = twitterscraper.content
soup2 = BeautifulSoup(content,"html.parser")
tweets = soup2.find_all('tweetText')

