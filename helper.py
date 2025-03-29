import math
import embeddings
from pymilvus import MilvusClient
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import re


def get_pdf(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()
    return pages


def get_map(sitemap_url):
    """Gets the sitemap for a given sitemap url
    - Returns a list of site links"""
    response = requests.get(sitemap_url)

    soup = BeautifulSoup(response.content, "xml")

    site_map = soup.find_all("loc")

    links = []

    # gets all links
    for i in site_map:
        temp = i.getText()
        links.append(temp)

    return links


def get_site_content(url):
    """Gets the content of that page from a sitemap link
     - We can use this function when parsing through each link of a site map."""

    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    body = soup.body
    body_text = body.get_text()

    return body_text


def clean_text(text):
    """Removes all unnecessary space from a text string and formats it into a more readable format."""

    # Remove empty lines.
    text = re.sub(r'\n\s*\n', '\n', text)

    # Remove leading and trailing spaces from each line.
    text = re.sub(r'^\s+|\s+$', '', text)

    # Remove duplicate lines.
    seen = set()
    text = '\n'.join([line for line in text.splitlines() if line not in seen and not seen.add(line)])

    # Remove any other unnecessary characters.
    text = re.sub(r'\t|\n', '', text)

    # Split the text into a list of lines.
    lines = text.splitlines()

    # Format the lines into a more readable format.
    formatted_lines = []
    for line in lines:
        if line.startswith('Skip to Content'):
            formatted_lines.append(f'**Skip to Content**')
        elif line.startswith('Home'):
            formatted_lines.append(f'**Home**')
        elif line.startswith('Writing'):
            formatted_lines.append(f'**Writing**')
        elif line.startswith('Contact'):
            formatted_lines.append(f'**Contact**')
        else:
            formatted_lines.append(line)

    # Return the formatted text.
    return '\n'.join(formatted_lines)


def insert_chunk(index, link, content, collection, client):
    """ Breaks up content into chunks readable for the embedding, using a text splitter.
    - Then, Goes through each chunk of content, creates an embedding and uploads to a vector db"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8191,  # len not token count
        chunk_overlap=math.floor(8191 / 10))
    # creates a list of document chunks

    docs = text_splitter.split_text(content)

    for doc in docs:
        vect = embeddings.create_embedding(doc)
        client.insert(collection_name=collection, data={"id": index, "vector": vect, "link": link, "Content": doc})


def insert_chunk_2(index,source, content, collection, client):  # w/o link attribute using source instead
    """ Breaks up content into chunks readable for the embedding, using a text splitter.
        - Then, Goes through each chunk of content, creates an embedding and uploads to a vector db"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8191,  # len not token count
        chunk_overlap=math.floor(8191 / 10))
    # creates a list of document chunks

    docs = text_splitter.split_text(content)

    for doc in docs:
        vect = embeddings.create_embedding(doc)
        client.insert(collection_name=collection, data={"id": index, "source": source,"vector": vect, "Content": doc})


def rem_helper(milvus_client, collection_name, url):
    """Takes in a sitemap url, Milvus client info, and collection name
    - cleans, breaks up, and inserts data, from a sitemap,into your vector database"""
    # xml file
    data = get_map(url)

    # parse through the sites, get the content // testing with 5 for POC

    for i in range(2):
        link = data[i]
        content = get_site_content(link)
        content = clean_text(content)  # make sure text is cleaned before

        insert_chunk(index=i, link=link, content=clean_text(content), client=milvus_client, collection=collection_name)
