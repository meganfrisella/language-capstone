import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlopen
import matplotlib.pyplot as plt

def make_database(image_embeddings):
    """
    :param image_embeddings: list(type: float)
    :return: nothing
    This method makes a pickle database out of the image_embeddings
    """
    database = image_embeddings
    output = open('database.p', 'wb')
    pickle.dump(database, output)
    output.close()

def text_to_embedding(text):
    """needs to be done"""
def query(text_input, num_images):
    """
    :param text_input: string
    :param num_images: int
    :return: list[string]
    This function returns a list of the (num_images) urls that
    are most related to the text_input search
    """
    images = []
    #reading from database:
    f = open("database.p", "rb")
    #Check this: the ordering of closing and stuff
    database = pickle.load(f)
    for unused_id, image_url, image_embedding in database:
        text_embedding = text_to_embedding[text_input]
        cos_sim = cosine_similarity(text_embedding, image_embedding)
        images.append((cos_sim, image_url))
    f.close()
    sorted_list = sorted(images)[:num_images]
    result = [url for sim, url in sorted_list]
    return result
def display_images(urls):
    """
    :param urls: list[string]
    :return: none
    Displays the images specified by their urls
    """
    # downloading the data
    for url in urls:
        data = urlopen(url)
        end_index = url.rfind('.')
        img = plt.imread(data, format=url[end_index+1:])
        fig, ax = plt.subplots()
        ax.imshow(img)



