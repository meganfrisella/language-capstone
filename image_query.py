import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlopen
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors


def load_glove():
    return KeyedVectors.load_word2vec_format("./glove.6B.50d.txt.w2v", binary=False)


def se_text(text, words_to_idfs, glove50):
    """
    :param text: string
    :param words_to_idfs: dict[string:float]
    :return: np.array[float]
    """

    words = text.split()
    filtered_words = [word for word in words if word in words_to_idfs and word in glove50]
    res = np.zeros((50,))
    for word in filtered_words:
        res += glove50[word]*words_to_idfs[word]

    # normalize the data
    res_squared = res**2
    return res/np.sqrt(res_squared.sum())


def query(text_embedding, num_images):
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

    database = pickle.load(f)
    f.close()
    for unused_id, image_url, image_embedding in database:
        cos_sim = cosine_similarity(text_embedding, image_embedding)
        images.append((cos_sim, image_url))
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