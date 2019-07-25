import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlopen
import matplotlib.pyplot as plt


def se_text(text, words_to_idfs, glove50):
    """
    :param text: string
    :param words_to_idfs: dict[string:float]
    :return: np.array[float]
    """

    words = text.split()
    filtered_words = [word for word in words if word in words_to_idfs and word in glove50]
    res = np.zeros((1, 50))
    for word in filtered_words:
        res += glove50[word]*words_to_idfs[word]

    # normalize the data
    res_squared = res**2
    return res/np.sqrt(res_squared.sum())


def query(text_embedding, num_images, database):
    """
    :param text_input: string
    :param num_images: int
    :return: list[string]
    This function returns a list of the (num_images) urls that
    are most related to the text_input search
    """
    images = []

    image_embeddings = database[0]
    text_embedding = text_embedding.flatten()
    res = (image_embeddings@text_embedding).reshape(text_embedding.shape)
    zipped = zip((res, database[1]))
    sort = sorted(zipped)[:num_images]
    result = [j for i,j in sort]
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