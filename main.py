import image_query
import token_to_idf_db
from gensim.models.keyedvectors import KeyedVectors
import pickle

# load glove 50
glove50 = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)

# load image dataset
f = open("database.p", "rb")
database = pickle.load(f)
f.close()


def run():
    """
    Performs semantic image search given an input phrase/keywords and displays
    the top related images.

    Parameters
    ----------

    Returns
    -------

    """
    text = input("Search: ")
    input_string = text.lower()

    words_to_idfs = token_to_idf_db.import_token_to_idf()

    embedded_text = image_query.se_text(input_string, words_to_idfs, glove50)
    urls = image_query.query(embedded_text, 10, database)
    image_query.display_images(urls)
