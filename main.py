import image_query
import token_to_idf_db


def run():
    text = input("Search: ")
    input_string = text

    glove50 = image_query.load_glove()
    words_to_idfs = token_to_idf_db.import_token_to_idf()

    embedded_text = image_query.se_text(input_string, words_to_idfs, glove50)
    urls = image_query.query(embedded_text, 10)
    image_query.display_images(urls)


