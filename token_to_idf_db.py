import pickle


def import_token_to_idf():
    """imports token_to_idf dictionary

    parameters:
    ----------
    none

    returns:
    --------
    [dict] token(string) to idf dictionary
    """
    f = open("token_to_idf.p", "rb")
    token_to_idf = pickle.load(f)
    f.close()
    return token_to_idf
