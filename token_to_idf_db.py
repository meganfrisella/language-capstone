def import_token_to_idf():
    """imports token_to_idf dictionary

    parameters:
    ----------
    none

    returns:
    --------
    [dict] token(string) to idf dictionary
    """
    f = open("triplets.p", "rb")
    token_to_idf = pickle.load(f)
    f.close()
    return token_to_idf