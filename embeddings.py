

def se_image(image_features):
    """
    Description

    Parameters
    ----------
    image_features: np.array, shape=(1, 512)
        Image feature array from COCO dataset

    M: np.array, shape=(512, 50)
        Trained parameter for embedding image feature array

    b: np.int
        Trained bias parameter for embedding image feature array

    Returns
    -------
    np.array, shape=(1, 50)
        Image embedding

    """
    M = # get from Christian
    b = # get from Christian
    return image_features * M + b
