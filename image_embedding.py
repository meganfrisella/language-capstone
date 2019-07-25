import pickle


def load_coco_images():
    """
    Loads the COCO dictionary from its pickle file.

    Parameters
    ----------

    Returns
    -------
    dict
        A dictionary that maps image IDs to (1, 512) image feature vectors

    """

    f = open("./resnet18_features.pkl", "rb")
    coco_images = pickle.load(f)
    f.close()

    return coco_images


def se_image(image_features, M, b):
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
    return image_features * M + b
