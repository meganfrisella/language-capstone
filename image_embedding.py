import pickle

myPath = 'Users/MeganFrisella/GitHub/language-capstone/'


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

    f = open("resnet18_features.pkl", "rb")
    coco_images = pickle.load(f)
    f.close()

    return coco_images


def se_image(image):
    """
    Description

    Parameters
    ----------
    var: var_type
        Description

    Returns
    -------
    var_type, shape=(27, 27)
        Description

    """