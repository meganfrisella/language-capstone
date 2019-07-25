import embeddings
import pickle
import json


def load_coco_features():
    """
    Loads the COCO feature array dictionary.

    Parameters
    ----------

    Returns
    -------
    dict
        A dictionary that maps image IDs to (1, 512) image feature vectors

    """

    f = open("./resnet18_features.pkl", "rb")
    coco_features = pickle.load(f)
    f.close()

    return coco_features


def load_coco_metadata():
    """
    Loads the COCO dictionary from its pickle file.

    Parameters
    ----------

    Returns
    -------
    dict
        A dictionary that maps image IDs to (1, 512) image feature vectors

    """
    f = open("./captions_train2014.json", "rb")
    coco_metadata = json.load(f)
    f.close()

    return coco_metadata


def image_dataset():
    """
    Saves the image dataset in a pickle file. The dataset is a a list of tuples in
    the form (ID, url, embedding) for each image.

    Parameters
    ----------

    Returns
    -------

    """
    dataset = []
    coco_features = load_coco_features()
    coco_metadata = load_coco_metadata()

    for image in coco_metadata['images']:
        ID = image['id']
        url = image['coco_url']
        if ID in coco_features:
            embedding = embeddings.se_image(coco_features[ID])
            dataset.append((ID, url, embedding))

    output = open('database.p', 'wb')
    pickle.dump(dataset, output)
    output.close()