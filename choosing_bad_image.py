"""
The purpose of this is to find the most similar image to be the "bad" image in order to better train the model
"""

def makeString(arr):
    res = ""
    for s in arr:
        res+=s+" "
    return res

for idx in tqdm(range(len(train_triplets))):
    good_image_id = random.choice(list(coco_features.keys()))
    bad_image_ids = [random.choice(list(coco_features.keys())) for i in range(10)]
    bad_image_ids = [image_id for image_id in bad_image_ids if image_id != good_imamge_id]
    good_image_text_embedding = image_id_to_annotation[good_image_id]
    bad_image_text_embeddings = [makeString(image_id_to_annotation[i], for i in bad_image_ids]

    bad_image_sims = [cosine_similarity(se_text(good_image_text_embedding, word_to_idfs, glove50),
                                        se_text(embed, word_to_idfs, glove50))
                      for embed in bad_image_text_embeddings]
    zipped = tuple(zip(bad_image_sims, bad_image_ids))
    bad_image_id = sorted(zipped)[0][1]

    annotation = random.choice(list(image_id_to_annotation[good_image_id]))
    train_triplets[idx] = (
    string_to_vector(annotation, glove50), coco_features[good_image_id], coco_features[bad_image_id])
    # train_triplets[idx] = (annotation,good_image_id,bad_image_id)
