#!/usr/bin/env python
# coding: utf-8

# In[231]:


import json
import pickle
from urllib.request import urlopen
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import random
import numpy as np

from image_query_weird import se_text as aaaa

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


with open(r'/Users/wellison/Desktop/BWSI/language-capstone/captions_train2014.json','rb') as f:
    captions_train = json.load(f)


# In[107]:


with open("./dat/stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]


# In[22]:


captions_train['images'][0]


# In[10]:


f = open(r'/Users/wellison/Desktop/BWSI/language-capstone/resnet18_features.pkl','rb')
coco_features = pickle.load(f)
f.close()

#coco_features is the len-82612 dictionary of captions_train image ids to 512-d vectors for the image


# In[49]:


coco_features[57870].shape


# In[ ]:





# In[16]:


len(coco_features)


# In[19]:


from urllib.request import urlopen
import matplotlib.pyplot as plt

# downloading the data
data = urlopen("http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg")

# converting the downloaded bytes into a numpy-array
img = plt.imread(data, format='jpg')  # shape-(460, 640, 3)

# displaying the image
fig, ax = plt.subplots()
ax.imshow(img)


# In[30]:


idx = 5
print(captions_train['images'][idx])
print()
print(image_id_to_annotation[57870])


# In[24]:


image_id_to_annotation = defaultdict(list)

for dic in captions_train['annotations']:
    image_id_to_annotation[dic['image_id']].append(dic['caption'])


# In[25]:


image_id_to_annotation[379340]


# In[236]:


training_set_length = 50000

train_triplets = np.ndarray((training_set_length,3),dtype=tuple)



print(train_triplets.shape)

for idx in tqdm(range(len(train_triplets))):
    good_image_id = random.choice(list(coco_features.keys()))
    bad_image_id = random.choice(list(coco_features.keys()))
    annotation = random.choice(list(  image_id_to_annotation[good_image_id]  ))
        train_triplets[idx] = (string_to_vector(annotation,glove50),coco_features[good_image_id],coco_features[bad_image_id])
    #train_triplets[idx] = (annotation,good_image_id,bad_image_id)
    
    
testing_set_length = 10000

test_triplets = np.ndarray((training_set_length,3),dtype=tuple)



print(test_triplets.shape)

for idx in tqdm(range(len(test_triplets))):
    good_image_id = random.choice(list(coco_features.keys()))
    bad_image_id = random.choice(list(coco_features.keys()))
    
    annotation = random.choice(list(  image_id_to_annotation[good_image_id]  ))
    
    
    
    test_triplets[idx] = (string_to_vector(annotation,glove50),coco_features[good_image_id],coco_features[bad_image_id])
    #train_triplets[idx] = (annotation,good_image_id,bad_image_id)
    


# In[240]:


test_triplets.shape

test_triplets = test_triplets[:10000]


# In[241]:


f = open("triplets.p", "wb")
pickle.dump(train_triplets, f)
pickle.dump(test_triplets, f)
f.close()


# In[ ]:


f = open("triplets.p", "rb")
train_triplets = pickle.load(f)
test_triplets = pickle.load(f)
f.close()


# In[91]:


triple_idx = 1

print(train_triplets[triple_idx,0])
print(train_triplets[triple_idx,1])
print(train_triplets[triple_idx,2])

# downloading the data
data = urlopen("http://images.cocodataset.org/train2014/COCO_train2014_000000489218.jpg")

# converting the downloaded bytes into a numpy-array
img = plt.imread(data, format='jpg')  # shape-(460, 640, 3)

# displaying the image
fig, ax = plt.subplots()
ax.imshow(img)


# In[93]:


len(image_id_to_annotation)


# In[115]:


import re, string

# this creates a regular expression that identifies all punctuation character
# don't include this in `strip_punc`, otherwise you will re-compile this expression
# every time you call the function
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def to_counter(doc):
    """ 
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.
    
    Parameters
    ----------
    doc : str
    
    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    # <COGINST>
    return Counter(set(strip_punc(doc).lower().split()))
    # </COGINST>
    
def to_vocab(counters, k=None, stop_words=None):
    """ 
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`
    
    Parameters
    ----------
    counters : Iterable[Iterable[str]]
    
    k : Optional[int]
        If specified, only the top-k words are returned
    
    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the vocabulary
    """
    # <COGINST>
    vocab = Counter()
    for counter in counters:
        vocab.update(counter)
        
    if stop_words is not None:
        for word in set(stop_words):
            vocab.pop(word, None)  # if word not in bag, return None
    return sorted(i for i,j in vocab.most_common(k))
    # </COGINST>

def to_idf(vocab, counters):
    """ 
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.
    
    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.
    
    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`: 
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of 
        documents in which the term `t` occurs.
    """
    # <COGINST>
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in vocab]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)
    # </COGINST>


# In[143]:


counters = [to_counter(doc) for doc in image_id_to_annotation[57870]]


vocab = to_vocab(counters,stop_words=stops)

print(vocab)
to_idf(vocab,counters)


# In[125]:


cnt = Counter()


# In[127]:


for capt_list in image_id_to_annotation:
    for caption in image_id_to_annotation[capt_list]:
        cnt.update(set(strip_punc(caption).lower().split()))


# In[223]:


cnt['frisbe']


# In[152]:


vocab = to_vocab([cnt],stop_words=stops)


# In[153]:


len(vocab)


# In[165]:


vocab[933]


# In[154]:


idfs = to_idf(vocab,cnt)


# In[166]:


idfs[933]


# In[155]:


token_to_idf = {v:i for v,i in zip(vocab,idfs)}


# In[219]:


token_to_idf['selfie']


# In[167]:


f = open("token_to_idf.p", "wb")
pickle.dump(token_to_idf, f)
f.close()


# In[ ]:


f = open("triplets.p", "rb")
token_to_idf = pickle.load(f)
f.close()


# In[181]:


from gensim.models.keyedvectors import KeyedVectors
path = r"/Users/wellison/Desktop/BWSI/language-capstone/dat/glove.6B.50d.txt.w2v"
glove50 = KeyedVectors.load_word2vec_format(path, binary=False)


# In[215]:


string_to_vector("this is a string",glove50)


# In[214]:


def string_to_vector(string,glove50):
    return aaaa(string,token_to_idf,glove50)


# In[ ]:


image_query.se_text()


# In[235]:


from image_query_weird2 import se_text as aaaa


# In[ ]:


aaaa()

