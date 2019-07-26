import numpy as np
import mygrad as mg
import mynn
import pickle

from mygrad.nnet import margin_ranking_loss

from mynn.layers.dense import dense
from mynn.initializers.normal import normal
from mynn.optimizers.adam import Adam

from sklearn.metrics.pairwise import cosine_similarity

def initialize_model(triplets, learning_rate=0.1):
    """
    Initializes the encoder model and optimizer for training.
    
    Parameters
    ----------
    triplets : Tuple[np.array, np.array, np.array]
        A tuple consisting of the embedded caption vector,
        good image feature vector, and bad image feature vector,
        each of shape=(N, d_feature), shape=(N, d_embed), and
        shape=(N, d_embed), respectively.
    
    learning_rate : float, default=0.1
        The learning rate for the model.
    
    Returns
    -------
    Tuple[Encoder, mynn.optimizers]
        A tuple consisting of an instance of the untrained
        encoder model, and the optimizer for training.
    """
    model = Encoder(d_feature=512, d_embed=50)
    optim = Adam(model.parameters, learning_rate=learning_rate)
    
    return (model, optim)

def train(model, optim, triplets, batch_size=100, num_epochs=10, margin=0.1):
    """
    Trains the encoder for embedding images by comparing
    the cosine similarities of the 'good' images with their
    corresponding caption, and the latter with random 'bad'
    images.
    
    Parameters
    ----------
    model : Encoder
        An instance of the encoder model, trained or untrained.
    
    optim : mynn.optimizers.adam
        The Adam optimizer for the model.
    
    triplet : Tuple[np.array, np.array, np.array]
        A tuple of length=3 of 2-dimensional, where column 0 is
        the embedded caption vector, column 1 is the good image
        feature vector, and column 2 is the bad image feature
        vectors, each of shape=(1, d_feature), shape=(1, d_embed),
        and shape=(1, d_embed), respectively.
    
    batch_size : int, default=100
        Size of the batch of the training dataset.
    
    num_epochs : int, default=10
        Number of times to train over the entire training dataset.
    
    margin : float, default=0.1
        A non-negative value to be used as the margin for the loss.
    
    Returns
    -------
    Encoder
        The trained encoder model for embedding images.
    """
    caption, good_img, bad_img = triplets
    
    for epoch_cnt in range(num_epochs):
        idxs = np.arange(len(caption))
        
        for batch_cnt in range(0, len(caption)//batch_size):
            batch_idxs = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
            caption_emb = caption[batch_idxs]
            good_batch = good_img[batch_idxs]
            bad_batch = bad_img[batch_idxs]
            
            good_emb = model(good_batch)
            bad_emb = model(bad_batch)
            
            good_cossim = cosine_similarity(good_emb, caption_emb)
            bad_cossim = cosine_similarity(bad_emb, caption_emb)
            
            loss = margin_ranking_loss(good_cossim, bad_cossim, 1, margin)
            
            loss.backward()
            optim.step()
            loss.null_gradients()