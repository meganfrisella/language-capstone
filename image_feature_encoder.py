import numpy as np
import mygrad as mg
import mynn

from mynn.layers.dense import dense
from mynn.initializers.normal import normal

class Encoder:
    def __init__(self, d_feature, d_embed):
        """
        Initializes all of the layers in our model and sets
        them as attributes of the model.
        
        Parameters
        ----------
        d_feature : int
            The dimension of the image feature vector.
        
        d_embed : int
            The dimension of the semantic embedding (i.e. the reduced
            dimensionality).
        """
        self.encode = dense(d_feature, d_embed, weight_initializer=normal)
    
    def __call__(self, x):
        """
        Passes the data as an input to the encoder model
        and performs a forward-pass, embedding the image feature
        vector into the semantic space.
        
        Parameters
        ----------
        x : np.array, shape=(N, d_feature)
            The image feature vectors of dimensionality d_feature.
        
        Returns
        -------
        mg.Tensor, shape=(N, d_embed)
            The semantic image embedding of dimensionality d_embed.
        """
        return self.encode(x)
    
    @property
    def parameters(self):
        """
        A convenience function for accessing all of the learnable
        parameters. This can be accessed as an attribute, via
        "model.parameters".
        
        Returns
        -------
        Tuple[mg.Tensor, ...]
            A tuple containing all of the learnable parameters of
            the model.
        """
        return self.encode.parameters