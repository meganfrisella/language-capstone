import pickle

def import_triplets():
    '''
    Imports the triplet training set of (caption, good image, bad image)

    Caption: [string]

    Good Image: [(1,512) shape np.array]

    Bad Image: [(1,512) shape np.array]
    
    Parameters:
    -----------
    None

    Returns:
    --------

    length-2 tuple:
        
        (train_triplets,test_triplets)

    train_triplets: np.array of shape (50000,3) with each row being a triplet (dtype above)

    test_triplets: np.array of shape (10000,3) with each row being a triplet (dtype above)

    '''
    f = open("triplets.p", "rb")
    train_triplets = pickle.load(f)
    test_triplets = pickle.load(f)
    f.close()
    return train_triplets,test_triplets