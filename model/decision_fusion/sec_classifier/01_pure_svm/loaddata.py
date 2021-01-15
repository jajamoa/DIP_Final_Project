import pandas as pd
import numpy as np
# import torch

def load_csv(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    dfa=df.values
    assert isinstance(dfa, np.ndarray)
    x,y=dfa[:,1:],np.squeeze(dfa[:,0])
    # ones=torch.sparse.torch.eye(3)
    # y=ones.index_select(0,torch.tensor(y,dtype=torch.int64))
    return x,y


