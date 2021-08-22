import numpy as np
#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import pandas as pd
import sys
from  sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE

#https://docs.rapids.ai/api/cuml/stable/
import cuml
from cuml import TSNE as  cudf_tsne
from cuml import PCA,UMAP
import cupy as cp


### features tsne ========
test_f = np.load("/home/abebe9849/ET/new_src/outputs/2021-08-02/22-16-24/test_features.npy")

features = cp.asarray(test_f)
X_reduced =cudf_tsne(n_components=2, random_state=0,perplexity=50).fit_transform(features)
X_reduced = cp.asnumpy(X_reduced)

np.save("/home/abebe9849/ET/new_src/outputs/2021-08-02/22-16-24/cuml_tsne_test.npy",X_reduced)
### features tsne ========


### features pca ========

test_f = np.load("/home/abebe9849/ET/new_src/outputs/2021-08-03/18-17-49/test_features.npy")

features = cp.asarray(test_f)

pca = PCA(n_components=2, random_state=0).fit_transform(features)
pca = cp.asnumpy(pca)
np.save("/home/abebe9849/ET/new_src/outputs/2021-08-03/18-17-49/cuml_pca_test.npy",pca)

### features pca ========


