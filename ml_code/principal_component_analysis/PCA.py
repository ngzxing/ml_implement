import numpy as np
import matplotlib.pyplot as plt

def loaddata(filepath):
    with open(filepath) as f:
        data=np.array([line.split() for line in f.readlines()]).astype(float)
        
    return data

def pca(X,k):
    Xn=X-np.mean(X,axis=0)
    colerration=np.cov(Xn,rowvar=0)
    eignvalue,eignvector=np.linalg.eig(colerration)
    sort_eignvalue=np.argsort(eignvalue)[::-1]
    k_eignvector=eignvector[:,sort_eignvalue[:k]]
    lowddata=(Xn @ k_eignvector)
    rendata=(lowddata @ k_eignvector.T)+np.mean(X,axis=0)
    return rendata,lowddata
    
    
X=loaddata("testSet.txt")
rendata,lowddata=pca(X,k=1)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(rendata[:,0],rendata[:,1],c="r")
ax.scatter(X[:,0],X[:,1])