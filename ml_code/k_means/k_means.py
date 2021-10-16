import numpy as np
import matplotlib.pyplot as plt

def group(X,miu,k):
    distance=[]
    for grp in range(k):
        distance.append(np.linalg.norm(X-miu[grp],axis=1)**2)
    return np.argmin(np.array(distance),axis=0)

    
def cal_means(X,grplist,k,miu):
    newmiu=[]
    for grp in range(k):
        S=X[grplist==grp]
        if np.any(S)==False:
            newmiu.append(miu[grp])
        else:
            newmiu.append(np.mean(S,axis=0))
        
    return np.array(newmiu)
        

def kmeans(X,k):
    m,n=X.shape
    index=np.random.randint(0,m,k); miu=X[index]
    differ=np.inf

    while np.any(differ>1e-10):
        grplist=group(X,miu,k)
        new_miu=cal_means(X,grplist,k,miu)
        differ=np.abs(new_miu-miu)
        miu=new_miu
        
    return grplist,miu

def run(X,k):
    grplist,miu=kmeans(X,k)
    #fig=plt.figure()
    #ax=fig.add_subplot(111)
    #ax.scatter(X[:,0],X[:,1],c=grplist)
    Ein=0
    
    for grp in range(k):
        Ein+=np.sum(np.linalg.norm(X[grplist==grp]-miu[grp],axis=1)**2)
        
    return Ein/X.shape[0]

def getdata(filepath):
    with open(filepath) as f:
        data=np.array([line.split() for line in f.readlines()]).astype(float)
        
    return data

Ein=0
for n in range(500):
    Ein+=run(getdata("Km.txt"),10)
print(Ein/500)
    
        
        
        

    