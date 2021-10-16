#K-nearest neighbour
import numpy as np

def knn(x,X,Y,k=1):
    yhat=[]
    for i in range(x.shape[0]):
        index=np.argsort(np.linalg.norm(x[i]-X,axis=1))[:k]
        yhat.append(np.sign(np.sum(Y[index])))
    return np.array(yhat)
    
def getdata(filepath):
    with open(filepath) as f:
        data=np.array([line.split() for line in f.readlines()]).astype(float)
        
    return data[:,:-1],data[:,-1]

def quiz15(k):
    X_train,Y_train=getdata("nbor_train.txt")
    X_test,Y_test=getdata("nbor_test.txt")
    
    yhat=knn(X_train,X_train,Y_train,k=k)
    print("Ein:",np.sum(yhat!=Y_train)/Y_train.shape[0])
    yhat=knn(X_test,X_train,Y_train,k=k)
    print("Eout:",np.sum(yhat!=Y_test)/Y_test.shape[0])
    
quiz15(5)
        