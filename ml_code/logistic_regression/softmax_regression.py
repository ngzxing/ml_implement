import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

class Multinomial():
    
    def __init__(self):
        pass
        
    def phi(self,X,w):
        s=X @ w
        max_s=np.max(s,axis=1,keepdims=True)
        expp=np.exp(s-max_s)
        phi=expp/np.sum(expp,axis=1,keepdims=True)
    
        return phi
    
    def t(self,Y,m,k):
        t=np.zeros((m,k))
        for i in range(m):
            t[i,Y[i]]=1
        return t
            
        
    def get_data(self):
        random_seed=np.random.randint(1,1000)
        X,Y=make_blobs(10000,random_state=3)
        m=Y.shape[0]
        return np.hstack((np.ones(m).reshape(m,1),X)),Y
    
    def training(self,X,Y,num_iter=1000,eta=0.01,Lambda=0):
        m,n=X.shape
        num_class=len(set(Y))
        w=np.zeros((n,num_class))
        t=self.t(Y,m,num_class)
        
        for num in range(num_iter):
            phi=self.phi(X,w)
            w-=eta*((X.T @ (phi-t))+(Lambda*w))
            
        return w
    
    def error(self,X,Y,w):
        yhat=np.argmax(self.phi(X,w),axis=1)
        
        return np.sum(yhat!=Y)/Y.shape[0]
        
    def run(self):
        X,Y=self.get_data()
        w=self.training(X,Y)
        print(self.error(X,Y,w))
            
        
    def plotting(self):
        X,Y=self.get_data()
        num_class=len(set(Y))
        w=self.training(X,Y,num_iter=1000,eta=0.05,Lambda=0.01)
        Xv=np.vstack( (np.ones(10),np.linspace(-8,2,10) ) ).T
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(X[:,1],X[:,2],c=Y)
        print(np.linalg.norm(w,axis=0))
        print("s: ",self.error(X,Y,w))
        LR=LogisticRegression(max_iter=100)
        clf=LR.fit(X,Y)
        print("sk:",LR.score(X,Y))
        print(np.linalg.norm(clf.coef_,axis=0))
        
        
        for k in range(num_class):
            wk=w[:,k]
            w_p=-np.array([wk[0], wk[1]] )/wk[2]
            ax.plot(Xv[:,1],Xv @ w_p)
        
        
M=Multinomial()
M.plotting()
        
        