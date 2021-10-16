
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Lr_reg():
    
    def __init__(self):
        pass
    
    
    def load_data(self,path):
        with open(path) as raw_data:
            data=np.array([ line.split() for line in raw_data.readlines()]).astype("float")

        self.X,self.X_test,self.Y,self.Y_test=train_test_split(np.hstack((np.ones((data.shape[0],1)),data[:,:-1])),data[:,-1],test_size=1/3)
    
        
    def ols(self):
        X=self.X.copy()
        Y=self.Y.copy()
        self.w=np.linalg.pinv(X) @ Y
        
        yhat=X @ self.w
        Rsq=self.measure(Y,yhat)
        print(" weight:{0} \n Rsq:{1}".format(self.w,Rsq))
        
        
    def lwlr(self,k=0.1):
        X=self.X.copy()
        Y=self.Y.copy()
        m,n=X.shape
        index=np.random.randint(0,m,int(m/4))
        self.x_random=X[index]
        xtry=self.x_random[:,1:]
        
        self.w_mat=np.array([self.beta_weight(xi,k) for xi in xtry])
        
        self.yhat_lwlr=np.array([ xi @ w_beta for xi,w_beta in zip(self.x_random,self.w_mat)])
        print(self.measure(Y[index],self.yhat_lwlr))
        
        
    def beta_weight(self,xtry,k=0.1):
        X=self.X.copy()
        Y=self.Y.copy()
        m,n=X.shape
        x=X[:,1:]
        beta=np.exp(np.linalg.norm(xtry-x,axis=1)/(-2*k**2))*np.identity(m)
        nbeta=np.linalg.pinv(X.T @ beta @ X) @ X.T @ beta @ Y
        return nbeta
        
        
    def measure(self,Y,yhat):
        error=Y-yhat
        SSE=error @ error
        Syy=(Y @ Y)- ((np.sum(Y)**2)/Y.size)
        return 1-(SSE/Syy)
        
        
    def plotols(self):
        pseudo_X=np.vstack( ( np.ones(10),np.linspace(np.min(self.X[:,1]),np.max(self.X[:,1]),10)) ).T
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(self.X[:,1:],self.Y)
        ax.plot(pseudo_X[:,1:],pseudo_X @ self.w,"g")
        
    def predict_ols(self):
         yhat=self.X_test @ self.w
         print(self.measure(self.Y_test,yhat))
        
    def plotlwlr(self):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(self.X[:,1:],self.Y)
        ax.scatter(self.x_random[:,1:],self.yhat_lwlr)
        
        
    def predict_lwlr(self,k=0.1):
        X_test=self.X_test.copy()
        Y_test=self.Y_test.copy()
        
        w_mat=np.array([self.beta_weight(xi,k) for xi in X_test[:,1:]])
        yhat_lwlr=np.array([ xi @ w_beta for xi,w_beta in zip(X_test,w_mat)])
        print(self.measure(Y_test,yhat_lwlr))
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(self.X[:,1:],self.Y)
        ax.scatter(X_test[:,1:],yhat_lwlr)
        
        

       
L=Lr_reg()
L.load_data("C:\\Users\\user\\Desktop\\algorithm_implement\\mcl\\Ch08\\abalone.txt")
L.predict_lwlr()

