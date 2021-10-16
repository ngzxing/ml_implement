#neural network implementation


import numpy as np
import time

class neural():
    
    def __init__(self,M):
        self.M=M #list of number of neurons in each network layer
        self.dtanh=lambda s: 1-(np.tanh(s)**2)
        
        
    def fit(self,X,Y,eta=0.1,r=0.1,T=50000): #training model
        m,n=X.shape
        layers=len(self.M)
        WL={layer:np.random.uniform(-r,r,(self.M[layer-1]+1)*self.M[layer]).
            reshape(1+self.M[layer-1],self.M[layer]) for layer in range(1,layers)}
        
        for t in range(T):
            index=np.random.choice(m); x=X[index]; y=Y[index]
            Slist={1:np.hstack((1,WL[1].T @ x))}
            
            for layer in list(range(2,layers)):
                Slist[layer]=np.hstack((1,WL[layer].T @ np.tanh(Slist[layer-1])))
            
            deltalist={layers-1:-2*self.dtanh(Slist[layers-1][1:])*(y-np.tanh(Slist[layers-1][1:]))}
            
            for layer in list(range(1,layers-1))[::-1]:
                deltalist[layer]=(WL[layer+1][1:] @ deltalist[layer+1])*self.dtanh(Slist[layer][1:])
            
            WL[1]-=eta*(x.reshape(n,1) @ deltalist[1].reshape(1,self.M[1]))
        
            for layer in range(2,layers):
                u=np.tanh(Slist[layer-1]).reshape(self.M[layer-1]+1,1) @ deltalist[layer].reshape(1,self.M[layer])
                WL[layer]-=eta*u
            
        return WL
    
    def predict(self,X,WL): #use model to predict
        layers=len(self.M)
        Xl=X
        m=X.shape[0]
        
        for layer in range(1,layers):
            Xlp=np.tanh(Xl @ WL[layer])
            Xl=np.hstack((np.ones(m).reshape(m,1),Xlp))
            
        return np.sign(Xlp.reshape(m,))
        

    def run(self,X_train,Y_train,X_test,Y_test,T=50000,eta=0.1,r=0.1):
        WL=self.fit(X_train,Y_train,T=T,eta=eta,r=r)
        yhat=self.predict(X_train,WL)
        Ein=np.sum(yhat!=Y_train)/Y_train.shape[0]
        yhat=self.predict(X_test,WL)
        Eout=np.sum(yhat!=Y_test)/Y_test.shape[0]
        
        return Ein,Eout
    
def getdata(filepath): #processing data from txt file
    with open(filepath) as f:
        data=np.array([line.split() for line in f.readlines()]).astype(float)
    m=data.shape[0]
    
    return np.hstack( (np.ones(m).reshape(m,1),data[:,:-1])),data[:,-1]
    
def cross_validation_nOfLayer(repeat):
    X_train,Y_train=getdata("nnet_train.txt")
    X_test,Y_test=getdata("nnet_test.txt")
    n=X_train.shape[1]
    
    for M in [1,6,11,16,21]:
        print("M:",M)
        clf=neural([n-1,M,1]) #the network structure is n-1;M;1 could be extended to more layers
        Ein_sum=0; Eout_sum=0
        for i in range(repeat):
            start=time.time()
            Ein,Eout=clf.run(X_train,Y_train,X_test,Y_test,50000)
            end=time.time()
            print(end-start)
            Ein_sum+=Ein; Eout_sum+=Eout 
        
        print("  Ein:",Ein_sum/repeat) #output the emperical error
        print("  Eout:",Eout_sum/repeat) #output the generalization error
        


cross_validation_nOfLayer(100)
