#adaboost algorithm

import numpy as np
from sklearn.model_selection import train_test_split


def stumps(X,Y,u):
    #divide the data with the feature and coefficient which can lower the error the most
    X_sort=np.sort(X,axis=0)
    X_sort_copy=np.delete(np.insert(X_sort,0,-99999999999,axis=0),-1,axis=0)
    mid=(X_sort+X_sort_copy)/2
    
    m,n=X.shape
    w={} ; omega=np.inf ; sum_u=np.sum(u)
    for j in range(n):
        for i in range(m):
            yhat=np.where(X[:,j]>=mid[i,j],1,-1)
            for sign in [-1,1]:
                err_point_temp=Y!=(sign*yhat)
                omega_temp=max(u @ err_point_temp,1e-16)
                if omega_temp<omega:
                    err_point=err_point_temp
                    omega=omega_temp               
                    w["feature"]=j
                    w["sign"]=sign
                    w["therhold"]=mid[i,j]
                    
    return w,err_point,omega/sum_u
    
    
    def __init__(self):
        pass
        
    
    def getdata(self,filepath):
        
        with open(filepath) as f:
            data=np.array([ line.split() for line in f.readlines()]).astype(float)
        
        return data[:,:-1],data[:,-1]
    
    
    def train(self,X,Y,T=300):
        #training the model
        m,n=X.shape
        u=np.ones(m)/m; stump_list=[] ; alpha=[] ; omega_list=[]
        
        for t in range(T):
            if t==(T-1):
                print("U(T):",np.sum(u))
            w,err_point,omega=stumps(X,Y,u)
            omega_list.append(omega)
            stump_list.append(w)
            square=np.sqrt((1-omega)/omega )
            u=np.where(err_point,u*square,u/square)
            alpha.append(np.log(square))
        
        print("omega range:",max(omega_list),min(omega_list))
        return stump_list,np.array(alpha)
       
    
    def predictor(self,X,stump_list,alpha):
        #predict unlabeled data with trained model
        m,n=X.shape
        Yt=[]
        
        for t in range(len(stump_list)):
            feature=stump_list[t]["feature"]
            sign=stump_list[t]["sign"]
            therhold=stump_list[t]["therhold"]
            Yt.append(sign*np.where(X[:,feature]>=therhold,1,-1))
      
        return np.sign(np.array(Yt).T @ alpha)
    
    def error(self,Y,y_hat):
        return np.sum(Y!=y_hat)/Y.shape[0]
    
    def run(self):
        X_train,Y_train=self.getdata("adaboost_train.txt")
        X_test,Y_test=self.getdata("adaboost_test.txt")
        stump_list,alpha=self.train(X_train,Y_train,T=500)
        yhat=self.predictor(X_train,stump_list,alpha)
        print("Ein:",self.error(Y_train,yhat)) #emperical error
        yhat=self.predictor(X_test,stump_list,alpha)
        print("Eout:",self.error(Y_test,yhat)) #generalization error

ab=adaboost()
ab.run()
        
        