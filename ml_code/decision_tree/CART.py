import numpy as np
import time

class CART():
    
    def __init__(self):
        pass
    
    def getdata(self,filepath):
        with open(filepath) as f:
            data=np.array([line.split() for line in f.readlines()]).astype("float")
        
        return data[:,:-1],data[:,-1]
        
    
    def Gini(self,X,Y,j,theta,m):
        cls,m_cls=np.unique(Y,return_counts=True)
        indexD1=X[:,j]>theta
        m_D1=X[indexD1].shape[0];  m_D2=X[True!=indexD1].shape[0]
        m_kD1=np.array([ np.sum((Y==k)*indexD1) for k in cls])
        gini=lambda m_kD,m_D : 1- (np.sum(m_kD**2)/m_D**2)
        G=((m_D1*gini(m_kD1,m_D1)) + (m_D2*gini(m_cls-m_kD1,m_D2)))/m
        return indexD1,G
        
    
    def decstump(self,subX,subY):
        m,n=subX.shape
        sort_X=np.sort(np.unique(subX,axis=0),axis=0)
        thetas=((np.vstack((sort_X,sort_X[-1]))+np.vstack((sort_X[0],sort_X)))/2)[1:-1]
        declist={}; ginimax=1; 
        
        for j in range(n):
            for theta in thetas[:,j]:
                
                indexD1,gini=self.Gini(subX,subY,j,theta,m)
                
                if gini<ginimax:
                    ginimax=gini; 
                    declist["feature"]=j
                    declist["theta"]=theta
                    declist["son"]=1
                    sX=[subX[indexD1],subX[True!=indexD1]]
                    sY=[subY[indexD1],subY[True!=indexD1]]
                    
        return sX,sY,declist
    
    
    def build_tree(self,X,Y,prune=np.inf):
        m,n=X.shape
        m_subX=m
        subX=X; subY=Y
        subXlist_temp=[X]; subYlist_temp=[Y]
        dectree={}; n=0; grow=True
        
        while (grow and (n<=prune)):
            n+=1
            subXlist=subXlist_temp; subYlist=subYlist_temp 
            subXlist_temp=[]; subYlist_temp=[]; i=0
            
            for subX,subY in zip(subXlist,subYlist):
                i+=1; name="{0}.{1}".format(n,i)
                
                if  ( (subX.shape[0]==1) or (np.sum(subY==subY[0])==subY.shape[0]) ): 
                    dectree[name]={"yhat":subY[0],"son":0}
                    
                elif (n==prune):
                    dectree[name]={"yhat":np.sign(np.sum(subY)),"son":0}
                    
                else:
                    sX,sY,declist=self.decstump(subX,subY)
                    dectree[name]=declist
                    subXlist_temp.extend(sX); subYlist_temp.extend(sY)
                    
            if len(subXlist_temp)==0:
                grow=False
        
        return dectree,n
    
    def cum_son(self,father_name,dectree):
        father,son=father_name.split(".")
        n_son=0
        for i in range(int(son)):
            name="{0}.{1}".format(father,i+1)
            n_son+=dectree[name]["son"]*2
        
        return n_son
            
    
    def predict(self,X,dectree,n):
        m=X.shape[0]; yhat=[]
        
        for x in X:
            next_stump=1
            for father in range(n):
                name="{0}.{1}".format(father+1,next_stump)
                decstump=dectree[name]
                if decstump["son"]==1:
                    j=decstump["feature"]; theta=decstump["theta"]
                    n_son=self.cum_son(name,dectree)
                    
                    if x[j]>theta:
                        next_stump=n_son-1
                    else: 
                        next_stump=n_son
                    
                    father_name=name
                    
                else:
                    yhat.append(decstump["yhat"])
                    break
                    
        return np.array(yhat)
                    
    
    def RF(self,X,Y,ntree=300,prune=np.inf):
        treelist=[]
        m=Y.shape[0]
        index=np.arange(m)
        
        for nt in range(ntree):
            rindex=np.random.choice(index,100)
            Xr=X[rindex]; Yr=Y[rindex]
            treelist.append(self.build_tree(Xr,Yr,prune))
            
        return treelist
            
    def RF_predict(self,X,treelist):
        yhat=np.zeros(X.shape[0])

        for tree in treelist:
            yhat+=self.predict(X,tree[0],tree[1])
            
        return np.sign(yhat)
             

    
def run():
    c=CART()
    X_train,Y_train=c.getdata("dectree_train.txt")
    X_test,Y_test=c.getdata("dectree_test.txt")
    m=Y_train.shape[0]
    index=np.arange(m); sum_ein=0; sum_eout=0
    ntest=100
    
    for i in range(ntest):
        start=time.time()
        treelist=c.RF(X_train,Y_train,ntree=1,prune=10000)
        end=time.time()
        print("time:",end-start)
        yhat=c.RF_predict(X_train,treelist)
        sum_ein+=np.sum(yhat!=Y_train)/Y_train.shape[0]
        yhat=c.RF_predict(X_test,treelist)
        sum_eout+=np.sum(yhat!=Y_test)/Y_test.shape[0]
        
    print("Average_Ein:",sum_ein/ntest)
    print("Avergae_Eout:",sum_eout/ntest)
    
            
run()
