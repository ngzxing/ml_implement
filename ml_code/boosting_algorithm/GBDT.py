import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GBDT():
    
    def __init__(self):
        pass
    
    def divideData(self,xp,yp):
        m,n=xp.shape
        xpsort=np.sort(xp,axis=0)
        slist=np.array([xpsort[i-1]+xpsort[i]  for i in range(1,m)])/2
        error_min=np.inf
        
        for j in range(n):
            for s in slist[:,j]:
                index1=xp[:,j]>s; index2=True!=index1
                ypDivide1=yp[index1]; ypDivide2=yp[index2]
                r1=np.mean(ypDivide1); r2=np.mean(ypDivide2); 
                error=np.sum((ypDivide1-r1)**2)+np.sum((ypDivide2-r2)**2)
                
                if error<error_min:
                    error_min=error
                    index1_best=index1; index2_best=index2
                    yp1_best=ypDivide1; yp2_best=ypDivide2
                    j_best=j; s_best=s
        
        return xp[index1_best],xp[index2_best],yp1_best,yp2_best,j_best,s_best
    
    def base_tree(self,X,Y,depth):
        m,n=X.shape; d=0
        xplist_temp=[X]; yplist_temp=[Y]
        tree={}
        
        while d<depth:
            d+=1; k=0
            xplist=xplist_temp; yplist=yplist_temp
            xplist_temp=[]; yplist_temp=[]
            
            for xp,yp in zip(xplist,yplist):
                k+=1
                name="{0}.{1}".format(d,k)
                
                if np.all(xp[0]==xp) or (d==depth):
                    tree[name]={"r":np.mean(yp),"son":0}
                    
                else:
                    xp1,xp2,yp1,yp2,j,s=self.divideData(xp,yp)
                    xplist_temp.extend([xp1,xp2]); yplist_temp.extend([yp1,yp2])
                    tree[name]={"j":j,"s":s,"son":2}
                    
        return tree
    
    def predict_basetree(self,X,tree,depth):
        yhat=[];
        
        for i in range(X.shape[0]):
            x=X[i]
            num_son=1
            for d in range(depth):
                father_name="{0}.{1}".format(d+1,num_son)
                father=tree[father_name]
                num_son=0
                
                if father["son"]!=0:
                    j=father["j"]; s=father["s"]
                    branch=int(father_name.split(".")[1])
                    
                    for b in range(0,branch):
                        num_son+=tree["{0}.{1}".format(d+1,b+1)]["son"]
                        
                    if x[j]>s:
                        num_son-=1
                    
                    else:
                        continue
                    
                else:
                    yhat.append(father["r"])
                    break
                    
        return np.array(yhat)
    
    def predict(self,X,etalist,treelist,plist,depth):
        yhat=0
        for eta,tree,p in zip(etalist,treelist,plist):
            yhat+=eta*self.predict_basetree(X @ p.T,tree,depth)
            
        return yhat
       
    def random_projection(self,X,residual):
        m,n=X.shape
        index=np.random.randint(0,m,m)
        #p=np.array([[np.random.choice([0,1],p=[0.9,0.1]) for j in range(n)] for i in range(5)])
        p=[]; index=list(range(n))
        for i in range(5):
            a=[0]*n; r=np.random.choice(index)
            a[r]=1
            p.append(a)
            index.remove(r)
            
        p=np.array(p)
        
        return (X @ p.T),residual,p
        
    def boosting(self,X,Y,m,depth,X_test,Y_test,rate):
        treelist=[]; etalist=[]; sn=0
        residual=Y; plist=[]; Einlist=[]; Eoutlist=[]
        
        for mi in range(m):
            print("\r mi:",mi,end="")
            #Xr=X; ydelta=residual
            Xr,ydelta,p=self.random_projection(X,residual)
            plist.append(p)
            
            tree=self.base_tree(Xr,ydelta,depth=depth)
            gt=self.predict_basetree(Xr,tree,depth)
            etalist.append(1)
            
            treelist.append(tree)
            sn+=rate*self.predict_basetree(Xr,tree,depth)
            residual=Y-sn
            
            if (mi+1)%5==0:
                Einlist.append(1-(np.sum((Y-sn)**2/np.sum((Y-np.mean(Y))**2))))
                yhat=self.predict(X_test,etalist,treelist,plist,depth)
                Eoutlist.append(1-(np.sum((Y_test-yhat)**2/np.sum((Y_test-np.mean(Y_test))**2))))
        
        return etalist,treelist,sn,plist,Einlist,Eoutlist
            
        
    
X=load_boston()["data"]; Y=load_boston()["target"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
G=GBDT()
m=1000
depth=2
etalist,treelist,sn,plist,Einlist,Eoutlist=G.boosting(X_train,Y_train,m,3,X_test,Y_test,rate=0.1)

fig=plt.figure()
ax=fig.add_subplot(111)
hbar=np.arange(1,m,5)
ax.plot(hbar,Einlist)
ax.plot(hbar,Eoutlist)



                    
                
            
        
        