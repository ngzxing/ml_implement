# support vector machine (with polynomial kernel)

import numpy as np
import time 
from sklearn.svm import SVC

class smo():
    
    def __init__(self,C=np.inf,toler=1e-3,eps=1e-3):
        self.C=C
        self.toler=toler
        self.eps=eps
        self.E=lambda Y,K,S: ((Y*S.alpha) @ K)+S.b - Y
        self.polykernel=lambda X1,X2,eta,gamma,degree: (eta+gamma*(X1 @ X2.T))**degree
        self.objf=lambda Y,alpha,S: 0.5*((alpha*Y) @ S.K @ (alpha*Y))- np.sum(alpha)
        
    def updateb(self,Y,i,j):
        b=lambda Y,k,S: Y[k]-((Y*S.alpha) @ S.K[k])
        
        if (self.alpha[i]<self.C) or (self.alpha[i]>0):
            return b(Y,i,self)
        
        elif (self.alpha[j]<self.C) or (self.alpha[j]>0):
            return b(Y,j,self)
        else:
            return (b(Y,i,self)+b(Y,j,self))/2
        
    def takestep(self,Y,i,j):
        if i==j:
            return 0
        
        alphaiold=self.alpha[i]; alphajold=self.alpha[j]; Ei=self.Elist[i]; Ej=self.Elist[j]
        
        if Y[i] != Y[j]:
            L=max(0,alphajold-alphaiold)
            H=min(self.C,self.C+alphajold-alphaiold)
        else:
            L=max(0,alphajold+alphaiold-self.C)
            H=min(self.C,alphajold+alphaiold)
                        
        if L==H:
            return 0
        
        eta=self.K[i,i]+self.K[j,j]-(2*self.K[i,j])
        
        if eta>0:
            alphaj=alphajold+(Y[j]*(Ei-Ej)/eta)
            
            if alphaj>H:
                self.alpha[j]=H
            elif alphaj<L:
                self.alpha[j]=L
            else:
                self.alpha[j]=alphaj
                
        else:
            alpha_temp=self.alpha.copy()
            alpha_temp[j]=L
            Lobj=self.objf(Y,alpha_temp,self)
            alpha_temp[j]=H
            Hobj=self.objf(Y,alpha_temp,self)
            
            if (Lobj<Hobj-self.eps):
                self.alpha[j]=L
                
            elif (Lobj>Hobj+self.eps):
                self.alpha[j]=H
                
            else:
                pass
        
        if abs(self.alpha[j]-alphajold)<self.eps*(self.alpha[j]+alphajold+self.eps):
            return 0
        
        self.alpha[i]+=Y[i]*Y[j]*(alphajold-self.alpha[j])
        self.b=self.updateb(Y,i,j)
        self.Elist=self.E(Y,self.K,self)
        
        return 1
            
    
    
    def examExample(self,i,Y,m):
        Ei=self.Elist[i]
        yi=Y[i]
        
        if ((Ei*yi<-self.toler) and (self.alpha[i]<self.C)) or ((Ei*yi>self.toler) and (self.alpha[i]>0)):
            indexs1=np.nonzero( (self.alpha>0)*(self.alpha<self.C) )[0]
        
            if indexs1.shape[0]>1:
                print(1)
                if Ei>0:
                    j=np.argmin(self.Elist)
                else:
                    j=np.argmax(self.Elist)        
                    
                if self.takestep(Y,i,j):
                    return 1

            
            if indexs1.shape[0]>0:
                np.random.shuffle(indexs1)
                if self.takestep(Y,i,indexs1[0]):
                    return 1
            print(2)
            indexs2=np.setdiff1d(np.arange(0,m),indexs1)
            np.random.shuffle(indexs2)
            print(3)
            if self.takestep(Y,i,indexs2[0]):
                return 1
        
        return 0
                
        
    def fit(self,X,Y,num_iter=1000,eta=1,gamma=1,degree=2):
        n=0; pair=0; wholeset=True
        m=Y.shape[0]
        self.K=self.polykernel(X,X,eta,gamma,degree)
        self.alpha=np.zeros(m)
        self.b=0
        self.Elist=self.E(Y,self.K,self)
        
        while (n<num_iter) and ( (pair>0) or wholeset):
            print(n)
            pair=0; n+=1
            
            if wholeset:
                indexs=list(range(m))
            
            else:
                indexs=np.nonzero( (self.alpha>0)*(self.alpha<self.C) )[0]
            
            for index in indexs:
                pair+=self.examExample(index,Y,m)
            
            if wholeset:
                wholeset=False
            elif pair==0:
                wholeset=True
                
        return self.alpha,self.b

def getdata(filepath,vs):
        with open(filepath) as file:
            data=np.array([ line.split() for line in file.readlines()]).astype(float)
            
        return data[:,1:],np.where(data[:,0]==vs,1,-1)

def run():
    #X=np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
    #Y=np.array([-1,-1,-1,1,1,1,1]) 
    X,Y=getdata("features.train.txt",0) ; 
   

    start=time.time()
    svm=smo(C=0.1,toler=0.1,eps=0.01)
    alpha,b=svm.fit(X,Y,num_iter=1000)
    end=time.time()
    print("my alpha:",np.sum(alpha))
    print("my time",end-start,"s")
    
    
run()