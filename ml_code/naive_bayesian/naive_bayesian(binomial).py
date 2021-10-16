import numpy as np

class binomial_nb():
    
    def __init__(self,Lambda=1):
        self.Lambda=Lambda
        
    def getdata(self):
        filepath="email\\"
        data=[]; Y=np.hstack((np.zeros(25),np.ones(25)))
        
        for folder in ["ham","spam"]:
            for i in range(1,26):
                with open(filepath+folder+"\\"+str(i)+".txt") as f:
                    data.append([word.lower() for line in f.readlines() \
                                 for word in re.findall(r'[a-zA-Z]+|[!~@#$%&*({:"<?,./`|]',line)])
        
        return data,Y
    
    def createSetVocab(self,data):
        self.SetVocab=list(set([word for document in data for word in document]))
    
    def createX(self,data):
        return np.array([[1 if vocab in document else 0 for vocab in self.SetVocab] for document in data ])
    
    def fit(self,data,Y,index):
        cls=set(Y); m=Y.shape[0]
        X=self.createX(data)[index]
        pk=np.array([np.sum(Y==k) for k in cls])/m
        pc1=np.array([(np.sum(X[Y==k],axis=0)+self.Lambda)/((pk[int(k)]*m)+(2*self.Lambda)) for k in cls]).T
        
        return np.log(pk),np.log(pc1),np.log(1-pc1),X
    
    def predict(self,X,logpk,logpc1,logpc0):
        return np.argmax(((X @ logpc1)+((1-X) @ logpc0))+logpk,axis=1)
            

def run():
    nb=binomial_nb(Lambda=1)
    data,Y=nb.getdata()
    nb.createSetVocab(data)
    
    index=np.arange(0,50)
    np.random.shuffle(index)
    logpk,logpc1,logpc0,X=nb.fit(data,Y[index],index)
    yhat=nb.predict(X,logpk,logpc1,logpc0)
    print("Ein:",np.sum(yhat!=Y[index])/50)
    
    data_test=["promotion ! free 80 % just need $ 200 buy now ! Hey ! how are you nice to meet you ".split()]
    print(data_test)
    X_test=nb.createX(data_test)
    print(nb.predict(X_test,logpk,logpc1,logpc0))
    
    
    
run()
        