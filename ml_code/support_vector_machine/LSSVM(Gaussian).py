class Lssvm():
    
    def __init__(self):
        self.classifier=lambda kernel,beta : np.sign(kernel @ beta)
        
    def get_data(self,filepath):
        with open(filepath) as f:
            data=np.array([ line.split() for line in f.readlines()]).astype(float)
        
        return data[:400,:-1],data[400:,:-1],data[:400,-1],data[400:,-1]
    
    def Gaussian(self,X,gamma):
        m,n=X.shape
        kernel=np.identity(m)
        for i in range(m):
            result=np.exp(-1*gamma*(np.linalg.norm(X-X[i,:],axis=1)**2))
            kernel[i,:]=result
            kernel[:,i]=result
            
        return kernel
    
    def Gaussian_predict(self,X_train,X_test,gamma):
        m1,n1=X_test.shape
        kernel=[]
        for i in range(m1):
            result=np.exp(-1*gamma*(np.linalg.norm(X_train-X_test[i,:],axis=1)**2) )
            kernel.append(result)
        
        return np.array(kernel)
    
    def trainer(self,X,Y,kernel,Lambda):
        m,n=X.shape
        beta=np.linalg.pinv(Lambda*np.identity(m)+kernel) @ Y
        return beta 
    
    
    def error(self,Y,yhat):
        return np.sum(yhat!=Y)/Y.shape[0]
    
    
    def routine(self,X_train,X_test,Y_train,Y_test,gamma,Lambda):
        kernel_train=self.Gaussian(X_train,gamma=gamma)
        kernel_test=self.Gaussian_predict(X_train,X_test,gamma=gamma)
        beta=self.trainer(X_train,Y_train,kernel_train,Lambda=Lambda)
        print("   Ein:",self.error(Y_train,self.classifier(kernel_train,beta)))
        print("   Eout:",self.error(Y_test,self.classifier(kernel_test,beta)))
        
    
    def run(self):
        X_train,X_test,Y_train,Y_test=self.get_data("lssvm.txt")
        
        for gamma in [32,2,0.125]:
            for Lambda in [0.001,1,1000]:
                print("Lambda:",Lambda)
                print("gamma:",gamma)
                self.routine(X_train,X_test,Y_train,Y_test,gamma,Lambda)
    
L=Lssvm()
L.run()