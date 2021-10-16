#logistic regression(classify breast cancer)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer,load_iris
import numpy as np



class Logistic_Regression():
    def __init__(self,lambda_=0,alpha=0.1,num_iter=1000,therhold=0.5,softmax=False):
        self.lambda_=lambda_
        self.alpha=alpha
        self.num_iter=num_iter
        self.therhold=therhold
        self.softmax=softmax
        self.multiclass_s=False
        self.multiclass=False
        self.sigmoid=lambda X,theta : 1/(1+np.exp(X @ theta))
        self.trans_X=lambda X : np.hstack((X,np.ones(X.shape[0]).reshape(X.shape[0],-1)))
        
    @staticmethod
    def sigmoid_m(z):
        z_max=np.max(z,axis=-1,keepdims=True)
        fx=np.exp(z-z_max)
        return fx/np.sum(fx,axis=-1,keepdims=True)
    
    @staticmethod
    def encode(y):
        n_cls=len(set(y))
        return np.eye(n_cls)[y],n_cls
    
    def gradient_descent(self,X,y):
        m,n=X.shape ; num=0
        theta=np.zeros(n)
        while num<=self.num_iter:
            num+=1
            theta-=(self.alpha*1/m)*((X.T @ (y-self.sigmoid(X,theta)))+(self.lambda_*theta))
        return theta
        
    def fit(self,X,y):
        X=self.trans_X(X)
        if len(set(y))>2:
            if self.softmax:
                self.multiclass_s=True
                self.fit_softmax(X,y)
            else:
                self.multiclass=True
                self.fit_onevone(X,y)
            return
        self.theta=self.gradient_descent(X,y)
        
        
    def fit_softmax(self,X,y):
        m,n=X.shape ; num=0
        y,n_cls=self.encode(y)
        self.theta=np.zeros((n,n_cls))
        while num <=self.num_iter:
            num+=1
            theta=np.copy(self.theta)
            pi_x=self.sigmoid_m(X @ self.theta)
            self.theta-=(self.alpha*1/m)*(X.T @ (pi_x-y)+(self.lambda_*theta))
            if np.allclose(theta,self.theta):
                break
                
    def fit_onevone(self,X,y):
        cls,n_cls=np.unique(y,return_counts=True)
        m,n=X.shape
        self.cls_group=[(cls[i],j) for i in range(cls.shape[0]) for j in cls[i+1:]]
        self.theta=np.zeros((len(self.cls_group),n))
        for k in range(len(self.cls_group)):
            group=self.cls_group[k]
            index=[y[i] in group for i in range(m)]
            x_=X[index] ; y_=np.where(y[index]==group[0],1,0)
            self.theta[k]=self.gradient_descent(x_,y_)
            
                       
    def predict(self,X):
        X=self.trans_X(X)
        if self.multiclass:
            yhat=self.predict_onevone(X)
            return yhat
        if self.multiclass_s:
            yhat=self.predict_softmax(X)
            return yhat
        posterior=self.sigmoid(X,self.theta)
        yhat=np.where(posterior>=self.therhold,1,0)
        return yhat
    
    def predict_softmax(self,X):
        pi_x=self.sigmoid_m(X @ self.theta)
        yhat=np.argmax(pi_x,axis=1)
        return yhat
    
    def predict_onevone(self,X):
        n_k=len(self.cls_group)
        yhat_=np.zeros((X.shape[0],n_k))
        for k in range(n_k):
            group=self.cls_group[k]
            y_=np.where(self.sigmoid(X,self.theta[k])>= self.therhold,group[0],group[1])
            yhat_[:,k]=y_
        yhat=np.array([list(set(yhat_[i,:]))[np.argmax(
            np.unique(yhat_[i,:],return_counts=True)[1])] for i in range(X.shape[0])])
        return yhat
        
        
    def score(self,X,y):
        m,n=X.shape
        yhat=self.predict(X)
        error=np.where(yhat==y,0,1)
        error_rate=np.sum(error)/m
        print('yhat: {}\ny: {}'.format(yhat,y))
        return 1-error_rate
    
    def confusion_matrix(self,yhat,y):
        ctg=np.unique(y)
        m=y.shape[0]
        mat=np.array([[np.sum(np.where((yhat==ki) & (y_train==k),1,0)) for ki in ctg] for k in ctg])
        confusion_matrix=[]
        for k in ctg:
            percision=mat[k,k]/np.sum(mat[:,k])
            recall=mat[k,k]/np.sum(mat[k,:])
            F1_score=1/(0.5*(1/percision)+(1/recall))
            confusion_matrix.append([percision,recall,F1_score])
        
        return (pd.DataFrame(np.array(confusion_matrix).T,index=['percision','recall','F1_score'],columns=ctg),mat)

def load_digit():
    import os ; import re
    file_path='C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\digit_data\\'    
    data=np.array([(re.findall(r'\w',open(file_path+file+name).read()
                  ),re.findall(r'^\w',name)) for file in ['testDigits\\','trainingDigits\\'
                                                         ] for name in os.listdir(file_path+file)])
    X=[] ; y=[]
    for i in range(data.shape[0]):
        X.append(np.array(data[i][0]).astype(int))
        y.append(np.array(data[i][1]).astype(int))
    return np.array(X),np.array(y).reshape(1,-1)[0]


if __name__=='__main__':
    data=load_breast_cancer()
    y,X=data.target,data.data
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=100)


    clf=Logistic_Regression(lambda_=0.6,alpha=0.1,therhold=0.5,num_iter=10000)
    clf.fit(X_train,y_train)
    yhat=clf.predict(X_train)
    conf_m,mat=clf.confusion_matrix(yhat,y_train)
    print(conf_m)
    print(mat)
    X,y=load_digit()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=100)
    clf=Logistic_Regression(lambda_=0.6,alpha=0.5,therhold=0.1,num_iter=1000,softmax=True)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))



print((yhat==1)[100])
print((y_train==1)[100])
print((y_train==1 *(yhat==1))[100])



