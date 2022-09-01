import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
# %matplotlib inline

df = pd.read_csv("Assignment_3_Hitters.csv")

df = df.dropna()

df.iloc[:,0] = df.iloc[:,0].str.replace("-","")

df['League'] = df.League.astype('category').cat.codes
df['NewLeague'] = df.NewLeague.astype('category').cat.codes
df['Division'] = df.Division.astype('category').cat.codes

df.head()

X = df.loc[:, ['AtBat','Hits','HmRun','Runs','RBI','Walks','Years','CAtBat','CHits','CHmRun','CRuns','CRBI','CWalks','League','Division','PutOuts','Assists','Errors','NewLeague']].to_numpy()
Y= df.loc[:, ['Salary']].to_numpy()

sc=MinMaxScaler()
X_transform=sc.fit_transform(X)
Y = sc.fit_transform(Y.reshape(-1, 1))
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X_transform, Y, test_size = 0.2, random_state = 12, shuffle = True)

df_final = pd.DataFrame({'Models': ['Without Regularisation','With Ridge Regularisation & Lamda=0.01','With Ridge Regularisation & Lamda=10','With Lasso Regularisation & Lamda=0.01', 'With Lasso Regularisation & Lamda=10']*1,
                          'Final Weights of train dataset': [0]*5,
                          'Final Weights of test dataset':[0]*5,
                          'Final Bias of train dataset' :[0]*5,
                          'Final Bias of test dataset' :[0]*5
                        })

"""# Batch Gradient Descent"""

def der_lass_reg(x):
    a = np.where(x<0,-1,np.where(x==0,0,1))
    return a

def der_ridge(x):
    a = (2*x)
    return a

def leaky_relu(x):
    x = np.where(x >= 0, x, x * 0.05)     
    return x

def der_leaky_relu(x):
    x  = np.where(x >= 0, 1, 0.05) 
    return x

def grad_w(weight,intercept,x,y,z1):
    fx = leaky_relu(z1)
    fdw = der_leaky_relu(z1)
    return -(y-fx) * fdw * x

def grad_b(weight,intercept,x,y,z1):
    fx = leaky_relu(z1)
    fdb = der_leaky_relu(z1)
    return -(y-fx) * fdb

def batch_gradient_descent(x,y, lr, dt):
    weight =np.random.randn(x.shape[1])
    #weight =np.ones(x.shape[1])
    intercept=0
    learning_rate = lr  #0.00001
    max_epochs = 500
    wtlst = []
    biaslst = []
    n = len(x)
    linear_loss = []

    for i in range(max_epochs):
        dw,db,loss = 0,0,0
        for i in range(0, len(x)):
            x = X[i]
            y = Y[i]
            z1 =  np.dot(weight,x) + intercept
            loss += (leaky_relu(z1)-y)**2
            dw += grad_w(weight,intercept,x,y,z1)
            db += grad_b(weight,intercept,x,y,z1)
        weight = weight-learning_rate * dw *2/n
        intercept = intercept-learning_rate * db*2/n
        wtlst.append(weight)
        biaslst.append(intercept)
        linear_loss.append(loss/n)

    #plot    
    plt.plot(np.arange(1,max_epochs),linear_loss[1:],color = 'blue')
    plt.title("MSE vs Epoch without regularisation for {} data with Learning Rate {}".format(dt, lr))
    plt.xlabel("Number of Epoch")
    plt.ylabel("MSE")    
    plt.show()

    
    return weight,intercept,wtlst,biaslst

df_final = df_final.astype(object)

"""# For Training Set without Regularisation, testing out multiple learning rate(0.5,0.008,0.0001,0.00001)"""

test_wt,b,wl,bl=batch_gradient_descent(x_train,y_train,0.5, 'training')

test_wt,b,wl,bl=batch_gradient_descent(x_train,y_train,0.008, 'training')

test_wt,b,wl,bl=batch_gradient_descent(x_train,y_train,0.0001,'training')

test_wt,b,wl,bl=batch_gradient_descent(x_train,y_train,0.00001, 'training')

"""# Finally using Learning rate as 0.00001

# For Training Set without Regularisation
"""

w,b,wl,bl=batch_gradient_descent(x_train,y_train,0.00001, 'training')
df_final.at[0,'Final Weights of train dataset'] = list(w)
df_final.at[0,'Final Bias of train dataset'] = list(b)

"""# For Testing Set without Regularisation"""

w,b,wl,bl=batch_gradient_descent(x_test,y_test,0.00001, 'testing')
df_final.at[0,'Final Weights of test dataset'] = list(w)
df_final.at[0,'Final Bias of test dataset'] = list(b)

def lasso_regression(x,y,l,dt):
    weight =np.random.randn(x.shape[1])
    #weight =np.ones(x.shape[1])
    intercept=0
    learning_rate = 0.00001
    max_epochs = 500
    wtlst = []
    biaslst = []
    n = len(x)
    lasso_loss = []

    for i in range(max_epochs):
        dw,db,loss = 0,0,0
        for i in range(0, len(x)):
            x = X[i]
            y = Y[i]
            z1 =  np.dot(weight,x) + intercept
            loss += (leaky_relu(z1)-y)**2 + (l * np.sum(abs(weight)))
            dw += grad_w(weight,intercept,x,y,z1)
            db += grad_b(weight,intercept,x,y,z1)
            lasso_comp = der_lass_reg(weight)
            lasso_comp_bias = der_lass_reg(intercept)
            reg_dw = dw + lasso_comp *l
            reg_db = db + lasso_comp_bias *l
    
        weight = weight-learning_rate * reg_dw *2/n
        intercept = intercept-learning_rate * reg_db*2/n
        wtlst.append(weight)
        biaslst.append(intercept)
        lasso_loss.append(loss/n)

    #plot    
    plt.plot(np.arange(1,max_epochs),lasso_loss[1:])
    plt.title("Loss(MSE+Regularisation) vs Epoch for {} data by Lasso using Lamda {}".format(dt,l))
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")    
    plt.show()

    
    return weight,intercept,wtlst,biaslst

"""# For Training Set with Lasso Regression and Lamda 0.01"""

w,b,wl,bl=lasso_regression(x_train,y_train, 0.01, 'training')
#df_final.iloc[3,1] = w
df_final.at[3,'Final Weights of train dataset'] = list(w)
df_final.at[3,'Final Bias of train dataset'] = list(b)

"""# For Training Set with Lasso Regression and Lamda 10"""

w,b,wl,bl=lasso_regression(x_train,y_train, 10, 'training')
#df_final.iloc[4,1] = w
df_final.at[4,'Final Weights of train dataset'] = list(w)
df_final.at[4,'Final Bias of train dataset'] = list(b)

"""# For Testing Set with Lasso Regression and Lamda 0.01"""

w,b,wl,bl=lasso_regression(x_test,y_test, 0.01, 'testing')
#df_final.iloc[3,2] = w
df_final.at[3,'Final Weights of test dataset'] = list(w)
df_final.at[3,'Final Bias of test dataset'] = list(b)

"""# For Testing Set with Lasso Regression and Lamda 10"""

w,b,wl,bl=lasso_regression(x_test,y_test, 10, 'testing')
#df_final.iloc[4,2] = w
df_final.at[4,'Final Weights of test dataset'] = list(w)
df_final.at[4,'Final Bias of test dataset'] = list(b)

def ridge_regression(x,y,l,dt):
    weight =np.random.randn(x.shape[1])
    #weight =np.ones(x.shape[1])
    intercept=0
    learning_rate = 0.00001
    max_epochs = 500
    wtlst = []
    biaslst = []
    n = len(x)
    ridge_loss = []

    for i in range(max_epochs):
        dw,db,loss = 0,0,0
        for i in range(0, len(x)):
            x = X[i]
            y = Y[i]
            z1 =  np.dot(weight,x) + intercept
            loss += (leaky_relu(z1)-y)**2 + (l * np.sum((weight)**2))
            dw += grad_w(weight,intercept,x,y,z1)
            db += grad_b(weight,intercept,x,y,z1)
            ridge_comp = der_ridge(weight)
            ridge_comp_bias = der_ridge(intercept)
            rd_reg_dw = dw + ridge_comp *l
            rd_reg_db = db + ridge_comp_bias *l
    
        weight = weight-learning_rate * rd_reg_dw *2/n
        intercept = intercept-learning_rate * rd_reg_db*2/n
        wtlst.append(weight)
        biaslst.append(intercept)
        ridge_loss.append(loss/n)

    #plot    
    plt.plot(np.arange(1,max_epochs),ridge_loss[1:])
    plt.title("Loss(MSE+Regularisation) vs Epoch for {} data by Ridge using Lamda {}".format(dt,l))
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")    
    plt.show()

    
    return weight,intercept,wtlst,biaslst

"""# For Training Set with Ridge Regression and Lamda 0.01"""

w,b,wl,bl=ridge_regression(x_train,y_train, 0.01,'training')
#df_final.iloc[1,1] = w
df_final.at[1,'Final Weights of train dataset'] = list(w)
df_final.at[1,'Final Bias of train dataset'] = list(b)

"""# For Training Set with Ridge Regression and Lamda 10"""

w,b,wl,bl=ridge_regression(x_train,y_train, 10,'training')
#df_final.iloc[2,1] = w
df_final.at[2,'Final Weights of train dataset'] = list(w)
df_final.at[2,'Final Bias of train dataset'] = list(b)

"""# For Testing Set with Ridge Regression and Lamda 0.01"""

w,b,wl,bl=ridge_regression(x_test,y_test, 0.01,'testing')
#df_final.iloc[1,2] = w
df_final.at[1,'Final Weights of test dataset'] = list(w)
df_final.at[1,'Final Bias of test dataset'] = list(b)

"""# For Testing Set with Ridge Regression and Lamda 10"""

w,b,wl,bl=ridge_regression(x_test,y_test, 10, 'testing')
#df_final.iloc[2,2] = w
df_final.at[2,'Final Weights of test dataset'] = list(w)
df_final.at[2,'Final Bias of test dataset'] = list(b)

"""# Table creation and saved as CSV"""

print(df_final)
df_final.to_csv('Final_table.csv',index =False)

