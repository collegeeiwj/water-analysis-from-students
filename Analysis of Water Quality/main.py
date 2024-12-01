## import neccesary or potentially used packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error,confusion_matrix,ConfusionMatrixDisplay,classification_report
import random as rand
######################################################


## reading redaing data from designated CSV file
data_set = pd.read_csv('./data/water_potability.csv')
print(data_set)

## describing data 

print(data_set.head(),'\n')
print(data_set.describe())

## we can see from the data that we have some N/A values, so the data will need cleaning.
## We can that a range of ph values which indicate that we have highly acidic and basic water.
## 



## data set contains NaN values
## we will clean the data by replacing NaN values with appropiate median values
median = data_set.median(0)
data_set.replace(np.NaN,median,inplace=True)


## Our objective is to analyze the potability of water, as such we will split the data set into two sets
## The X set will contain the input features while the Y set will contain the potability column that we want to predict

X = data_set.loc[:,'ph':'Turbidity']
Y = data_set.loc[:,'Potability']

## We will now standardize the data to make it compatible for PCA analysis , Y values range from 0 to 1 so it is already standardized, but will be converted to float instead of int
Y = pd.to_numeric(Y,downcast='float')
X_mean = X.mean(0)
X_standardized = (X - X_mean)/X.std()

## Covariance analysis
cov_matrix =  X_standardized.cov()


## plotting covariance
## we can see that we have low correlation between features which suggests fully independant data, this could be problematic in reducing features
sb.set_theme(rc={'figure.figsize':(9,9)})
sb.heatmap(data=cov_matrix,annot=True,cmap='coolwarm')
plt.title('covariance')
plt.show()


## We will now extract the eigenvectors and eigenvalues from the dataset
eigen_vals,eigen_vecs = np.linalg.eig(cov_matrix)

## sorting eigenvalues and eigenvectors
index = eigen_vals.argsort()[::-1]
eigen_vals = eigen_vals[index]
eigen_vecs = eigen_vecs[index]
print(eigen_vals)

## calculating explained variance ratios 
eigen_vals_sum = eigen_vals.sum()
explained_ratios = np.array(eigen_vals/eigen_vals_sum)

## checking cumulative percentage of eigen_values
print(explained_ratios.cumsum())

## we will take 91% of values which we believe is enough to capture most of the variance 
PCA_eigen_vectors = eigen_vecs.T[0:eigen_vals.__len__()-1]
PCA_eigen_vectors = PCA_eigen_vectors.T

## calculating Prinicpal Components
Zpc =  pd.DataFrame(np.dot(X_standardized,PCA_eigen_vectors))
print(Zpc)

##----------------------------- Now lets create a predictive model---------------------------------


##------------------------------------- model function -----------------------------------
def Create_LogisticRegression_Model(trainLowBound : float, trainHighBound : float, title : str):

    train = rand.uniform(trainLowBound,trainHighBound)
    test = 1-train
    X_train , X_test , Y_train , Y_test = train_test_split(Zpc,Y,train_size=train,test_size=test)

    model = LogisticRegression()
    model.fit(X_train,Y_train)

    Y_pred = model.predict(X_test)

    Y_train_pred = model.predict(X_train)

    plt.figure(figsize=(9,9))
    plt.yticks([0,1])
    plt.plot(Y_pred)
    plt.title(title)
    plt.show()

    
    ## calculating accuracy losses, based on test results we can see that the model is not effective in predicting potability of water
    ## 0 column for negative, 1 column for positive
    ## 0,0 : true negative , 0,1 : false positive , 1,0 : false negative, 1,1 : true positive

    ConfusionMatrixDisplay.from_predictions(Y_train,Y_train_pred,cmap='coolwarm')
    train_loss = pd.DataFrame(confusion_matrix(Y_train,Y_train_pred))
    plt.title(f'{title} Train Loss')
    plt.show()
    
    ConfusionMatrixDisplay.from_predictions(Y_test,Y_pred,cmap='coolwarm')
    test_loss = pd.DataFrame(confusion_matrix(Y_test,Y_pred))
    plt.title(f'{title} Test Loss')
    plt.show()
    
    ## classification reports
    print(f'---------------------- {title} -----------------------------')
    print('train loss:\n',train_loss,'\n',classification_report(Y_train,Y_train_pred),'\n')
    print('test loss:\n',test_loss,'\n',classification_report(Y_test,Y_pred),'\n')
    print(f'---------------------- end of {title}-----------------------------')
#-------------------------------------------- end of model function ------------------------------------------------



Create_LogisticRegression_Model(0.1,0.2,'test 1 low training')

## our model seems to have difficulty detecting drinkable water



## increasing train size--------------------------------------------  
Create_LogisticRegression_Model(0.7,0.8,'test 2 high training')

## now potentially train overfitted for detecting undrinkable water



## lets balance between train and test
## balanced-------------------------------
Create_LogisticRegression_Model(0.5,0.5,'test 3 balanced training')

## our models tends to ignore the drinkable water the higher the training is, we need to lower training
## final optimization -------------------------------
Create_LogisticRegression_Model(0.20,0.30,'test 4')

## the proper range seems to be from 20% to 30% training
## model more proficient at detecting undrinkable water than drinkable
## model is not accurate









































