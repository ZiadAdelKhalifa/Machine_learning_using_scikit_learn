# Import Libraries
from sklearn.datasets import load_iris
#----------------------------------------------------

#load iris data

IrisData = load_iris()

#X Data which is the features
X = IrisData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , IrisData.feature_names)

#y Data target is the values of y
y = IrisData.target
print('y Data is \n' , y[:150])
print('y shape is ' , y.shape)
print('y Columns are \n' , IrisData.target_names)