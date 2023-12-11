"""
from sklearn.datasets import load_wine


bost=load_wine()

x=bost.data

y=bost.target

print(x[0:9])

print(x.shape)
print("]]]]]]]]]]]]]]]]]]]")
print(y[0:9])
print(y.shape)
"""
"""
from sklearn.datasets import load_digits

digits=load_digits()

x=digits.data
y=digits.target

import matplotlib.pyplot as plt

plt.gray()
for g in range(10):
    print("image number of :",g)
    plt.matshow(digits.images[g])

    plt.show()
"""
"""
from sklearn.datasets import load_wine

data=load_wine()

x=data.data
y=data.target

print("x colums are :",data.feature_names)

print("y colums are :",data.target_names)

"""
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
import numpy as np

braest=load_breast_cancer()

x=braest.data
y=braest.target

impute=SimpleImputer(missing_values=np.nan,strategy='mean')
imputrx=impute.fit(x)
fx=imputrx.transform(x)

print(fx[0:30])