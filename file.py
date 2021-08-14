import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pickle

boston = datasets.load_boston()
X= pd.DataFrame(boston.data ,columns=boston.feature_names)
y = boston.target

## Normalizing data 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=255)

#model 

r_f = RandomForestRegressor()
r_f.fit(X_train,y_train)


#model file 

with open('scaler.sav','wb') as f:
    pickle.dump(scaler,f)

with open('model.sav','wb') as f:
    pickle.dump(r_f,f)
    