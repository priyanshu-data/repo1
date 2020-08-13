import pandas as pd
vado=pd.read_csv("C:/Users/PRIYA/WINDOWS FILE/Desktop/DATASCIENCE/CASE STUDY/vadodara price prediction/vadodara_house_price_dataset_new.csv")
vado.head()
vado=vado.drop(["yr_built"],axis=1)
vado=vado.drop(["society","office","school","college","hospital","population","railway","airport","on_road","air_quality","restaurant","park"],axis=1)
vado=vado.drop(["furniture","sale_type","amenities","market"],axis=1)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
#encoding category features.
vado.iloc[:,0]=labelencoder.fit_transform(vado.iloc[:,0].values)
vado.iloc[:,1]=labelencoder.fit_transform(vado.iloc[:,1].values)
vado.iloc[:,2]=labelencoder.fit_transform(vado.iloc[:,2].values)
X=vado.iloc[:250,:]
Y=vado.iloc[250:,:]
Y.drop(['price'],axis=1,inplace=True)
x_train=X.drop(["price"],axis=1)
y_train=X['price']
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
print("[1]randomforest training accuracy:",rf.score(x_train,y_train))
y_pred=rf.predict(Y)
y_pred
import pickle
filename = 'vadodaratest_house_model_rf.pkl'
pickle.dump(rf, open(filename, 'wb'))