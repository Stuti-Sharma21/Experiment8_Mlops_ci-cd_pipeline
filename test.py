import pandas as pandas
import joblib
from sklearn.meterics import accuracy_score

model=joblib.load('model.pkl')

data=pd.read_csv('iris.csv')
data=data.drop("Id",axis=1)

x=data.drop("Species",axis=1)
y=data['species']

print(x.head())

pred=model.predict(x)
accuracy=accuracy_score(pred,y)
print(accuracy_score)

assert len(pred)==len(y)
assert accuracy=0.50

print("All Test Passed")


