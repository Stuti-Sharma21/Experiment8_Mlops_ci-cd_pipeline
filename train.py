import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.meterics import accuracy_score
import jobilb

df.read_csv("iris.csv")
df=df.drop("Id",axis=1)

x=df.drop("Species",axis=1)
y=df["Species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

model=LogisticRegression()
model.fit(s_train,y_train)
pred=model.pred(x_test)
accuracy=accuracy_score(pred,y_test)
print(accuracy)

joblib.dump(model,"model.pkl")
print("model trained and saved successfully")