import pandas as pd
datas = pd.read_csv("sn1500.csv")
print(datas)
datas.drop(["Unnamed: 0"],axis=1,inplace=True)
print(datas["Expresion"].value_counts())
import seaborn as sns
print(sns.countplot(data=datas,x="Expresion"))

from sklearn.model_selection import train_test_split
X = datas.new_com
y = datas.Expresion

X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver='lbfgs')

from sklearn.pipeline import Pipeline
model = Pipeline([('vectirizer',tvec),('classifier',clf2)])

model.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix

prediction = model.predict(X_test)

ex = ["i think he is joyful"]
re = model.predict(ex)
if re == 0:
    print("Sad")
elif re == 4:
    print("happy")
else:
    print("something went right..")

from sklearn.metrics import accuracy_score
print("accuracy score is :- " ,accuracy_score(prediction,y_test))

co = confusion_matrix(prediction,y_test)
print(co)