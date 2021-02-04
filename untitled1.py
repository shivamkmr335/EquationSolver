import pandas as pd
from sklearn import svm
from sklearn import metrics
import joblib


dataframe=pd.read_csv('train_final.csv')
dataframe=dataframe.sample(frac=1).reset_index(drop=True)
print(dataframe)

X=dataframe.drop(['784'],axis=1)
Y=dataframe['784']
X_train,Y_train=X[0:44000],Y[0:44000]
X_test,Y_test=X[44000:],Y[44000:]

model=svm.SVC(kernel="linear")
model.fit(X_train,Y_train)
joblib.dump(model,"model/svm_1.sav")
loadmodel=joblib.load("model/svm_1.sav")
predictions=loadmodel.predict(X_test)
print(predictions)
print("model accuracy",metrics.accuracy_score(Y_test,predictions))