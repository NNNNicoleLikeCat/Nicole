import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

column_n=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

tr_d=pd.read_csv('adult.data', header=None, names=column_n, na_values=' ?', skipinitialspace=True)
te_d=pd.read_csv('adult.test', header=None, names=column_n, na_values=' ?', skipinitialspace=True, skiprows=1)
tr_d.dropna(inplace=True)
te_d.dropna(inplace=True)
tr_d['income'] = tr_d['income'].apply(lambda x: 1 if x == '>50K' else 0)
te_d['income'] = te_d['income'].apply(lambda x: 1 if x == '>50K.' else 0)
X_tr=tr_d.drop('income', axis=1)
y_tr=tr_d['income']
X_te=te_d.drop('income', axis=1)
y_te=te_d['income']
categorical_features=X_tr.select_dtypes(include=['object']).columns.tolist()
pre_p=ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
dt_pl=Pipeline(steps=[('preprocessor', pre_p), ('classifier', DecisionTreeClassifier())])
svm_pl=Pipeline(steps=[('preprocessor', pre_p), ('classifier', SVC())])

start_t=time.time()
dt_pl.fit(X_tr, y_tr)
dt_training_t=time.time()-start_t
dt_pre=dt_pl.predict(X_te)
dt_m=classification_report(y_te, dt_pre)

start_t=time.time()
svm_pl.fit(X_tr, y_tr)
svm_training_t=time.time()-start_t
svm_pre=svm_pl.predict(X_te)
svm_m=classification_report(y_te, svm_pre)

print("Decision Tree Metrics:\n", dt_m)
print(f"Decision Tree Training Time: {dt_training_t:.4f} seconds")
print("SVM Metrics:\n", svm_m)
print(f"SVM Training Time: {svm_training_t:.4f} seconds")