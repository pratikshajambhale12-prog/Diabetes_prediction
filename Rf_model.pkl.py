import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import SMOTE


df_Diabetes = pd.read_csv('diabetes.csv')
print(df_Diabetes)

df_Diabetes.info()

#extract X and y
X = df_Diabetes.drop(columns=['Outcome'])
y = df_Diabetes['Outcome']

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                                     
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state = 42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled , y_train)

model = RandomForestClassifier(n_estimators= 100,random_state=42)
# fit the model 
model.fit(X_train_balanced, y_train_balanced)

y_pred = model.predict(X_test_scaled)

#evaluate the classification report
print(classification_report(y_test, y_pred))

# save every thing
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))












