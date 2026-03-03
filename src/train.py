#Import Libraries and Functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from preprocess import streamline_features
import joblib

#Process Data
X, Y = streamline_features("./data/credit_risk_dataset.csv")

#Split Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

#Apply Polynomial Features
polynomial = PolynomialFeatures(degree=2)
x_tr = polynomial.fit_transform(x_train)
x_te = polynomial.transform(x_test)

#Scale Data
scaler = StandardScaler()
x_train_final = scaler.fit_transform(x_tr)
x_test_final = scaler.transform(x_te)

#Create and Train Model
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
model.fit(x_train_final, y_train)

#Save Model, Polynomial and Scaler
joblib.dump(model, "./models/c_model.pkl")
joblib.dump(x_test_final, "./models/x_test.pkl")
joblib.dump(y_test, "./models/y_test.pkl")