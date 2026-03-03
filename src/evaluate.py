#Import Libraries and Variables
from sklearn.metrics import precision_score, confusion_matrix, classification_report
import joblib

#Import Test Data
x_test = joblib.load("./models/x_test.pkl")
y_test = joblib.load("./models/y_test.pkl")

#Write Function for Evaluation
def predict_output():
    #Load Model
    model = joblib.load("./models/c_model.pkl")

    #Predict Output
    y_pred = model.predict(x_test)

    #Compute Precision Score, Confusion Matrix and Classification Report
    score = precision_score(y_test, y_pred)
    c_matrix = confusion_matrix(y_test, y_pred)
    c_report = classification_report(y_test, y_pred)

    #Print Output
    labels = {1: "Default", 0: "No Default"}
    print("Actual\n")
    print([labels[k] for k in y_test[0:50]])
    print("Predicted\n")
    print([labels[p] for p in y_pred[0:50]])
    print(f"Precision Score of the model: {round(score, 2)}")
    print("Confusion Matrix\n")
    print(c_matrix)
    print("Classification Report\n")
    print(c_report)

#Run Evaluation Function
predict_output()