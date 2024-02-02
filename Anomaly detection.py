import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('credit_card_data.csv')

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

isolation_forest = IsolationForest(contamination=0.01, random_state=42)
y_pred = isolation_forest.fit_predict(X_test)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
