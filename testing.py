from pandas import read_csv
from sklearn.metrics import accuracy_score
import joblib

dataset = read_csv('/home/vasanth/tutorial/airflow/simple_ml_project/test_data.csv', index_col = 0)

array = dataset.values

X_test = array[:,0:4]
Y_test = array[:,4]

model = joblib.load('/home/vasanth/tutorial/airflow/simple_ml_project/model_joblib')

predictions = model.predict(X_test)

print('accuracy score is:', accuracy_score(Y_test, predictions))
