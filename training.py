from pandas import read_csv
from sklearn.svm import SVC
import joblib

dataset = read_csv('/home/vasanth/tutorial/airflow/simple_ml_project/train_data.csv', index_col = 0)

print('data set consists of', dataset.head())

array = dataset.values

X_train = array[:,0:4]
Y_train = array[:,4]


model = SVC(gamma='auto')
model.fit(X_train, Y_train)

joblib.dump(model , '/home/vasanth/tutorial/airflow/simple_ml_project/model_joblib')


