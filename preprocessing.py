# added the comment
from pandas import read_csv
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
train, test = train_test_split(dataset, test_size=0.3)
train.to_csv('/home/vasanth/tutorial/airflow/simple_ml_project/train_data.csv')
test.to_csv('/home/vasanth/tutorial/airflow/simple_ml_project/test_data.csv')
