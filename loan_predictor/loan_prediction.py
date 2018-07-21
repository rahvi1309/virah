import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('loan_predictor\\train_loan_prediction.csv')
data1 = data.dropna()
train_feature, train_label = data.iloc[:,::-1], data.iloc[:, -1]

train_x, test_x, train_y, test_y = train_test_split(train_feature, train_label, test_size=0.3)

