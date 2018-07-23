import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df = pd.read_csv('loan_predictor\\train_loan_prediction.csv')

### Data Analysis ###

df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='ApplicantIncome', by='Education')
df['Dependents'].hist(bins=50)
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status', index=['Dependents'], aggfunc=lambda x: x.map({'Y': 1, 'N': 0}).sum())
print('Frequency Table for Credit History:')
print(temp1)

print('\nProbability of getting loan for each Credit History class:')
print(temp2)

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

temp3 = pd.crosstab([df['Credit_History'],df['Gender']], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=True)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
test.apply(lambda x: sum(x.isnull()), axis=0)
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
df.dtypes

### Data Cleaning ###

def cleaning_data(data, columns, flag=0):
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
    data['Self_Employed'].fillna('No', inplace=True)
    data['Dependents'].fillna('0', inplace=True)
    data['Credit_History'].fillna(1, inplace=True)
    data['ApplicantIncome'].fillna(data['ApplicantIncome'].mean(), inplace=True)
    data['CoapplicantIncome'].fillna(data['CoapplicantIncome'].mean(), inplace=True)
    var_mod = ['Gender', 'Dependents', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

    le = LabelEncoder()
    data.dropna(inplace=True)
    if flag==1:
        for i in var_mod[0:-1]:
            data[i] = le.fit_transform(data[i])
        data = data[columns[0:-1]]
    else:
        for i in var_mod:
            data[i] = le.fit_transform(data[i])
        data = data[columns]
    return data


def build_features(data, features):
    data['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    data = data[features]
    return data

columns = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome']

data = cleaning_data(df, columns)

### Model ###

def model_eval(data, model, features):
    train_feature, train_label = build_features(data, features), data['Loan_Status']
    train_x, test_x, train_y, test_y = train_test_split(train_feature, train_label, test_size=0.25)
    predicted = cross_val_predict(model, train_x, train_y, cv=10)
    print('Cross Validation Accuracy- ' + str(metrics.accuracy_score(predicted, train_y)))
    print(metrics.classification_report(predicted, train_y))
    model.fit(train_x, train_y)
    predicted_test = model.predict(test_x)
    print('Test Accuracy- ' + str(metrics.accuracy_score(predicted_test, test_y)))
    return model


### Logistic Regression
model = LogisticRegression(random_state=0)
model = model_eval(df, model, features)

Var_Corr = df.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)


### Random Forest
model1 = RandomForestClassifier(n_estimators=15, random_state=0)
model1 = model_eval(df, model1, features)


### Leaderboard
test = pd.read_csv('loan_predictor\\test_loan_prediction.csv')
submission = pd.read_csv('loan_predictor\\Sample_Submission_loan_prediction.csv')

def output_lb(model, test, features):
    clean_test = cleaning_data(test, columns, 1)
    test_x = build_features(clean_test, features)
    test['Loan_Status'] = model.predict(test_x)
    return test

submission['Loan_ID'] = test['Loan_ID']
submission['Loan_Status'] = test['Loan_Status']
submission.to_csv('loan_predictor\\Sample_Submission_loan_prediction.csv')

