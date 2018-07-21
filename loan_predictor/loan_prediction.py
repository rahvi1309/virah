import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('loan_predictor\\train_loan_prediction.csv')
data1 = data.dropna()
train_feature, train_label = data.iloc[:,::-1], data.iloc[:, -1]

train_x, test_x, train_y, test_y = train_test_split(train_feature, train_label, test_size=0.3)

data['ApplicantIncome'].hist(bins=50)
data.boxplot(column='ApplicantIncome')
data.boxplot(column='ApplicantIncome', by='Education')
data['LoanAmount'].hist(bins=50)
temp1 = data['Credit_History'].value_counts(ascending=True)
temp2 = data.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y': 1, 'N': 0}).mean())
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

temp3 = pd.crosstab([data['Credit_History'],data['Gender']], data['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=True)
