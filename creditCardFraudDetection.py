from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
naivea=0;
naivemse=0;
adaa=0;
adamse=0;
df=pd.read_csv('creditcard.csv')
print('NAIVE BAYES\n')
kf = KFold(n_splits=10)
cf=[]
cr=[]
a=[]
mse=[]
i=1;
for trainx, testx in kf.split(df):
    train=df.iloc[trainx,:]
    test=df.iloc[testx,:]
    trainTargets = np.array(train['Class']).astype(int)
    testTargets = np.array(test['Class']).astype(int)
    features = df.columns[0:-1]
    model = GaussianNB()
    model.fit(train[features], trainTargets)
    expected = trainTargets
    predicted = model.predict(train[features])
    target_names=['Not Fraudulent','Fraudulent']
    print('Fold ',i,' Done');
    i+=1;
print('\nConfusion Matrix: \n',metrics.confusion_matrix(expected, predicted))
print('\n',metrics.classification_report(expected, predicted,target_names=target_names))
naivea=metrics.accuracy_score(expected, predicted)
naivemse=metrics.mean_squared_error(expected, predicted)
print('\nAccuracy: ',naivea)
print('MSE: ',naivemse)

p=[[0,0],[0,0]]
q=0
r=0
s=0
for i in cf:
    for j in range(len(i)):
        for k in range(len(i[j])):
            p[j][k]+=i[j][k]
for j in range(len(p)):
    for k in range(len(p[j])):
        p[j][k]=p[j][k]/10;

fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
roc_auc = metrics.auc(fpr, tpr)
plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('\nADABOOST\n')
kf = KFold(n_splits=10)
cf=[]
cr=[]
a=[]
mse=[]
i=1
for trainx, testx in kf.split(df):
    train=df.iloc[trainx,:]
    test=df.iloc[testx,:]
    trainTargets = np.array(train['Class']).astype(int)
    testTargets = np.array(test['Class']).astype(int)
    features = df.columns[0:-1]
    model = AdaBoostClassifier(n_estimators=2,
                         learning_rate=1,
                         random_state=0)
    model.fit(train[features], trainTargets)
    expected = trainTargets
    predicted = model.predict(train[features])
    target_names=['Not Fraudulent','Fraudulent']
    print('Fold ',i,' Done');
    i+=1;
print('\nConfusion Matrix: \n',metrics.confusion_matrix(expected, predicted))
print('\n',metrics.classification_report(expected, predicted,target_names=target_names))
adaa=metrics.accuracy_score(expected, predicted)
adamse=metrics.mean_squared_error(expected, predicted)
print('\nAccuracy: ',adaa)
print('MSE: ',adamse)
print('\n');
if(adaa>naivea and adamse<naivemse):
    print('AdaBoost has higher accuracy and lower error rate');
elif(adaa<naivea and adamse<naivemse):
    print('Naive Bayes has higher accuracy but AdaBoost lower error rate');
elif(adaa<naivea and adamse>naivemse):
    print('Naive Bayes has higher accuracy and lower error rate');
elif(adaa>naivea and adamse<naivemse):
    print('AdaBoost has higher accuracy but Naive Bayes lower error rate');
if(adaa==naivea):
    print('They both have same accuracy');
if(adamse==naivemse):
    print('They both have same error rate');

p=[[0,0],[0,0]]
q=0
r=0
s=0
for i in cf:
    for j in range(len(i)):
        for k in range(len(i[j])):
            p[j][k]+=i[j][k]
for j in range(len(p)):
    for k in range(len(p[j])):
        p[j][k]=p[j][k]/10;

fpr, tpr, thresholds = metrics.roc_curve(expected, predicted)
roc_auc = metrics.auc(fpr, tpr)
plt.title('ROC-AUC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
