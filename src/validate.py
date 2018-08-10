import pandas as pd
from sklearn.metrics import confusion_matrix


dat=pd.read_csv('../data/pred.csv')

dat['validate']= dat['Class']==dat['Expression']

dat_true=dat[dat.validate]
print(float(len(dat_true))/float(len(dat)))


y_true = dat['Expression']
y_pred = dat['Class']
lb=y_true.unique()
X=confusion_matrix(y_true, y_pred,labels=lb)
Y=pd.DataFrame(X, index=lb, columns=lb)
print(Y)

