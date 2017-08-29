import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

import pandas as pd





#Wineデータセットを読み込む

df_wine = pd.read_csv(

    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',

    header = None)





#特徴量とクラスラベルを別々に抽出

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values





#トレーニングデータとテストデータに分割

#全体の30%をテストデータにする

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)





#normalization

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)

X_test_norm = mms.transform(X_test)







#standardardization

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)









#L1正則化

print('\n' + '●L1正則化●')

lr1 = LogisticRegression(penalty='l1', C=0.1)

lr1.fit(X_train_norm, y_train)

print('-normalization-')

print('Training accuracy : ', lr1.score(X_train_norm, y_train))

print('Test accuracy : ', lr1.score(X_test_norm,y_test))



lr1.fit(X_train_std, y_train)

print('\n' + '-standardization-')

print('Training accuracy : ', lr1.score(X_train_std, y_train))

print('Test accuracy : ', lr1.score(X_test_std,y_test))







#L2正則化

print("\n" + "●L2正則化●")

lr2 = LogisticRegression(penalty='l2', C=0.1)

lr2.fit(X_train_norm, y_train)

print('-normalization-')

print('Training accuracy : ', lr2.score(X_train_norm, y_train))

print('Test accuracy : ', lr2.score(X_test_norm,y_test))



lr2.fit(X_train_std, y_train)

print('\n' + '-standardization-')

print('Training accuracy : ', lr2.score(X_train_std, y_train))

print('Test accuracy : ', lr2.score(X_test_std,y_test))



print('\n')
