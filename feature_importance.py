#-*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def ip(dsp_ip,imp_ip):
    level = 0
    dsp_ip = dsp_ip.split('.')
    imp_ip = imp_ip.split('.')
    if dsp_ip[:1] == imp_ip[:1]:
        level = 1
    if dsp_ip[:2] == imp_ip[:2]:
        level = 2
    if dsp_ip[:3] == imp_ip[:3]:
        level = 3
    if dsp_ip[:4] == imp_ip[:4]:
        level = 4
    return level

dsp_data = pd.read_csv('./dsp_data.csv')
imp_data = pd.read_csv('./imp_data.csv')
dsp_data = dsp_data.drop_duplicates(subset=['idx'], keep='first')
imp_data = imp_data.drop_duplicates(subset=['idx'], keep='first')
data = pd.merge(dsp_data,imp_data,how='inner',on='idx')
temp = data.isnull().any()
print(type(temp))
print(temp)
data = data.dropna(axis=0,how='any')

data['time_dif'] = pd.Series(map(lambda x,y:x-y, data['imp_time'],data['dsp_time']))
data['ip_dif'] = pd.Series(map(lambda x,y:ip(x,y), data['dsp_ip'],data['imp_ip']))
data['hour'] = pd.to_datetime(data['dsp_time'],unit='s').dt.hour
data['category'] = pd.Series(map(lambda x:int(''.join(list(x)[-4:-1])), data['mcategory']))
data['count_mid'] = data.groupby(['mid'])['mid'].transform('count')
data['count_adid'] = data.groupby(['adid'])['adid'].transform('count')

x_train, x_test, y_train, y_test = train_test_split(data[['time_dif','imp_time','hour','category',
                                                          'count_mid','count_adid','ip_dif',]],
                                                    data['label'],
                                                    test_size = 0.2, 
                                                    random_state = 0)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))

def feature_importance(features_num=7):
    forest = RandomForestClassifier(n_estimators=500,random_state=0,n_jobs=-1,max_features=7)
    forest.fit(x_train,y_train)
    y_true, y_pred = y_test, forest.predict(x_test)
    print(classification_report(y_true, y_pred))
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    print("----the importance of features and its importance_score------")
    j=1
    features_names=[]
    im_list= []
    for i in indices[0:features_num]:
        f_name = x_train.columns.values[i]
        print(j,f_name,importance[i])
        features_names.append(x_train.columns.values[i])
        im_list.append(importance[i])
        j+=1
    draw_importance(features_names,im_list)

def draw_importance(features,importances):
    indices = np.argsort(importances)
    print(indices)
    print(features)
    plt.title('Feature Importances')
    autolabel(plt.bar(range(len(indices)), np.array(importances)[indices], color='b', align='center'))
    plt.xticks(range(len(indices)), np.array(features)[indices])
    plt.ylim(0,1)
    plt.xlabel('Relative Importance')
    plt.show()

if __name__=="__main__":
    feature_importance()
