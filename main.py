#-*- coding:utf-8 -*-
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from sklearn.externals import joblib
import pylab as pl
import matplotlib.pyplot as mp, seaborn

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

def train():
    dsp_data = pd.read_csv('./dsp_data.csv')
    imp_data = pd.read_csv('./imp_data.csv')
    dsp_data = dsp_data.drop_duplicates(subset=['idx'], keep='first')
    imp_data = imp_data.drop_duplicates(subset=['idx'], keep='first')
    data = pd.merge(dsp_data,imp_data,how='inner',on='idx')
    temp = data.isnull().any()
    print(temp)
    data = data.dropna(axis=0,how='any')
    
    data['time_dif'] = pd.Series(map(lambda x,y:x-y, data['imp_time'],data['dsp_time']))
    data['ip_dif'] = pd.Series(map(lambda x,y:ip(x,y), data['dsp_ip'],data['imp_ip']))
    data['hour'] = pd.to_datetime(data['dsp_time'],unit='s').dt.hour
    data['category'] = pd.Series(map(lambda x:int(''.join(list(x)[-4:-1])), data['mcategory']))
    data['count_mid'] = data.groupby(['mid'])['mid'].transform('count')
    data['count_adid'] = data.groupby(['adid'])['adid'].transform('count')
    
    x_train, x_test, y_train, y_test = train_test_split(data[['time_dif','imp_time','hour',
                                                              'category','count_mid',
                                                              'count_adid','ip_dif']],
                                                        data['label'],
                                                        test_size = 0.2, 
                                                        random_state = 0)
    
    print(y_train.groupby(data['label']).count())
    print(y_test.groupby(data['label']).count())
    
    #特征分布
    df = data[['time_dif','imp_time','hour','category','count_mid','count_adid','ip_dif','label']]
    corrmat = df.corr()
    seaborn.heatmap(corrmat, center=0, annot=True)
    mp.show()
    df.describe()
    df.hist()
    pl.show()
    
    ##决策树
    tree = DecisionTreeClassifier(criterion='gini', max_depth = 15, random_state = 0)#'entropy'
    model = tree.fit(x_train, y_train)
    
    ##模型保存与加载
    #joblib.dump(model, './tree.m')
    #clf = joblib.load("tree.m")
    #y_predict = clf.predict(x_test)
    
    ##朴素贝叶斯模型
    #nb = BernoulliNB(alpha = 1.0, binarize = 0.0005)
    #model = nb.fit(x_train, y_train)
    
    ##GBDT
    #gbdt = GradientBoostingClassifier(learning_rate=0.01, 
                                      #n_estimators =100, 
                                      #max_depth=3,
                                      #min_samples_split = 50, 
                                      #loss = 'deviance', 
                                      #random_state = 0)
    #model = gbdt.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    
    ##生成决策图
    #dot_data = export_graphviz(tree, 
                               #out_file=None,
                               #feature_names=['time_dif','imp_time','hour','category','count_mid',
                                              #'count_adid','ip_dif'],
                               #class_names='label',
                               #filled=True, 
                               #rounded=True,
                               #special_characters=True)
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #graph.get_nodes()[8].set_fillcolor("#FFF2DD")
    #graph.write_png("out2.png")
    
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1mean = f1_score(y_test, y_predict)
    print(classification_report(y_test, y_predict))
    
    fpr,tpr,thre = roc_curve(y_test,y_predict)
    aucc = auc(fpr,tpr)
    plt.plot(fpr,tpr,color = 'darkred',label = 'roc area:(%0.2f)'%aucc)
    plt.plot([0,1],[0,1],linestyle = '--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc_curve')
    plt.legend(loc = 'lower right')
    plt.show()
    
if __name__ == '__main__':
    train()