# -*- coding: utf-8 -*-
"""
Created on Tuesday June 06 12:05:47 2019
@author: Dr Clement Etienam
Question 1: Crete a machine to predict stroke occurence
A Class imbalnce problem with non-stroke (0's) dominating stroke (1's)
"""
from __future__ import print_function
print(__doc__)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.stats import rankdata, norm
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import datetime 
import multiprocessing
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
np.random.seed(5)
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import os
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix,classification_report
## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
##------------------------------------------------------------------------------------

## Start of Programme

print( 'Parallel CCR for TGLF ')


oldfolder = os.getcwd()
cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in parallel '%cores)
#print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))

print('-------------------LOAD FUNCTIONS---------------------------------')
def interpolatebetween(xtrain,cdftrain,xnew):
    numrows1=len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2=np.zeros((numrows1,numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:,i]), cdftrain[:,i],kind='linear')
        cdftest = f(xnew[:,i])
        norm_cdftest2[:,i]=np.ravel(cdftest)
    return norm_cdftest2


def gaussianizeit(input1):
    numrows1=len(input1)
    numcols = len(input1[0])
    newbig=np.zeros((numrows1,numcols))
    for i in range(numcols):
        input11=input1[:,i]
        newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
        newbig[:,i]=newX.T
    return newbig

def getoptimumk(X,i):
#    X=matrix
    distortions = []
    Kss = range(1,60)
    
    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #ncusters2=np.where(distortions == distortions.min())
    
    myarray = np.array(distortions)
    
    knn = KneeLocator(Kss,myarray,curve='convex',direction='decreasing',interp_method='interp1d')
    kuse=knn.knee
    
    # Plot the elbow
    plt.figure(figsize=(12, 5))
    plt.plot(Kss, distortions, 'bx-')
    plt.xlabel('cluster size')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal n_clusters fr machine%d'%(i))
    plt.show()
    return kuse
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("model_run.pdf")
    plt.show()
    
def Replace_clement(data):
    data.gender[data.gender == 'Male'] = 1.0
    data.gender[data.gender == 'Female'] = 2.0
    data.gender[data.gender == 'Other'] = 3.0
    data.gender = data.gender.astype(int)
    data.ever_married[data.ever_married == 'No'] = 1.0
    data.ever_married[data.ever_married == 'Yes'] = 2.0
    data.ever_married = data.ever_married.astype(int)
    data.work_type[data.work_type == 'children'] = 1.0
    data.work_type[data.work_type == 'Private'] = 2.0
    data.work_type[data.work_type == 'Never_worked'] = 3.0
    data.work_type[data.work_type == 'Self-employed'] = 4.0
    data.work_type[data.work_type == 'Govt_job'] = 5.0
    data.work_type = data.work_type.astype(int)
    data.Residence_type[data.Residence_type == 'Rural'] = 1.0
    data.Residence_type[data.Residence_type == 'Urban'] = 2.0
    data.Residence_type = data.Residence_type.astype(int)
    data.smoking_status[data.smoking_status == 'smokes'] = 1.0
    data.smoking_status[data.smoking_status == 'formerly smoked'] = 2.0
    data.smoking_status[data.smoking_status == 'never smoked'] = 3.0
    data.smoking_status = data.smoking_status.astype(int)
    return data
def classification_report_csv(report,string):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    os.chdir(result_path)
    dataframe.to_csv('%s.csv'%string, index = False)
    os.chdir(oldfolder)
    
def method1(data3,string):
    print('In this method we fix a classifier on the raw data') 
    output=data3[:,[11]]
    input1=np.delete(data3, [0,11], axis=1)

    output=np.reshape(output,(-1,1),'F')
    print('')
    

    X_train, X_test, y_train, y_test = train_test_split(input1, output, test_size=0.1, random_state=42)
    
    model_tree = RandomForestClassifier( n_estimators=500,n_jobs=-1)
    model_tree.fit(X_train, y_train)
    pickle.dump(model_tree, open('Stroke_classifer1.asv', 'wb'))
    
    predict = model_tree.predict(X_test)
        
    print(confusion_matrix(y_test, predict))
    report=classification_report(y_test, predict)
    os.chdir(result_path)
    text_file = open("%s.txt"%string, "w")
    n = text_file.write(report)
    text_file.close()
    os.chdir(oldfolder)
#    classification_report_csv(report,'method_1')
    print(classification_report(y_test, predict))
    return report
    
def method2(data3,string):
    print('In this method we deal with the class imbalance problem with 2 classes\
          alone')
    output=data3[:,[11]]
    input1=np.delete(data3, [0,11], axis=1)

    output=np.reshape(output,(-1,1),'F')
    print('')
    
    sm = SMOTE(sampling_strategy='minority')
    inputoverall, outputoverall = sm.fit_resample(input1, output)
    print('Resampled dataset shape %s' % Counter(outputoverall))
    X_train, X_test, y_train, y_test = train_test_split(inputoverall, outputoverall, test_size=0.1, random_state=42)
    
    model_tree = RandomForestClassifier( n_estimators=500,n_jobs=-1)
    model_tree.fit(X_train, y_train)
    pickle.dump(model_tree, open('Stroke_classifer2.asv', 'wb'))
    
    X_train, X_test, y_train, y_test = train_test_split(input1, output, test_size=0.1, random_state=42)

    predict = model_tree.predict(X_test)
        
    print(confusion_matrix(y_test, predict))
    report=classification_report(y_test, predict)
    os.chdir(result_path)
    text_file = open("%s.txt"%string, "w")
    n = text_file.write(report)
    text_file.close()
    os.chdir(oldfolder)
    print(classification_report(y_test, predict))
    return report

    
def method3(data3,string):
    print('In this method we subdivide the class imbalance problem\
          by subdividing the domina class into small classes first,\
          then equilibrating the total nmber of classes and fit the classifier')
    output=data3[:,[11]]
    input1=np.delete(data3, [0,11], axis=1)
    inputraw=input1
    output=np.reshape(output,(-1,1),'F')
    print('')
    print('Standardize and normalize the input data')


    numclement = len(input1[0])
    ydamir=output
    scaler1 = MinMaxScaler()
    (scaler1.fit(ydamir))
    ydamir=(scaler1.transform(ydamir))
    ydami=numclement*10*ydamir
    input1=inputraw
    label0=(np.asarray(np.where(output == 0))).T
    a0=input1[label0[:,0],:]
    a0=np.reshape(a0,(-1,numclement),'F')
    b0=ydami[label0[:,0],:]
    b0=np.reshape(b0,(-1,1),'F')
    matrix=np.concatenate((a0,b0), axis=1)
    i=1
    kuse=getoptimumk(matrix,i)
    print('')
    print('the optimum number of clusters for machine is',kuse)

    ruuth=kuse # 
    nclusters=ruuth
    print('')
    print('Do the K-means clustering of [X,y] and get the labels')
    kmeans = MiniBatchKMeans(n_clusters=ruuth,random_state=0,batch_size=10,max_iter=20).fit(matrix)
    dd=kmeans.labels_
    dd=dd.T
    dd=np.reshape(dd,(-1,1))

    print('')
    print('Ovearall input now')
    label1=(np.asarray(np.where(output == 1))).T
    a1=input1[label1[:,0],:]
    a1=np.reshape(a1,(-1,numclement),'F')
    b1=output[label1[:,0],:]
    b1=np.reshape(b1,(-1,1),'F')
    b1=b1+nclusters-1
    print('People with stroke now are label',nclusters)
    inputoverall=np.concatenate((a0,a1), axis=0)
    outputoverall=np.concatenate((dd,b1), axis=0)
    
    
    sm = SMOTE(sampling_strategy='minority')
    inputoverall, outputoverall = sm.fit_resample(inputoverall, outputoverall)
    print('Resampled dataset shape %s' % Counter(outputoverall))
    X_train, X_test, y_train, y_test = train_test_split(inputoverall, outputoverall, test_size=0.1, random_state=42)
    
    model_tree = RandomForestClassifier( n_estimators=500,n_jobs=-1)
    model_tree.fit(X_train, y_train)
    pickle.dump(model_tree, open('Stroke_classifer3.asv', 'wb'))
    X_train, X_test, y_train, y_test = train_test_split(inputraw, output, test_size=0.1, random_state=42)

    predict = model_tree.predict(X_test)
    
    predict[predict < nclusters] = 0
    predict[predict == nclusters] = 1
        
    print(confusion_matrix(y_test, predict))
    report=classification_report(y_test, predict)
    os.chdir(result_path)
    text_file = open("%s.txt"%string, "w")
    n = text_file.write(report)
    text_file.close()
    os.chdir(oldfolder)
    print(classification_report(y_test, predict))
    return report
    np.savetxt('cluster.out',nclusters, fmt = '%d', newline = '\n')
#training_true = str(input('Enter the folder name you want to save the results: '))
training_true='Question_1'
result_path =  os.path.join(oldfolder,training_true,'results')
#if os.path.isdir(result_path): 
#    shutil.rmtree(result_path)     
os.makedirs('Question_1/results')

print('read in the data')
data1=pd.read_csv("train_2v.csv")
print(data1.groupby(['smoking_status']).count())
data1.drop(data1.columns[[0,11]], axis=1, inplace=True)
clementlist=list(data1)
print('')
print('Input columns must be arranged in this format:')
print('')
for col in data1.columns: 
    print(col)

df=pd.read_csv("train_2v.csv")
df=df.dropna()

data3=Replace_clement(df)

data3=data3.values
print('')
print(' Now Running the 3 experiments')
print('')
print('For the first experiment')
report1=method1(data3,'method_1')
print('')
print('For the second experiment')
report2=method2(data3,'method_2')
print('For the 3rd experiment')
report3=method3(data3,'method_3')