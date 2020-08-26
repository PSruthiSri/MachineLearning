# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""
#initial imports and data setup
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
data=pd.read_csv('data.csv')
x=data[data.columns[0:-1]]
y=data[data.columns[-1]]
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
min_max_scaling=MinMaxScaler()
X_train=min_max_scaling.fit_transform(X_train)
X_test=min_max_scaling.transform(X_test)
Y_train=Y_train.to_numpy()
Y_test=Y_test.to_numpy()

#accuracy function
def accuracy_fn(y_true,y_pred):
    accu=0
    cm=confusionMatrix(y_true, y_pred)
    correct_pred=np.trace(cm)
    total_pred=np.sum(cm)
    accu=(correct_pred/total_pred)*100
    return accu

#recall function
def recall_fn(y_true,y_pred):
    rec=0
    cm = confusionMatrix(y_true, y_pred)
    rec = (np.diag(cm))/ (np.sum(cm, axis = 1))
    rec=(np.sum(rec)/len(np.unique(y_true)))*100
    return rec

#precision function
def precision_fn(y_true,y_pred):
    prec=0
    cm = confusionMatrix(y_true, y_pred)
    prec = (np.diag(cm))/ (np.sum(cm, axis = 0))
    prec=(np.sum(prec)/len(np.unique(y_true)))*100
    return prec

#Within Cluster Sum of Squares implementation
def WCSS(Clusters):
    wcss=0
    diff= []
    centroid_initial_tuple = [(np.mean(i, axis=0), i) for i in Clusters]
    for m in centroid_initial_tuple:
        diff.append(np.sum((m[0] - m[1]) ** 2))
    wcss = sum(diff)
    return wcss

#confusion matrix function
def confusionMatrix(y_true,y_pred):
    y_actual = np.array(y_true).reshape(-1,1)
    y_predicted = np.array(y_pred).reshape(-1,1)
    n=len(np.unique(y_true))
    confusionMatrix = np.zeros(shape=(n,n),dtype=int)
    for i in range(len(y_actual)):
        a=y_actual[i][0]
        b=y_predicted[i][0]
        confusionMatrix[a-1][b-1] += 1
    return confusionMatrix 

#KNearest Neighbour function implementation
def KNN(X_train,X_test,Y_train,N):
    distMat=np.empty((0,X_train.shape[0]))
    print(distMat)
    y_final=[]
    k=N
    for m in range(X_test.shape[0]):
        #Euclidean distance is used
        a=np.sqrt(np.sum(np.square(np.subtract(X_test[m],X_train)),axis=1)).reshape(1,X_train.shape[0])
        b=np.argsort(a)
        y_pred=np.empty((0,1))
        for i in range(k):
            j=b[0][i]
            y_pred=np.vstack((y_pred,Y_train[j]))
        y_pred=y_pred.flatten().astype(int)
        counts=np.bincount(y_pred).argmax()
        y_final=np.append(y_final,counts)
    y_final=y_final.astype(int)
    return y_final

#Random forest function is pasted in the end
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
#PCA implementation
def PCA(X_train,N):
    X_train_T=np.transpose(X_train)
    X_covariance_matrix=np.cov(X_train_T)
    eigen_values,eigen_vectors=np.linalg.eig(X_covariance_matrix)    
    eigen_list = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    eigen_list.sort()
    eigen_list.reverse()
    eig_vec_mat=np.empty((X_train.shape[1],0))
    for i in range(N):
        eig_vec_mat = np.hstack((eig_vec_mat,eigen_list[i][1].reshape(X_train.shape[1],1)))
    reduced_dim=np.dot(X_train,eig_vec_mat)
    return reduced_dim
    
#K-means clustering implementation
def Kmeans(X_train,N):
    X_train = pd.DataFrame(data=X_train)
    centroid_initial = X_train.sample(N)
    X_train = X_train.to_numpy()
    centroid_initial = centroid_initial.to_numpy()
    centroid_old=[]
    for iterations in range(100):
        dict={}
        for i in range(X_train.shape[0]):
            u= X_train[i]-centroid_initial
            sqDiffMat = np.square(u)
            sqDistances = (sqDiffMat.sum(axis=1))**0.5
            minDistanceIndex = np.argmin(sqDistances)
            minDistance = np.min(sqDistances)
            if minDistanceIndex in dict:
                dict[minDistanceIndex].append(X_train[i])
            else:
                dict[minDistanceIndex] = [X_train[i]]
        centroid_initial_new= [np.mean(i,axis=0) for i in dict.values()]
        diff = np.linalg.norm( np.array(centroid_initial_new) - np.array(centroid_initial))
        if(diff < 0.15):
            break;
        centroid_initial = centroid_initial_new
    Clusters = [np.array(xi) for xi in dict.values()]
    return Clusters

#KNN using Sklearn
def SklearnSupervisedLearning_KNN(X_train,Y_train,X_test):
    knn = KNeighborsClassifier(n_neighbors=90)
    knn.fit(X_train, Y_train)
    knn_pred = knn.predict(X_test)
    return knn_pred

#Decision Tree using Sklearn
def SklearnSupervisedLearning_DTC(X_train,Y_train,X_test):
    #DecisionTree
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(X_train,Y_train)
    dtc_pred = dtc.predict(X_test)
    return dtc_pred

#Logistic Regression using Sklearn
def SklearnSupervisedLearning_Logreg(X_train,Y_train,X_test):
    #LogisticRegression
    logreg = LogisticRegression(max_iter=2000,multi_class='auto',solver='lbfgs')
    logreg.fit(X_train, Y_train)
    logreg_pred = logreg.predict(X_test)
    return logreg_pred

#SVM using Sklearn
def SklearnSupervisedLearning_SVM(X_train,Y_train,X_test):   
    svclassifier = svm.SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    svm_pred = svclassifier.predict(X_test)
    return svm_pred

#Voting Classifier ensemble model
def SklearnVotingClassifier(X_train,Y_train,X_test):
    kfold = model_selection.KFold(n_splits=10, random_state=None)
# create the sub models
    estimators = []
    model1 = LogisticRegression(max_iter=2000,multi_class='auto',solver='lbfgs')
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = svm.SVC(kernel = 'linear')
    estimators.append(('svm', model3))
    model4=KNeighborsClassifier(n_neighbors=90)
    estimators.append(('knn',model4))
    ensemble = VotingClassifier(estimators)
    ensemble=ensemble.fit(X_train,Y_train)
    voting_predictions=ensemble.predict(X_test)
    return voting_predictions

#Reporting of accuracy for all Sklearn supervised learning methods and Sklearn Voting Classifier
def accuracy_report(Y_test):
    knn_pred=SklearnSupervisedLearning_KNN(X_train,Y_train,X_test)
    accuracy=accuracy_fn(Y_test,knn_pred)
    print('KNN accuracy- %.2f' %accuracy)
    
    dtc_pred=SklearnSupervisedLearning_DTC(X_train,Y_train,X_test)
    accuracy=accuracy_fn(Y_test, dtc_pred)
    print('Decision tree accuracy- %.2f' %accuracy)
    
    logreg_pred=SklearnSupervisedLearning_Logreg(X_train,Y_train,X_test)
    accuracy=accuracy_fn(Y_test,logreg_pred)
    print('Logistic Regression accuracy- %.2f' %accuracy)
    
    svm_pred=SklearnSupervisedLearning_SVM(X_train,Y_train,X_test)
    accuracy=accuracy_fn(Y_test, svm_pred)
    print('SVM accuracy- %.2f' %accuracy)
    
    voting_predictions=SklearnVotingClassifier(X_train,Y_train,X_test)
    accuracy=accuracy_fn(Y_test,voting_predictions)
    print('Sklearn Voting Classifier accuracy- %.2f' %accuracy)
    
#Confusion Matrix for all Sklearn supervised learning methods and Sklearn Voting Classifier
def confusionMatrix_plot():
    #printing of confusion matrix
    knn_pred=SklearnSupervisedLearning_KNN(X_train,Y_train,X_test)
    cm_knn = confusion_matrix(Y_test,knn_pred) 
    print("Confusion Matrix for KNN:")
    print(cm_knn)
    print("")
    
    dtc_pred=SklearnSupervisedLearning_DTC(X_train,Y_train,X_test)
    cm_dtc = confusion_matrix(Y_test,dtc_pred) 
    print("Confusion Matrix for Decision Tree:")
    print(cm_dtc)
    print("")
   
    
    logreg_pred=SklearnSupervisedLearning_Logreg(X_train,Y_train,X_test)
    cm_logreg = confusion_matrix(Y_test,logreg_pred) 
    print("Confusion Matrix for Logistic Regression:")
    print(cm_logreg)
    print("")
    
    
    svm_pred=SklearnSupervisedLearning_SVM(X_train,Y_train,X_test)
    cm_svm = confusion_matrix(Y_test,svm_pred) 
    print("Confusion Matrix for SVM:")
    print(cm_svm)
    print("")
    
    
    voting_predictions=SklearnVotingClassifier(X_train,Y_train,X_test)
    cm_voting = confusion_matrix(Y_test,voting_predictions) 
    print("Confusion Matrix for Sklearn Voting Classifier:")
    print(cm_voting)
    print("")
   
    
    #plotting of confusion matrix
    plt.subplot(231)
    plt.imshow(cm_knn)
    plt.title('KNN Confustion Matrix')
    plt.xlabel('y predicted from KNN')
    plt.ylabel('Actual y')

    subplot2 = plt.subplot(232)
    subplot2.imshow(cm_dtc)
    plt.title('Decision Tree Confustion Matrix')
    plt.xlabel('y predicted from Decision Tree')
    plt.ylabel('Actual y')

    plt.subplot(233)
    plt.imshow(cm_logreg)
    plt.title('Logistic Regression Confustion Matrix')
    plt.xlabel('y predicted from Logistic Regression')
    plt.ylabel('Actual y')

    subplot2 = plt.subplot(234)
    subplot2.imshow(cm_svm)
    plt.title('SVM Confustion Matrix')
    plt.xlabel('y predicted from SVM')
    plt.ylabel('Actual y')

    subplot2 = plt.subplot(235)
    subplot2.imshow(cm_voting)
    plt.title('Voting Classifier Confustion Matrix')
    plt.xlabel('y predicted from Voting Classifier')
    plt.ylabel('Actual y')
    plt.subplots_adjust(top=1.9, bottom=0.08, left=0.10, right=1.95, hspace=0.35,wspace=0.35)
    
#Grid search on hyper parameters(neighbours) of KNN
def knn_hyper_param():
    k_values=range(5,50,5)
    k_accuracies=[]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train) 
        knn_predictions = knn.predict(X_test)
        accuracy = accuracy_fn(Y_test, knn_predictions)
        k_accuracies.append(accuracy)
    plt.plot(k_values, k_accuracies)
    plt.title('Grid Search on Hyper paramaters of KNN')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Accuracy')
    
#Grid search on hyper parameters(maximum depth) of Decision Tree
def dtc_hyper_param():
    maxdepth_values = range(4,16,1)
    decision_tree_accuracies=[]
    for d in maxdepth_values:
        dt = DecisionTreeClassifier(max_depth=d).fit(X_train,Y_train)
        dt_predictions= dt.predict(X_test)
        accuracy=accuracy_fn(Y_test, dt_predictions)
        decision_tree_accuracies.append(accuracy)
    plt.plot(maxdepth_values, decision_tree_accuracies)
    plt.title('Grid Search on Hyper paramaters of Decision Tree')
    plt.xlabel('Values of max depth for Decision Tree')
    plt.ylabel('Accuracy')

#Grid search on hyper parameters(C) of SVM
def svm_hyper_param():
    c_values=range(1,25,5)
    svm_accuracies=[]
    for c in c_values:
        svm_model_linear = svm.SVC(kernel = 'linear', C=c).fit(X_train, Y_train) 
        svm_predictions = svm_model_linear.predict(X_test)
        accuracy=accuracy_fn(Y_test, svm_predictions)
        svm_accuracies.append(accuracy)
    plt.plot(c_values, svm_accuracies)
    plt.title('Grid Search on Hyper paramaters of SVM')
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Accuracy')
    

#Random Forest implementation 

#splitting data freshly here, not considering Y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
data=pd.read_csv('data.csv')
x_train_rf,x_test_rf=train_test_split(data,test_size=0.2)
    
#random forest function implementation where it creates n decision trees and returns voting for test data
def random_forest(x_train_rf,x_test_rf,n_trees,sample_size):
    trees=list()
    for i in range(n_trees):
        x_train_bs=bootstrapping(x_train_rf,sample_size)
        no_features=6
        x_train_reduced=bagging(x_train_bs,no_features)
        #min_leaf=30
        tree=decision_tree_fn(x_train_reduced)
        trees.append(tree)
    #for i in range(x_test_rf):
        #test_data_votes=vote_test_data(trees,i)(need to write function to get the classification of test instance from each decision tree) )
    #return test_data_votes
    
#bootstrapping training data into sample size
def bootstrapping(x_train_rf,sample_size):
        x_train_bs=x_train_rf.sample(sample_size,replace=True)
        return x_train_bs

#bagging: reducing the number of features in bootstrapped data
def bagging(x_train_bs,no_features):
    column_indices=random.sample(range(0,48),no_features)
    i=0
    for i in column_indices:
        x_train_reduced=x_train_bs.iloc[:,column_indices]
        x_train_reduced = pd.concat([x_train_reduced,x_train_bs.iloc[:,-1]],axis=1)
    return x_train_reduced

#decision tree function
def decision_tree_fn(x_train_reduced):
    mc=x_train_reduced.mean(axis=0)
    parent_node,mean=data_split(x_train_reduced,mc)
    left,right=split_data(x_train_reduced,parent_node,mean)
    question= "{} <= {}".format(parent_node,mean)
    sub_tree={question:[]}

    yes_answer=decision_tree(left,parent_node,mean)

    sub_tree[question].append(yes_answer)

    sub_tree[question].append(yes_answer)

    no_answer=decision_tree(right,parent_node,mean)


    sub_tree[question].append(no_answer)

    print(sub_tree)

#function to create decision tree till it reaches min_leaf
def decision_tree(data,parent_node,local_mean):
    if(len(data))<=30:
        return (classify_data(data))
    else:
        data_mean=data.drop(parent_node,axis=1)
        mc=data_mean.mean(axis=0)
        parent_node1,mean1=data_split(data_mean,mc)
        question1= "{} <= {}".format(parent_node1,mean1)
        sub_tree1={question1:[]}
        left,right=split_data(data_mean,parent_node1,mean1)
        subt=decision_tree(left,parent_node1,mean1)
        sub_tree1[question1].append(subt)
        subtR=decision_tree(right,parent_node1,mean1)
        sub_tree1[question1].append(subtR)
        return sub_tree1
    
#function to label leaf nodes
def classify_data(data):
    label_column=data[:,-1]
    unique_classes,counts_unique_classes=np.unique(label_column,return_counts=True)
    index=counts_unique_classes.argmax()
    classification=unique_classes[index]
    return classification

#function to return left and right nodes 
def split_data(x_train_rf,split_column,split_value):
    split_column_values=x_train_rf.loc[:,split_column]
    left=x_train_rf[split_column_values<=split_value]
    right=x_train_rf[split_column_values>split_value]
    return left,right

#function to calculate Gini Index 
def gini_index(subTree):
    li=[]
    for l in range(len(subTree)):
        li.append(subTree[l][0])
    li=np.array(li)
    li1=np.array(np.unique(li, return_counts=True)).T
    gini=0
    for gin_ind in range(len(li1)):
        gini_temp = (li1[gin_ind][1]/len(left))**2
        gini= gini+gini_temp
    return 1-gini

#function which calculates weighted gini of each split and selects the parent node
def data_split(rf,mc):
    weighted_gini=[]
    for mean1 in range(len(rf.columns)-1):
        col1=rf.iloc[:,mean1]
        col1=col1.to_numpy()
        mc_temp=mc[mean1]
        left=[]
        right=[]
        for i in range(len(rf.values)):
            if col1[i]>mc_temp:left.append((rf.iloc[i][-1],col1[i]))
            else:right.append((rf.iloc[i][-1],col1[i]))
        gini_left = gini_index(left)
        gini_right = gini_index(right)
        wg = ((len(left)/len(rf.values))*gini_left) + ((len(right)/len(rf.values))*gini_right)
        weighted_gini.append((wg,rf.columns[mean1]))
    return(sorted(weighted_gini)[0][1]),mc[sorted(weighted_gini)[0][1]]