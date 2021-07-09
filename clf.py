# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.decomposition import PCA
import numpy as np
import re
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier



def load_data(input_file_name_in):
    input_file = open(input_file_name_in, 'r', encoding='utf-8')
    id_list_out = []
    data_list_out = []
    label_list_out = []
    lines_1 = input_file.read().split('\n')
    count = 0
    last = ''
    print('loading:')
    for i in lines_1:
        count += 1
        print('\r', count/len(lines_1)*100, '%', end='')
        if '<review id=' in i:
            i = i.replace('1.0', '1')
            i = i.replace('0.0', '0')
            isp = i.split(' ')
            for sp in isp:
                if 'id' in sp:
                    id_ind = sp.index("=")
                    new_id = sp[id_ind + 2:-1]
                    id_in = int(new_id)
                    id_list_out.append(id_in)
                elif 'label' in sp:
                    lb_ind = sp.index("=")
                    new_lb = sp[lb_ind + 2:-2]
                    label_in = int(new_lb)
                    label_list_out.append(label_in)
        else:
            if '</review>' in i:
                data_list_i = []
                new_data_str = last.split(' ')
                for data in new_data_str:
                    if re.search(r'\d', data):
                        if '[' in data or ']' in data:
                            data = data.replace('[', '')
                            data = data.replace(']', '')
                        data_in = float(data)
                        data_list_i.append(data_in)
                data_list_out.append(data_list_i)
                last = ''
            else:
                last += i
    id_list_output = np.array(id_list_out[0:len(id_list_out)])
    label_list_output = np.array(label_list_out[0:len(label_list_out)])
    data_list_output = np.array(data_list_out[0:len(data_list_out)])
    print('\n')
    print('loading successfully!\n')
    return id_list_output, label_list_output, data_list_output


def normalization(data):
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    input_file_name_sample = './sample_0.98_d2v_40000.txt'
    id_list_sample, label_list_sample, data_list_sample = load_data(input_file_name_sample)
    input_file_name_test = './test_0.98_d2v.txt'
    id_list_test, label_list_test, data_list_test = load_data(input_file_name_test)
    x_train = standardization(data_list_sample)
    y_train = label_list_sample
    x_test = standardization(data_list_test)
    y_test = label_list_test
    clf1 = BaggingClassifier(KNeighborsClassifier(),n_estimators=20, max_samples=0.5, max_features=0.5)
    clf1.fit(x_train, y_train)
    y_true, y_pred = y_test, clf1.predict(x_test)
    print(classification_report(y_true, y_pred))


    # Set the parameters by cross-validation




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""

    SVC(kernel='linear')
    BaggingClassifier(KNeighborsClassifier(),n_estimators=20, max_samples=0.5, max_features=0.5)
    RandomForestClassifier(n_estimators=10)
    AdaBoostClassifier(n_estimators=10)
    
    base_clf = SVC(kernel='linear', probability=True)
    clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=5)
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
    print(scores)

    clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = SVC(kernel='linear', probability=True)
clf3 = GaussianNB()
clf4 = DecisionTreeClassifier(criterion="entropy")
lr = LogisticRegression( solver='lbfgs')
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],  meta_classifier=lr)
for clf, label in zip([clf1, clf2, clf3, clf4, sclf], ['KNN', 'SVC', 'Naive Bayes','Decision Tree','StackingClassifier']):
       scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
       print("AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
    neighbors.KNeighborsClassifier()
    GaussianNB()
    Perceptron(max_iter=40, eta0=0.01, random_state=1)
    tree.DecisionTreeClassifier(criterion="entropy")
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    SVC(kernel='linear')

    clf1 = SVC(kernel='linear', C=10)
    clf1.fit(X_train, y_train)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5)
        
    y_pred = knn.fit(X_train, y_train).predict(X_test)
    y_true, y_pred = y_test, knn.predict(X_test)
    print(classification_report(y_true, y_pred))
    
    clf1 = SVC(kernel='linear', C=10)
    clf1.fit(x_train, y_train)
    y_true, y_pred = y_test, clf1.predict(x_test)
    print(classification_report(y_true, y_pred))
    
    clf = Perceptron(max_iter=40, eta0=0.01, random_state=1)
    
     clf = tree.DecisionTreeClassifier(criterion="entropy")
    scores = cross_val_score(clf, x, y, cv=10)
    
    clf = neighbors.KNeighborsClassifier()
    
    
    clf1.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc(criterion="entropy")
    print('clf success').fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train_std, y_train)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    
    
    input_file_name_test = './output_0.9_test.txt'
    id_list_test, label_list_test, data_list_test = load_data(input_file_name_test)
    x_test = normalization(data_list_test)
    y_test = label_list_test"""