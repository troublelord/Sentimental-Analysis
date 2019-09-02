import string
import itertools

import nltk
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
'''
def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])

if __name__ == '__main__':

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    data_dir = sys.argv[1]
    classes = ['pos', 'neg']
'''
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def isNormal(s):
    try:
        s.decode('utf8').encode('ascii', errors='ignore')
    except UnicodeEncodeError:
        return False
    else:
        return True
def preprocess(X,stop_words):
    X_new=X
    for iter in range(len(X_new)):
        s=[]
        for w in X_new[iter]:
            if w in string.punctuation:
                s.append(' ')
            elif w not in string.printable:
                s.append('')
            elif not isEnglish(w):
                s.append('')
            else:
                s.append(w)
        X_new[iter]=''.join(s)
    for iter in range(len(X_new)):
        s = []
        for word_iter in X_new[iter].split():
            if word_iter not in stop_words:
                s.append(word_iter)
        X_new[iter] = ' '.join(s)    
        X_new[iter] = ''.join(i for i, _ in itertools.groupby(X_new[iter])).lower()
    return X_new

import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


filename = r'C:\Users\user\Downloads\Classifier\codemixed.txt'
with open(filename,'r',encoding='utf8') as f:
    x = f.readlines()
    
#preparing the dataset from the read content
''' 
    X_temp is the comment
    Y is the label
'''

X_temp=[[] for i in range(len(x))]

Y=[0 for i in range(len(x))]
for i in range(1,len(x)-1):
    
    X_temp[i]=x[i].rstrip().lstrip().split(",")[2]
    Y[i]=int(x[i].rstrip().split(",")[1])

preprocess(X_temp,stop_words)



# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
X_temp = vectorizer.fit_transform(X_temp)
x_train, x_test, y_train, y_test = train_test_split(X_temp, Y, test_size=0.25, random_state=42)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(x_train, y_train)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(x_test)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(x_train, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(x_test)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(x_train, y_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(x_test)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(y_test, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(y_test, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(y_test, prediction_liblinear))
