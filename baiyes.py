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
for i in range(1,len(x)-1):#len(x)-1):
    #print(x[i].rstrip().lstrip().split(",")[2].strip())
    #print(int(x[i].rstrip().split(",")[1]))
    
    X_temp[i]=x[i].rstrip().lstrip().split(",")[2]
    Y[i]=int(x[i].rstrip().split(",")[1])

preprocess(X_temp,stop_words)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

x_train, x_test, y_train, y_test = train_test_split(X_temp, Y, test_size=0.25, random_state=0)

from sklearn.metrics import classification_report
import numpy as np

score = 'f1_macro'
print("# Tuning hyper-parameters for %s" % score)
print()
np.errstate(divide='ignore')
clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
clf.fit(x_train, y_train)

'''
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                             clf.cv_results_['std_test_score'], 
                             clf.cv_results_['params']):
w    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()

'''
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
Y_pred =clf.predict(x_test)
print(classification_report(y_test,Y_pred , digits=4))
print()
#print("Confusion matrix:\n%s" % confusion_matrix(y_test, y))

