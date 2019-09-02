
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
for i in range(1,len(x)-1):    
    X_temp[i]=x[i].rstrip().lstrip().split(",")[2]
    Y[i]=int(x[i].rstrip().split(",")[1])




preprocess(X_temp,stop_words)


from sklearn.ensemble import RandomForestClassifier

#from sklearn.datasets import make_classification

#X, y = make_classification(n_samples=1000, n_features=4,
#...                            n_informative=2, n_redundant=0,
#...                            random_state=0, shuffle=False)
clf  = RandomForestClassifier(n_estimators=100, random_state=0)
'''                            
clf  = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features=5, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)

X_temp = vectorizer.fit_transform(X_temp)

x_train, x_test, y_train, y_test = train_test_split(X_temp, Y, test_size=0.25, random_state=42)


#x_train = vectorizer.fit_transform(x_train)

#test_vectors = vectorizer.transform(test_data)





clf.fit(x_train,y_train)

print(clf.feature_importances_)

#print(clf.predict([[0, 0, 0, 0]]))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
Y_pred=clf.predict(x_test)

print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
print(classification_report(y_test, Y_pred, digits=4))
print()
print("Confusion matrix:\n%s" % confusion_matrix(y_test, Y_pred))

