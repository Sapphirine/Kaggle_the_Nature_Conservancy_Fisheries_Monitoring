import re
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


# <=====================================================================================>
# utils
def find_label(path):
    '''define the label from path name'''
    if re.search('ALB', path):
        return(0)
    elif re.search('BET', path):
        return(1)
    elif re.search('DOL', path):
        return(2)
    elif re.search('LAG', path):
        return(3)
    elif re.search('NoF', path):
        return(4)
    elif re.search('OTHER', path):
        return(5)
    elif re.search('SHARK', path):
        return(6)
    elif re.search('YFT', path):
        return(7)


def print_clf(clf, trainx, testx, trainy, testy):
    start = time.time()
    model = clf.fit(trainx, trainy)
    end = time.time()
    pred = model.predict(testx)
    print "log_loss: ", log_loss(testy, model.predict_proba(testx))
    print confusion_matrix(np.array(testy), pred)


def binary_trans(trainy, testy, i):
    y1 = trainy['image_path'].apply(lambda x: 1 if x == int(i) else 0).tolist()
    y2 = testy['image_path'].apply(lambda x: 1 if x == int(i) else 0).tolist()
    return(y1, y2)


# <=====================================================================================>
# load the data
os.chdir('/Users/pengfeiwang/Desktop/f/fisheries/')

df = joblib.load('./data/8000_data.pkl', 'rb')
x = pd.DataFrame(df['features'].tolist())
y = df['image_path'].apply(lambda x: find_label(x)).to_frame()
trainx, testx, trainy, testy = train_test_split(
    x, y, test_size=0.5, random_state=1)

# chain 0
trainy0, target0 = binary_trans(trainy, testy, '0')
clf0 = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=1, n_estimators=1000)
print_clf(clf0, trainx, testx, trainy0, target0)  # 0.482109036119

# chain 1
trainy1, target1 = binary_trans(trainy, testy, '1')
clf1 = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=1, n_estimators=1000)
print_clf(clf1, trainx, testx, trainy1, target1)  # 0.482109036119

# chain 2
trainy2, target2 = binary_trans(trainy, testy, '2')
clf2 = GradientBoostingClassifier(max_depth=5, learning_rate=0.3, subsample=0.9, n_estimators=1000)
print_clf(clf2, trainx, testx, trainy2, target2)

# chain 3
trainy3, target3 = binary_trans(trainy, testy, '3')
clf3 = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=0.9, n_estimators=1000)
print_clf(clf3, trainx, testx, trainy3, target3)

# chain 4
trainy4, target4 = binary_trans(trainy, testy, '4')
clf4 = GradientBoostingClassifier(max_depth=5, learning_rate=0.3, subsample=0.9, n_estimators=1000)
print_clf(clf4, trainx, testx, trainy4, target4)

# chain 5
trainy5, target5 = binary_trans(trainy, testy, '5')
clf5 = GradientBoostingClassifier(max_depth=15, learning_rate=0.3, subsample=0.9, n_estimators=1000)
print_clf(clf5, trainx, testx, trainy5, target5)

# chain 6
trainy6, target6 = binary_trans(trainy, testy, '6')
clf6 = GradientBoostingClassifier(max_depth=5, learning_rate=0.3, subsample=0.9, n_estimators=1000)
print_clf(clf6, trainx, testx, trainy6, target6)

# chain 7
trainy7, target7 = binary_trans(trainy, testy, '7')
clf7 = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=0.9, n_estimators=1000)
print_clf(clf7, trainx, testx, trainy7, target7)


# tune the parameters
gbc_grid = GridSearchCV(estimator=GradientBoostingClassifier(),
                        param_grid={'max_depth': [5, 10, 15, 20],
                                    'n_estimators': [300],
                                    'learning_rate': [0.1],
                                    'subsample': [1]},
                        scoring='neg_log_loss')
gbc_grid.fit(trainx, trainy)
print gbc_grid.best_estimator_

# <=====================================================================================>
# ensemble
test_df = testy.copy()
test_df.columns = ['label']
test_df.index = range(test_df.shape[0])
clf = [clf0, clf1, clf2, clf3, clf4, clf5, clf6, clf7]

for i, j in enumerate(clf):
    result = j.predict_proba(testx)
    test_df[i] = [prob[1] for prob in result]

row_max = test_df.ix[:, 1:].apply(lambda x: x.max(), axis=1).tolist()
for i in range(test_df.shape[0]):
    test_df.ix[i, 1:] = [1 if item == row_max[i] else 0 for item in test_df.ix[i, 1:].tolist()]

x2 = test_df.ix[:, 1:]
y2 = test_df['label']
tr_x, te_x, tr_y, te_y = train_test_split(
    x2, y2, test_size=0.8, random_state=1)
clf_logis = LogisticRegression(multi_class='ovr')
clf_logis.fit(tr_x, tr_y)
log_loss(te_y, clf_logis.predict_proba(te_x))


# <=====================================================================================>
# for the test data
t = joblib.load('./data/8000_bow_test.pkl', 'rb')
index_name = t['image_path']
t = pd.DataFrame(t['features'].tolist())
d = pd.DataFrame(index=t.index)
for i, j in enumerate(clf):
    result = []
    result = j.predict_proba(t)
    d[i] = [prob[1] for prob in result]

row_max = d.apply(lambda x: x.max(), axis=1).tolist()
for i in range(t.shape[0]):
    d.ix[i,:] = [1 if item == row_max[i] else 0 for item in d.ix[i,:].tolist()]

result = clf_logis.predict_proba(d)
# result to csv
result = pd.DataFrame(result)
result.to_csv('/Users/pengfeiwang/Desktop/ttt.csv', index=index_name)



