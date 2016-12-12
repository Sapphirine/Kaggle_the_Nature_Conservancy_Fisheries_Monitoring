import re, time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV


# <=====================================================================================>
# utils
def find_label(path):
	'''define the label from path name'''
	if re.search('ALB',path): return(0)
	elif re.search('BET',path): return(1)
	elif re.search('DOL',path): return(2)
	elif re.search('LAG',path): return(3)
	elif re.search('NoF',path): return(4)
	elif re.search('OTHER',path): return(5)
	elif re.search('SHARK',path): return(6)
	elif re.search('YFT',path): return(7)

def print_clf(clf, trainx, testx, trainy, testy):
	start = time.time()
	model = clf.fit(trainx, trainy)
	end = time.time()
	pred = model.predict(testx)
	print "log_loss: ", log_loss(testy, model.predict_proba(testx))
	print confusion_matrix(np.array(testy), pred)

def binary_trans(trainy, testy, i):
	y1 = trainy['image_path'].apply(lambda x: 1 if x==int(i) else 0).tolist()
	y2 = testy['image_path'].apply(lambda x: 1 if x==int(i) else 0).tolist()
	return(y1, y2)

# <=====================================================================================>
# load the data
df = joblib.load('/Users/pengfeiwang/Desktop/f/fisheries/data/ .pkl','rb')

x = pd.DataFrame(df['features'].tolist())
y = df['image_path'].apply(lambda x: find_label(x)).to_frame()
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.25, random_state=1)

# chain 0
trainy0, target0 = binary_trans(trainy, testy, '0')
clf = GradientBoostingClassifier(max_depth=10, learning_rate=0.2, subsample=0.8, n_estimators=800)
print_clf(clf, trainx, testx, trainy0, target0) # 0.482109036119

# chain 1
trainy1, target1 = binary_trans(trainy, testy, '1')
clf = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=1, n_estimators=800)
print_clf(clf, trainx, testx, trainy1, target1) # 0.482109036119

# chain 2
trainy2, target2 = binary_trans(trainy, testy, '2')
clf = GradientBoostingClassifier(max_depth=10, learning_rate=0.1, subsample=0.9, n_estimators=800)
print_clf(clf, trainx, testx, trainy2, target2)

# chain 3
trainy3, target3 = binary_trans(trainy, testy, '3')
clf = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=0.9, n_estimators=800)
print_clf(clf, trainx, testx, trainy3, target3)

# chain 4
trainy4, target4 = binary_trans(trainy, testy, '4')
clf = GradientBoostingClassifier(max_depth=5, learning_rate=0.1, subsample=0.9, n_estimators=800)
print_clf(clf, trainx, testx, trainy4, target4)

gbc_grid = GridSearchCV(estimator = GradientBoostingClassifier(),
                        param_grid = {'max_depth': [10], 'n_estimators': [300], 'learning_rate': [0.1,0.2,0.3],
                        'subsample':[1]},
                        scoring = 'neg_log_loss')

gbc_grid.fit(trainx, trainy4['image_path'])
print gbc_grid.best_estimator_
print gbc_grid.best_score_
