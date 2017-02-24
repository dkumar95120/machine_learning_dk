import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# Load the Census dataset
data = pd.read_csv("census.csv")
# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = 100*n_greater_50k/n_records

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.replace(['<=50K','>50K'],[0,1])

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# TODO: Calculate accuracy
# since there are 24% of people with income more than $50,000, the accuracy would be .24
accuracy = float(n_greater_50k)/n_records

# TODO: Calculate F-score using the formula above for beta = 0.5
tp = n_greater_50k # at most n_greater_50k people can be correctly classified as earning >50k
fp = n_at_most_50k # at most n_at_most_50k people can be misclassified as earning >50k
fn = 0             # since model always predicts person making more than $50k
precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
print 'precision, recall:', precision,',',recall
beta = 0.5
fscore = (1+beta*beta)*(precision*recall)/(beta*beta*precision + recall)

from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit (X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict (X_test)
    predictions_train = learner.predict (X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score (y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score (y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(max_depth=3)
#clf_B = SVC (kernel='linear', gamma=10, C=10)
clf_B = GaussianNB()
clf_C = AdaBoostClassifier(clf_A, random_state=0)
#clf_C = RandomForestClassifier(n_estimators=10)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = n_records/100
samples_10 = n_records/10
samples_100 = n_records
print 'samples:[', samples_1, samples_10, samples_100, ']'
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)
        print 'Best fscore for', clf_name, results[clf_name][i]['f_test']
        print 'Training Time:', results[clf_name][i]['train_time']
        print 'Prediction Time:', results[clf_name][i]['pred_time']

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(y_train, 10, test_size=0.2, random_state=42)
# TODO: Initialize the classifier
clf = AdaBoostClassifier(DecisionTreeClassifier(), random_state=0)

# TODO: Create the parameters list you wish to tune
#parameters = {'max_depth':[2,4,5,6,10], 
#             'max_features':[None,'auto','log2'], 
#              'min_samples_split':[4, 20, 50]}
parameters = {'base_estimator__max_depth': [3,4],
		'base_estimator__min_samples_split':[4, 50],
		'n_estimators':[10,20]}
# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=.5)
print "starting Grid Search:"
start = time()
# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, cv=cv)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(np.array(X_train), np.array(y_train))
end = time ()
print "time taken for grid search:", end - start
# Get the estimator
best_clf = grid_fit.best_estimator_

print "best parameters:", grid_fit.best_params_
# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

# TODO: Train the supervised model on the training set 
clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = clf.feature_importances_

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
start = time()
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
end = time()
print "Training time:", end-start
# Make new predictions
start = time()
reduced_predictions = clf.predict(X_test_reduced)
end = time()
print "Prediction time:", end-start
# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))
