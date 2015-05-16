#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
'salary',
'total_payments',
'exercised_stock_options',
'restricted_stock',
'restricted_stock_deferred',
'total_stock_value',
'expenses',
'from_messages',
'other',
'from_this_person_to_poi',
'deferred_income',
'from_poi_to_this_person']



print features_list

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )





### Task 2: Remove outliers

#remove observations that do not belong
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

#Fix incorrect data - see enron6102insiderpay.pdf 

data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093

data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290


### Task 3: Create new feature(s)


### Store to my_dataset for easy export below.
my_dataset = data_dict

###create new features

#create features for calculating the % of messages that are to or from a POI
for name in my_dataset:
    if my_dataset[name]['to_messages'] == 'NaN' or my_dataset[name]['from_this_person_to_poi'] == 'NaN':
        my_dataset[name]['ratio_to_poi'] = 'NaN'
        continue
    to_total = float(my_dataset[name]['to_messages'])
    to_poi = float(my_dataset[name]['from_this_person_to_poi'])
    my_dataset[name]['ratio_to_poi'] = to_poi/to_total
       
for name in my_dataset:
    if my_dataset[name]['from_messages'] == 'NaN' or my_dataset[name]['from_poi_to_this_person'] == 'NaN':
        my_dataset[name]['ratio_from_poi'] = 'NaN'
        continue
    from_total = float(my_dataset[name]['from_messages'])
    from_poi = float(my_dataset[name]['from_poi_to_this_person'])
    my_dataset[name]['ratio_from_poi'] = from_poi/from_total


#for each of the payment categories, calculate the % of total_payments for that category, and create a new variable 

payments = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses',
            'director_fees']
for name in my_dataset:
    if my_dataset[name]['total_payments'] == 'NaN':
        for category in payments:
            new_var = category + '_ratio'
            my_dataset[name][new_var] = 'NaN'
        continue
    for category in payments:
        if my_dataset[name][category] == 'NaN':
            my_dataset[name][category] = 0
        new_var = category + '_ratio'
        my_dataset[name][new_var] = float(my_dataset[name][category]) / my_dataset[name]['total_payments']



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2), n_estimators=30, learning_rate = .8)



########Classifier 'graveyard'

# clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

# base = DecisionTreeClassifier()

# parameters = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70], 'learning_rate' : [x * .1 for x in range(1,11)]}

# clf = GridSearchCV(AdaBoostClassifier(), parameters)



# clf = DecisionTreeClassifier(min_samples_leaf=1)

# clf = GradientBoostingClassifier(max_depth=2, min_samples_leaf=2, max_features=3)


# param_grid = {'C' : [1, 10], 'gamma' : [1, .1]}

# pipeline = [('scaler', MinMaxScaler()), ('pca', RandomizedPCA(n_components=5, whiten=True)), ('SVM', SVC(C = 10, gamma = .1))]
# clf = Pipeline(pipeline)
######################



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)