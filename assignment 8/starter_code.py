import pandas as pd
import sqlite3

# Importing necessary libraries from sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# importing soccer data
conn = sqlite3.connect("database.sqlite")

# Reading Player_Attributes table to dataframe

player_attr_df = pd.read_sql("SELECT strength, stamina, jumping FROM Player_Attributes", conn)

# Filling with 11 for all null values
player_attr_df.fillna(11, inplace=True)

x = player_attr_df[['strength', 'stamina']].values
y = player_attr_df[['jumping']].values
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.3, random_state=0)

#####################################################  Start: DecisionTreeClassifier #####################################################
# Applying grid search on DecisionTreeClassifier
desicion_tree_params_grid = {'criterion':['gini','entropy'], 'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50], 'splitter':["best", "random"], 'random_state':[0,1,2,4,6,8,10,12,14,16,20,40,42]}
grid_search_decision_tree_classifier = GridSearchCV(DecisionTreeClassifier(), desicion_tree_params_grid, cv=10)
grid_search_decision_tree_classifier.fit(X_train, y_train)
print("Decision Tree best grid score: " + str(grid_search_decision_tree_classifier.best_score_))
print("Decision Tree grid test score: " + str(grid_search_decision_tree_classifier.score(X_test, y_test)))
decision_tree_best_params = grid_search_decision_tree_classifier.best_params_
print("Decision Tree best params: " + str(decision_tree_best_params))

# Predicting the Y_test values using the model from gridsearch
y_pred = grid_search_decision_tree_classifier.predict(X_test)

# Generating the classification_report
grid_search_decision_tree_classification_reprt = classification_report(y_test, y_pred)
print("Decision Tree Classification report with whole data")
print(grid_search_decision_tree_classification_reprt)


# Selecting features using RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)

# Applying DecisionTreeClassifier using the best params from the grid search and with selected data
decision_tree_classifier = DecisionTreeClassifier(criterion = decision_tree_best_params['criterion'],\
    max_depth = decision_tree_best_params['max_depth'],\
    random_state = decision_tree_best_params['random_state'],\
    splitter = decision_tree_best_params['splitter'])

decision_tree_classifier.fit(X_train_selected, y_train)
y_pred = decision_tree_classifier.predict(X_test_selected)
decision_tree_classification_reprt = classification_report(y_test, y_pred)
print("Decision Tree Classification report with selected data")
print(decision_tree_classification_reprt)


#####################################################  Start: LogisticRegression #####################################################

# Todo: Similarly apply for LogisticRegression








#####################################################  Start: SVC #####################################################################

# Todo: Similarly apply for SVC








#####################################################  Start: KNeighborsClassifier #####################################################
# Todo: Similarly apply for KNeighborsClassifier








#Todo: Based on all the above classification_reports conclude which technique and it's best params is the best fit for this data
