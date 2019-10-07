import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from stackerpy import StackerModel
import matplotlib.pyplot as plt

np.random.seed(np.sum([ord(i) for i in 'StackerPy Testing']))

data = load_breast_cancer()

X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=np.sum([ord(i) for i in 'StackerPy'])
)

# ----------------------------------------
# Results df creation for storage
# ----------------------------------------

results = pd.DataFrame()

# ----------------------------------------
# Original classifier results / eval
# ----------------------------------------

random_forest = RandomForestClassifier()
random_forest.fit(X_train, np.ravel(y_train))
random_forest_predictions = random_forest.predict(X_test)

print('Original Random Forest Classifier Results')
print(
    classification_report(
        y_true=y_test,
        y_pred=random_forest_predictions
    )
)
random_forest_accuracy = accuracy_score(y_true=y_test, y_pred=random_forest_predictions)
random_forest_recall = recall_score(y_true=y_test, y_pred=random_forest_predictions)
random_forest_precision = precision_score(y_true=y_test, y_pred=random_forest_predictions)
random_forest_f1 = f1_score(y_true=y_test, y_pred=random_forest_predictions)

print('Accuracy Score: ', random_forest_accuracy)
print('Recall Score: ', random_forest_recall)
print('Precision Score: ', random_forest_precision)
print('F1 Score: ', random_forest_f1)

print('*'*100)

logistic = LogisticRegression(solver='lbfgs', max_iter=10000)
logistic.fit(X_train, np.ravel(y_train))
logistic_predictions = logistic.predict(X_test)

print('Original Logistic Regression Classifier Results')
print(
    classification_report(
        y_true=y_test,
        y_pred=logistic_predictions
    )
)
logistic_accuracy = accuracy_score(y_true=y_test, y_pred=logistic_predictions)
logistic_recall = recall_score(y_true=y_test, y_pred=logistic_predictions)
logistic_precision = precision_score(y_true=y_test, y_pred=logistic_predictions)
logistic_f1 = f1_score(y_true=y_test, y_pred=logistic_predictions)

print('Accuracy Score: ', logistic_accuracy)
print('Recall Score: ', logistic_recall)
print('Precision Score: ', logistic_precision)
print('F1 Score: ', logistic_f1)

print('*'*100)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, np.ravel(y_train))
dtree_predictions = dtree.predict(X_test)

print('Original Decision Tree Classifier Results')
print(
    classification_report(
        y_true=y_test,
        y_pred=dtree_predictions
    )
)
dtree_accuracy = accuracy_score(y_true=y_test, y_pred=dtree_predictions)
dtree_recall = recall_score(y_true=y_test, y_pred=dtree_predictions)
dtree_precision = precision_score(y_true=y_test, y_pred=dtree_predictions)
dtree_f1 = f1_score(y_true=y_test, y_pred=dtree_predictions)

print('Accuracy Score: ', dtree_accuracy)
print('Recall Score: ', dtree_recall)
print('Precision Score: ', dtree_precision)
print('F1 Score: ', dtree_f1)

print('*'*100)

ridge = RidgeClassifier()
ridge.fit(X_train, np.ravel(y_train))
ridge_predictions = ridge.predict(X_test)

print('Original Ridge Classifier Results')
print(
    classification_report(
        y_true=y_test,
        y_pred=ridge_predictions
    )
)
ridge_accuracy = accuracy_score(y_true=y_test, y_pred=ridge_predictions)
ridge_recall = recall_score(y_true=y_test, y_pred=ridge_predictions)
ridge_precision = precision_score(y_true=y_test, y_pred=ridge_predictions)
ridge_f1 = f1_score(y_true=y_test, y_pred=ridge_predictions)

print('Accuracy Score: ', ridge_accuracy)
print('Recall Score: ', ridge_recall)
print('Precision Score: ', ridge_precision)
print('F1 Score: ', ridge_f1)

print('*'*100)
# -----------------------------------------------------
# Stacker Model classifier results / eval with blending
# -----------------------------------------------------

lr2 = LogisticRegression(solver='lbfgs', max_iter=10000)
dt2 = DecisionTreeClassifier()
rf2 = RandomForestClassifier(n_estimators=25)
rc2 = RidgeClassifier()
models = [lr2, dt2, rf2]

stacker = StackerModel()
stacker.fit(
    X=X_train,
    y=y_train,
    models=models,
    stacker=rc2,
    blend=True,
    splits=5,
    model_feature_indices=None)

stacker_predictions = stacker.predict(X_test)

print('Stacker Model Results with blending')
print(
    classification_report(
        y_true=y_test,
        y_pred=stacker_predictions
    )
)
stacker_accuracy = accuracy_score(y_true=y_test, y_pred=stacker_predictions)
stacker_recall = recall_score(y_true=y_test, y_pred=stacker_predictions)
stacker_precision = precision_score(y_true=y_test, y_pred=stacker_predictions)
stacker_f1 = f1_score(y_true=y_test, y_pred=stacker_predictions)

print('Accuracy Score: ', stacker_accuracy)
print('Recall Score: ', stacker_recall)
print('Precision Score: ', stacker_precision)
print('F1 Score: ', stacker_f1)

print('*'*100)

# results['model'] = [i.__str__().split('(')[0] for i in [random_forest, logistic, dtree]] + ['StackerModel']
results['model'] = ['Random\nForest',
                    'Logistic\nRegression',
                    'Decision\nTree',
                    'Ridge\nClassifier',
                    'Stacker\nModel']

results['accuracy'] = [random_forest_accuracy,
                       logistic_accuracy,
                       dtree_accuracy,
                       ridge_accuracy,
                       stacker_accuracy]

results['recall'] = [random_forest_recall,
                     logistic_recall,
                     dtree_recall,
                     ridge_recall,
                     stacker_recall]

results['precision'] = [random_forest_precision,
                        logistic_precision,
                        dtree_precision,
                        ridge_precision,
                        stacker_precision]

results['f1'] = [random_forest_f1,
                 logistic_f1,
                 dtree_f1,
                 ridge_f1,
                 stacker_f1]

fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=250)
fig.subplots_adjust(hspace=0.3)
((ax1, ax2), (ax3, ax4)) = axes

for ax, score in zip([ax1, ax2, ax3, ax4], results.columns[1:]):
    ax.set_title(f'{score} score')
    ax.set_ylabel('Score')
    results.set_index('model')[score].sort_values().plot(kind='bar', ax=ax, fontsize=10, color='#2E73A0')
    ax.set_xlabel('Model Name')
    ax.set_ylim([0.8, 1])
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

fig.suptitle('Model Scoring', fontsize=20)
plt.savefig('./Model Scoring Results.png')
# plt.show()
