print('-'*200)
print('-'*200)
print('-'*200)
print('Hello, World!')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from stackerpy import  StackerPyClassifier


data = load_breast_cancer()

X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.sum([ord(i) for i in 'StackerPy']))


# ----------------------------------------
# Original ridge classifier results / eval
# ----------------------------------------

ridge = RandomForestClassifier()
ridge.fit(X_train, np.ravel(y_train))
ridge_predictions = ridge.predict(X_test)

print('Original Ridge Classifier Results')
print(
    classification_report(
        y_true=y_test,
        y_pred=ridge_predictions
    )
)

print('Accuracy Score: ', accuracy_score(
    y_true=y_test,
    y_pred=ridge_predictions
))

print('*'*100)
# ----------------------------------------------------
# Stacker Model classifier results / eval w/o blending
# ----------------------------------------------------

lr = LogisticRegression(solver='lbfgs', max_iter=10000)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=25)
rc = RidgeClassifier()
models = [lr, dt, rf]

stacker = StackerPyClassifier()
stacker.fit(
    X=X_train,
    y=y_train,
    models=models,
    stacker=rc,
    blend=False,
    splits=5,
    model_feature_indices=None)

stacker_predictions = stacker.predict(X_test)

print('Stacker Model Results without blending')
print(
    classification_report(
        y_true=y_test,
        y_pred=stacker_predictions
    )
)

print('Accuracy Score: ', accuracy_score(
    y_true=y_test,
    y_pred=stacker_predictions
))

print('*'*100)
# -----------------------------------------------------
# Stacker Model classifier results / eval with blending
# -----------------------------------------------------

lr2 = LogisticRegression(solver='lbfgs', max_iter=10000)
dt2 = DecisionTreeClassifier()
rf2 = RandomForestClassifier(n_estimators=25)
rc2 = RidgeClassifier()
models = [lr2, dt2, rf2]

stacker2 = StackerPyClassifier()
stacker2.fit(
    X=X_train,
    y=y_train,
    models=models,
    stacker=rc2,
    blend=True,
    splits=5,
    model_feature_indices=None)

stacker_predictions2 = stacker2.predict(X_test)

print('Stacker Model Results with blending')
print(
    classification_report(
        y_true=y_test,
        y_pred=stacker_predictions2
    )
)

print('Accuracy Score: ', accuracy_score(
    y_true=y_test,
    y_pred=stacker_predictions2
))


