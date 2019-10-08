# StackerPy

[![Build Status](https://travis-ci.org/philipkalinda/stackerpy.svg?branch=master)](https://travis-ci.org/philipkalinda/stackerpy)

Model Stacking for Scikit-Learn Models
 
More Details on Website @: [StackerPy - Model Stacking For Scikit-Learn Models](https://philipkalinda.com/ds10)


### How to use
Stacker Model:
```py
# base models
lr2 = LogisticRegression(solver='lbfgs')
dt2 = DecisionTreeClassifier()
rf2 = RandomForestClassifier()
models = [lr2, dt2, rf2]

# stacker
rc2 = RidgeClassifier()

#fitting
stacker = StackerModel()
stacker.fit(
    X=X_train,
    y=y_train,
    models=models,
    stacker=rc2,
    blend=True,
    splits=5,
    model_feature_indices=None)

# predicting
stacker_predictions = stacker.predict(X_test)
```

### Performance

