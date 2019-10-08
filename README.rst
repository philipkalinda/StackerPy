|Travis|_ |PyPi|_ |DOI|_

.. |Travis| image:: https://travis-ci.org/philipkalinda/stackerpy.svg?branch=master
.. _Travis: https://travis-ci.org/philipkalinda/stackerpy

.. |Codecov| image:: https://codecov.io/github/scikit-learn/scikit-learn/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/scikit-learn/scikit-learn?branch=master

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn/scikit-learn/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn/scikit-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

.. |PyPi| image:: https://badge.fury.io/py/scikit-learn.svg
.. _PyPi: https://badge.fury.io/py/scikit-learn

.. |DOI| image:: https://zenodo.org/badge/21369/scikit-learn/scikit-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/scikit-learn/scikit-learn




[![N|Solid](http://www.philipkalinda.com/uploads/8/6/5/4/86541022/untitled-1.png)][MyWebsite]

StackerPy
=========

[![Build Status](https://travis-ci.org/philipkalinda/stackerpy.svg?branch=master)](https://travis-ci.org/philipkalinda/stackerpy)

[![PyPI version](https://badge.fury.io/py/stackerpy.svg)](https://badge.fury.io/py/stackerpy)

Model Stacking for Scikit-Learn Models (including the ability to blend)
 
More Details on Website @: [StackerPy - Model Stacking For Scikit-Learn Models](https://philipkalinda.com/ds10)

### Tech

StackerPy uses a number of open source projects to work properly:

* SciKit-Learn
* Numpy
* Pandas
* Matplotlib


And of course StackerPy itself is open source with a [public repository][StackerPy]
 on GitHub.

### Installation
Install the dependencies (although pip should do this when installing stackerpy)

```sh
$ pip install numpy
$ pip install pandas
$ pip install matplotlib
$ pip install sklearn

```

Install the package

```sh
$ pip install stackerpy
```

### How to use
Stacker Model:
```py
# base models

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from stackerpy import StackerModel

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = pd.DataFrame(data.data)
y = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=np.sum([ord(i) for i in 'StackerPy'])
)

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

![Results](https://raw.githubusercontent.com/philipkalinda/StackerPy/master/stackerpy/Model%20Scoring%20Results.png)



License
----

MIT


[//]: # 


   [StackerPy]: <https://github.com/philipkalinda/StackerPy>
   [MyWebsite]: <http://philipkalinda.com>
   
