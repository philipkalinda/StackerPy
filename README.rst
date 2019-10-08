
|PKalinda|_

.. |Travis| image:: https://travis-ci.org/philipkalinda/stackerpy.svg?branch=master
.. _Travis: https://travis-ci.org/philipkalinda/stackerpy

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/stackerpy

.. |PyPi| image:: https://badge.fury.io/py/stackerpy.svg
.. _PyPi: https://badge.fury.io/py/stackerpy

.. |PKalinda| image:: http://www.philipkalinda.com/uploads/8/6/5/4/86541022/untitled-1.png
.. _PKalinda: http://philipkalinda.com


StackerPy
=========

|Travis|_ |PyPi|_ |Python35|_


Model Stacking for Scikit-Learn Models (including the ability to blend)

More Details on Website @: `StackerPy - Model Stacking For Scikit-Learn Models <https://philipkalinda.com/ds10>`_

Tech
----

StackerPy uses a number of open source projects to work properly: ::
    
    * SciKit-Learn
    * Numpy
    * Pandas
    * Matplotlib


And of course StackerPy itself is open source with a  `public repository <https://github.com/philipkalinda/StackerPy>`_ on GitHub.

Installation
------------
Install the dependencies (although pip should do this when installing stackerpy) 
.. code-block:: console

    $ pip install numpy
    $ pip install pandas
    $ pip install matplotlib
    $ pip install sklearn


Install the package ::

    ```sh
    $ pip install stackerpy
    ```

How to use
----------
Stacker Model: 

.. code-block:: python

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


Performance
-----------

.. image:: https://raw.githubusercontent.com/philipkalinda/StackerPy/master/stackerpy/Model%20Scoring%20Results.png




License
-------

MIT

