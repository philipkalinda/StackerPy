import pandas as pd
from sklearn.model_selection import KFold
import numpy as np


class StackerPyClassifier:

    def __init__(self):
        """
        """
        self.X = None
        self.y = None
        self.blend = None
        self.splits = None
        self.model_feature_indices = None
        self.model_methods = []
        self.metafeatures_df = None
        self.X_with_metafeatures = None
        self.raw_models = None
        self.fit_models = dict()
        self.fit_blended_models = dict()
        self.raw_stacker = None
        self.stacker = None

    def validate_models(self, models):
        for model in models:
            # model name
            model_name = model.__str__().split('(')[0]

            # assert that each model has a predict_proba or predict method
            assert any(['predict' in dir(model), 'predict_proba' in dir(model)]), \
                f"""{model.__str__().split('(')[0]} does not have a predict or predict_proba method"""

            # determine which method to use from each of the models in the ist and keep stored
            if 'predict_proba' not in dir(model) and 'predict' in dir(model):
                print(f'Note: There is no predict_proba method in {model_name}, therefore predict method will be used')
                self.model_methods.append('predict')

            if 'predict_proba' in dir(model):
                self.model_methods.append('predict_proba')

    @staticmethod
    def model_predictor(model, X, method):
        """
        :param model:
        :param X:
        :param method:
        :return:
        """
        
        predictions = None
        if method == 'predict_proba':
            predictions = model.predict_proba(X)
        if method == 'predict':
            predictions = model.predict(X)

        return predictions

    def fit(self, X, y, models, stacker, blend=False, splits=5, model_feature_indices=None):
        """
        :param X: 
        :param y: 
        :param models: 
        :param stacker: 
        :param blend: 
        :param splits: 
        :param model_feature_indices: 
        :return: 
        """

        # re-initialise so that you can
        self.__init__()

        # some saved variables during fit
        self.raw_models = models
        self.raw_stacker = stacker
        self.blend = blend
        self.splits = splits
        self.model_feature_indices = model_feature_indices

        self.validate_models(models)

        # convert X into a dataframe if not already
        if str(type(X)) != """<class 'pandas.core.frame.DataFrame'>""":
            X = pd.DataFrame(X)

        if self.model_feature_indices is None:
            self.model_feature_indices = [[i for i in range(X.shape[1])] for _ in models]

        for model, features, method in zip(self.raw_models, self.model_feature_indices, self.model_methods):

            # model_name
            model_name = model.__str__().split('(')[0]

            # train model
            X_model_features = X.iloc[:, features]

            metafeatures = None
            # blending if required
            if self.blend is False:
                model.fit(X_model_features, y)

                metafeatures = self.model_predictor(model, X_model_features, method)

                # store fit model for future metafeature predictions
                self.fit_models[model_name] = model

            if self.blend is True:

                self.fit_blended_models[model_name] = []
                # folder for blending
                kf = KFold(n_splits=self.splits)

                # metafeatures
                metafeatures = pd.Series(np.zeros(X.shape[0]))
                for idx, (train_idx, meta_idx) in enumerate(kf.split(X_model_features)):

                    # fit to train
                    model.fit(X_model_features.iloc[train_idx, :], y.iloc[train_idx, :])

                    meta = None
                    # predict meta
                    if method == 'predict_proba':
                        meta = model.predict_proba(X_model_features.iloc[meta_idx, :])
                    if method == 'predict':
                        meta = model.predict(X_model_features.iloc[meta_idx, :])

                    # append metas
                    metafeatures.iloc[meta_idx, :] = meta

                    # store fit model for future metafeature predictions
                    self.fit_blended_models[model_name].append(model)

            # append metafeatures to the metafeature dataframe
            self.metafeatures_df[model_name] = metafeatures

        # create final df with metafeatures
        self.X_with_metafeatures = pd.concat([X, self.metafeatures_df], axis=1)

        # final stacked X-fit
        self.stacker = stacker.fit(self.X_with_metafeatures, y)

    def predict(self, X):

        # convert X into a dataframe if not already
        if str(type(X)) != """<class 'pandas.core.frame.DataFrame'>""":
            X = pd.DataFrame(X)

        metafeatures_df = pd.DataFrame()

        if self.blend is False:

            for (model_name, model), features, method in zip(self.fit_models.items(), self.model_methods):

                X_model_features = X.iloc[:, features]

                metafeatures = self.model_predictor(model, X_model_features, method)

                metafeatures_df[model_name] = metafeatures

        if self.blend is True:

            # loop through all the available model types
            for (model_name, model_list), features, method in zip(self.fit_blended_models.items(), self.model_methods):

                X_model_features = X.iloc[:, features]

                model_df = pd.DataFrame()

                # loop through all the different models that were split during blended
                for model_idx, model in enumerate(model_list):

                    # predict meta
                    meta = self.model_predictor(model, X_model_features, method)

                    model_df[model_idx] = meta

                # average predictions from all different models from the blending process
                metafeatures = np.mean(model_df, axis=1)

                metafeatures_df[model_name] = metafeatures

        X_with_metafeatures = pd.concat([X, metafeatures_df], axis=1)

        # make predictions
        predictions = self.stacker.predict(X_with_metafeatures)

        return predictions
