import numpy as np
import pandas as pd
from agora.abc import ParametersABC
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from postprocessor.core.abc import PostProcessABC


class miParameters(ParametersABC):
    """Parameters for the 'mi' process

    Parameters for the 'mi' process.

    Attributes
    ----------
    overtime: boolean (default: True)

        If True, calculate the mutual information as a function of the duration of
        the time series, by finding the mutuation information for all possible
        sub-time series that start from t= 0.


    n_bootstraps: int, optional (default: 100)

        The number of bootstraps used to estimate errors.


    ci: 1x2 array or list, optional (default: [0.25, 0.75])

        The lower and upper confidence intervals.

        E.g. [0.25, 0.75] for the interquartile range


    Crange: array, optional

        An array of potential values for the C parameter of the support vector machine
        and from which the optimal value of C will be chosen.

        If None, np.logspace(-3, 3, 10) is used. This range should be increased if
        the optimal C is one of the boundary values.

        See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


    gammarange: array, optional

        An array of potential values for the gamma parameter for the radial basis
        function kernel of the support vector machine and from which the optimal
        value of gamma will be chosen.

        If None, np.logspace(-3, 3, 10) is used. This range should be increased if
        the optimal gamma is one of the boundary values.

        See https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


    train_test_split_seeding: boolean, optional (default: False)

        If True, force a random state for the train-test split in each bootstrap. This is
        useful in case the user requires reproducibility e.g. code testing.

        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    _defaults = {
        "overtime": True,
        "n_bootstraps": 100,
        "ci": [0.25, 0.75],
        "Crange": None,
        "gammarange": None,
        "train_test_split_seeding": False,
    }


class mi(PostProcessABC):
    """
    Process to estimate power spectral density (autoregressive model).

    Methods
    -------
    run(signal: pd.DataFrame)
        Estimates the mutual information between classes of time series.
    """

    def __init__(
        self,
        parameters: miParameters,
    ):
        super().__init__(parameters)

    def run(self, signals):
        """
        Estimates the mutual information between classes of time series.

        Uses sklean to optimise a pipeline for classifying the individual time series,
        choosing the number of PCA components (3-7), the classifier - a support vector
        machine with either a linear or a radial basis function kernel - and its C and
        gamma parameters.

        Errors are found using bootstrapped datasets.

        Parameters:  signals: list of pandas.DataFrames

                        A list of DataFrames.  Each DataFrame stores a set of time series, with rows
                        indicating individual time series (e.g. from each cell), and columns
                        indicating time points.

        Returns:    res: array

                        Summary statistics from the bootstrapped datasets -- the median mutual
                        information and the 10% and 90% confidence limits.

                        If overtime is True, each row corresponds to a different duration of the time
                        series with the shortest duration, just the first time point, in the first row
                        and the longest duration, the entire time series, in the last row.
        """

        # default values
        if not np.any(self.Crange):
            self.Crange = np.logspace(-3, 3, 10)
        if not np.any(self.gammarange):
            self.gammarange = np.logspace(-3, 3, 10)

        # data is a list with one array of time series for different class
        data = [signal.to_numpy() for signal in signals]
        n_classes = len(data)
        Xo = np.vstack([timeseries for timeseries in data])
        y = np.hstack(
            [
                i * np.ones(timeseries.shape[0])
                for i, timeseries in enumerate(data)
            ]
        )

        if self.overtime:
            # loop over time series iteratively
            durations = np.arange(1, Xo.shape[1] + 1)
        else:
            # full time series only
            durations = [Xo.shape[1]]
        # array for results
        res = np.zeros((len(durations), 3))

        for j, duration in enumerate(durations):
            # slice of of time series
            # if verbose:
            #    print("duration of time series is", duration)
            X = Xo[:, :duration]

            # initialise sself.cikit-learn routines
            nPCArange = (
                range(1, X.shape[1] + 1) if X.shape[1] < 7 else [3, 4, 5, 6, 7]
            )
            params = [
                {"project__n_components": nPCArange},
                {
                    "classifier__kernel": ["linear"],
                    "classifier__C": self.Crange,
                },
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": self.Crange,
                    "classifier__gamma": self.gammarange,
                },
            ]
            pipe = Pipeline(
                [
                    ("project", PCA()),
                    ("rescale", StandardScaler()),
                    ("classifier", svm.SVC()),
                ]
            )

            # find best params for pipeline
            grid_pipeline = GridSearchCV(pipe, params, n_jobs=-1, cv=5)
            grid_pipeline.fit(X, y)
            # if verbose:
            #    print(grid_pipeline.best_estimator_)
            pipe.set_params(**grid_pipeline.best_params_)

            # find mutual information for each bootstrapped dataset
            mi = np.empty(self.n_bootstraps)
            for i in range(self.n_bootstraps):
                # force random state, useful for code testing/reproducibility
                if self.train_test_split_seeding:
                    random_state = i
                else:
                    random_state = None
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=random_state,
                )
                # run classifier use optimal params
                pipe.fit(X_train, y_train)
                y_predict = pipe.predict(X_test)
                # estimate mutual information
                p_y = 1 / n_classes
                p_yhat_given_y = confusion_matrix(
                    y_test, y_predict, normalize="true"
                )
                p_yhat = np.sum(p_y * p_yhat_given_y, 0)
                h_yhat = -np.sum(
                    p_yhat[p_yhat > 0] * np.log2(p_yhat[p_yhat > 0])
                )
                log2_p_yhat_given_y = np.ma.log2(p_yhat_given_y).filled(0)
                h_yhat_given_y = -np.sum(
                    p_y * np.sum(p_yhat_given_y * log2_p_yhat_given_y, 1)
                )
                mi[i] = h_yhat - h_yhat_given_y

            # summary statistics - median and confidence intervals
            res[j, :] = [
                np.median(mi),
                np.sort(mi)[int(np.min(self.ci) * self.n_bootstraps)],
                np.sort(mi)[int(np.max(self.ci) * self.n_bootstraps)],
            ]
            # if verbose:
            #    print(f"median MI= {res[j,0]:.2f} [{res[j,1]:.2f}, {res[j,2]:.2f}]")
        return res
