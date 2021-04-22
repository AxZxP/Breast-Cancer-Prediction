# Fundamental libraries
import pandas as pd
import numpy as np

# Utility libraries
import joblib
from tempfile import mkdtemp
from shutil import rmtree, copy
from pathlib import Path

# Data pipelining
from sklearn.model_selection import (cross_val_score, GridSearchCV, train_test_split,
                                     StratifiedShuffleSplit, RandomizedSearchCV, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

# Data processing
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.preprocessing import (StandardScaler, PowerTransformer, PolynomialFeatures,
                                   OneHotEncoder, QuantileTransformer)

from sklearn.datasets import fetch_openml, load_breast_cancer

# sampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, RandomUnderSampler
from imblearn.pipeline import Pipeline as SamplePipeline

# Baseline classifiers algo and metrics
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (precision_score, recall_score, classification_report,
                             confusion_matrix, plot_confusion_matrix)


seed = 42


def dump_model_to_drive(model, name):
    filename = f'{name}.sav'
    model_folder = Path.cwd() / 'models'
    Path.mkdir(model_folder, exist_ok=True)
    joblib.dump(model, model_folder / filename)
    #copy(Path.cwd() / f'{filename}', '/content/gdrive/MyDrive')


def main():
    features, labels = load_breast_cancer(return_X_y=True, as_frame=True)
    features.head()

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        stratify=labels)

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    model = Pipeline([
        ('preprocess', StandardScaler()),
        ('f_extract', PolynomialFeatures()),
        ('lda', LinearDiscriminantAnalysis())
    ])

    param_grid = [
        {
            'preprocess': [StandardScaler(), PowerTransformer(method='yeo-johnson', standardize=True)],
            'f_extract__degree': [2, 3],
            'lda__solver': ['lsqr', 'eigen'],
            'lda__shrinkage': ['auto', *[x.round(2) for x in np.arange(0.1, 1, 0.2)]],
        },

        {
            'preprocess': [StandardScaler(), PowerTransformer(method='yeo-johnson', standardize=True)],
            'f_extract__degree': [2, 3],
            'lda__solver': ['svd']
        },
    ]

    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring='recall',
                        cv=folds,
                        refit=True,
                        return_train_score=False,
                        n_jobs=2,
                        verbose=3)

    grid.fit(X_train, y_train)

    dump_model_to_drive(grid, 'breast_cancer_gscv')
    print('DONE')


if __name__ == '__main__':
    main()
