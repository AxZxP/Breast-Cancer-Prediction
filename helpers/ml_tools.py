import numpy as np
import pandas as pd

from .viz import pink, yellow, bluewhite, water, blue, green, red
import plotly.graph_objects as go
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, KFold, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Define a feature ingineering object
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # First indicate the indices of each requisite feature
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=False,
                 add_rooms_per_households=False,
                 add_population_per_household=False,
                 remove_base_features=False):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_rooms_per_households = add_rooms_per_households
        self.add_population_per_household = add_population_per_household
        self.remove_base_features = remove_base_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_features = []
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            all_features.append(bedrooms_per_room)
        if self.add_rooms_per_households:
            rooms_per_households = X[:, self.rooms_ix] / X[:, self.households_ix]
            all_features.append(rooms_per_households)
        if self.add_population_per_household:
            population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
            all_features.append(population_per_household)

        return np.column_stack([X, *[f for f in all_features]])


def baseline_classifiers_cross_val(X, y,
                                   scoring='accuracy', scaling=True, multinomial=False,
                                   big_dataset=False, max_iter=100):
    base_models = [
        ('Linear Discrimant Analysis', LinearDiscriminantAnalysis()),
        ('Kneighbors', KNeighborsClassifier()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('SVM linear', SVC(kernel='linear')),
        ('SVM radial', SVC(kernel='rbf'))]

    if multinomial:
        if big_dataset:
            solver = 'saga'
        else:
            solver = 'sag'
    else:
        solver = 'lbfgs'

    base_models.append(('Logistic Regression', LogisticRegression(solver=solver, max_iter=max_iter)))

    baseline_summary = pd.DataFrame(data=None,
                                    columns=['Name', 'Model',
                                             f'{scoring.title()} Mean',
                                             f'{scoring.title()} Std'])

    results = list()
    names = list()
    for idx, (name, model) in enumerate(base_models):

        if big_dataset:
            folds = StratifiedShuffleSplit(n_splits=8, train_size=0.2, test_size=0.08, random_state=seed)
        else:
            folds = KFold(n_splits=10, shuffle=True, random_state=seed)

        if scaling:
            p = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            p = model

        score = cross_val_score(estimator=p,
                                X=X,
                                y=y,
                                scoring=scoring,
                                cv=folds,
                                n_jobs=-1)

        baseline_summary.loc[idx] = [name, model, round(score.mean(), 3), round(score.std(), 3)]
        results.append(score)
        names.append(name)

    baseline_summary.sort_values(f'{scoring.title()} Mean', inplace=True, ascending=False)

    to_plot = pd.DataFrame(data=np.array(results).transpose(), columns=names).T.reset_index()
    to_plot.columns = ['Models'] + to_plot.columns[1:].to_list()
    to_plot = pd.melt(to_plot, id_vars='Models', value_name=scoring.title())

    fig = px.box(to_plot, y=scoring.title(), points='outliers', x='Models')
    fig.update_xaxes(categoryorder='mean descending')

    table = go.Figure(data=[go.Table(
        header=dict(
            values=baseline_summary.columns,
            font_family='AkzidGrtskNext-Med',
            font_size=14,
            height=40,
            fill_color=water,
            align=['center', 'center']),
        cells=dict(
            values=baseline_summary.applymap(lambda x: str(x)).T.values,
            font_family='AkzidGrtskNext-Regular',
            font_size=12,
            height=30,
            align=['center', 'center'],
            fill=dict(color=[bluewhite])
        ))
    ])

    fig.update_layout(
        margin=dict(b=0),
        title={'text': 'Cross Validation Results'},
        width=1000)

    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''

    table.update_layout(
        margin=dict(t=0),
        width=1000)

    fig.show()
    table.show()

    return None


def cross_val_metrics(model, X_train, y_train, seed=42):
    scores = ['accuracy', 'precision', 'recall', 'f1']

    for one_score in scores:
        folds = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring=one_score)
        print(f'[{one_score.title()}] : %0.5f (+/- %0.5f)' % (scores.mean(), scores.std()))
