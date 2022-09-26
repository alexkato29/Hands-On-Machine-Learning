import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""
We can create custom transformers that are 100% compatible with scikit. All you need is a 
class that has 3 methods: fit, transform, and fit_transform. By including TransformerMixin
you automatically gain fit_transform.
"""

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6  # Hardcoded indices from data


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
