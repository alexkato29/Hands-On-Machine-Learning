import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from CombinedAttributesAdder import CombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

"""
Configure the paths for getting and storing data.
DOWNLOAD_ROOT represents the root url for the github repo with the data
HOUSING_PATH is the area where the data is to be stored
HOUSING_URL just appends the necessary directories to the download root url
"""
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    os.makedirs creates the housing directory (exist_ok prevents an error being thrown
    if it already exists). We then make a get request to download the data. The tgz file
    is unzipped and the csv is saved to the directory.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Load and preview the dataset
fetch_housing_data()
housing = load_housing_data()

pd.options.display.max_columns = None
print(housing.head())
print("\n\n")
housing.info()
print("\n\n")
print(housing["ocean_proximity"].value_counts())  # See what the categories are
print("\n\n")
print(housing.describe())


# housing.hist(bins=50, figsize=(10,10))
# plt.show()


def split_train_test(data, test_ratio):
    """
    :param data: the dataset we are splitting
    :param test_ratio: the ratio of the data we want to be test data
    :return: returns the dataset split into a train and test portion
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print("\n\nTrain Set Length: %d\nTest Set Length: %d" % (len(train_set), len(test_set)))
"""
This train test split works fine, but it always generates a random split. Suppose you
train a model, test it, and want to tweak it or the dataset. Now, when you call the function
again, you will see new data. Eventually, your model will see all the data and be able to
train on it, making this useless.

Instead we use other techniques to implement a train test split in practice.
"""


def test_set_check(identifier, test_ratio):
    # crc32 is a hash function. To be honest I don't know why this reliably produces a
    # perfect split everytime. It seems like this should produce varying results within
    # a few percentage points of the train test split...
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_col):
    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index()  # Adds an index column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print("\n\nTrain Set Length: %d\nTest Set Length: %d" % (len(train_set), len(test_set)))

"""
When sampling data, we want to make sure that it is representative of the true population.
If 54% of people are female and 46% are male, our TEST DATA should reflect this since we
want our testing and validation data to be as true to life as possible. This way we know
our model generalizes well when finalizing it and does not over/under fit the training data.

This is known as stratified sampling. Suppose we want to do so for median_income. We can
make a category for income and ensure that the test data has an amount of each category
proportional the true population (in this case our dataset since we do not know the true
population). It's important we don't have too many strata, as strata with not enough 
instances may give inaccurate estimates of its importance.

Scikit can perform this stratified train/test split for you!
"""
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove the income_cat from the data now that it has been used
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# We put aside the test data so that we don't potentially make biased assumptions in our analysis
housing = strat_train_set.copy()

# Plot the median house value and population density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,  # alpha=0.1 lets you see density
             s=housing["population"] / 100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,  # Color map doesn't work (??)
             )
plt.legend()
# plt.show()

# Show the correlation between median house value and all the other attributes
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

"""
From this last plot we can see there is a very strong correlation. There are also some 
weird straight lines at 500k, 450k, and 350k. We might want to remove these data points
to prevent our model from replicating these quirks.
"""

# We add additional features we think would be helpful
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
# Some of our new features prove useful! Like bedrooms_per_room

# We (using the fresh data) separate the predictors from the labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

"""
We want to clean the data. For numerical values suppose we have missing vals. We can:
1) Get rid of individual rows with the missing data
2) Delete the feature altogether
3) Set the missing values to something (zero, mean, median, etc)
This can be done with scikit
"""
imputer = SimpleImputer(strategy="median")
# We must remove the non-numerical features since median can only be computed on numbers
# Also note: .drop() returns an updated copy and DOES NOT modify the original data
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# For text or categorical data, we usually want to convert them to numbers
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()  # Converts text to a numerical scale
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

"""
A problem with this is that while its great we have 1, 2, 3, ... models consider closer
numbers to be similar and further numbers to be different. This can be good sometimes, but,
in the case of ocean prox., it is not intended.

We create 5 (for the 5 ocean prox options) features to replace ocean prox and set them to
0 or 1 to represent our categories. We call these new attributes dummy attributes.
"""

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print("\n\n")
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

"""
Models don't perform well when their features range massively in scale. We scale features
to combat this. There are two common way to do this:

min-max scaling: make all values between 0 and 1
standardization: (x - mean) / std. dev for all points

HUGE NOTE: it is crucial with any transformation that you apply it to only the training
data, then after the fact to the test data. We don't want test data to influence any
feature scaling or transformations since test data is supposed to be natural data!

There are lots of data cleaning transformations we want to do that require a specific 
order. Scikit can help us do this using pipelines
"""

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Replace missing vals w/ median
    ("attribs_adder", CombinedAttributesAdder()),  # Adds attributes we want
    ("std_scaler", StandardScaler())  # Feature scaling
])

num_attributes = list(housing_num)
cat_attributes = ["ocean_proximity"]

"""
This full pipeline applies all numerical AND categorical changes. Still requires we make
a num_pipeline though. Tuple is (name, transformer, columns to apply to). If you want,
putting "drop" or "passthrough" in transformer will cause the columns to be dropped or
ignored. You can also specify the "remainder" hyperparameter, which will tell the pipeline
what to do with all leftover columns.
"""
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", OneHotEncoder(), cat_attributes)
])
housing_prepared = full_pipeline.fit_transform(housing)

# Now we can train our model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# We can now test it
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
# Calling fit_transform before does not alter the original data but rather returns a copy
some_data_prepared = full_pipeline.transform(some_data)  # Why not fit_transform??
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("\n\n", lin_rmse)

"""
The first model does not do a great job. The root mean squared error is ~$68k. This is very high
and an indication that our model is very underfit. We can combat this in 3 ways:

1) Select a more powerful model
2) Feed the training algorithm with better features
3) Reduce the constraints on the model
"""
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)  # The RMSE is 0. This is likely because the model is very overfit
print(tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Dev:", scores.std())
    print("\n\n")


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

"""
Here we use cross validation to really assess how the models are doing. This sklearn function
splits the training data into 10 folds. It trains the model on 9 of those 10 folds, and then 
validates it on the remaining fold. It does this for the other 9 folds as well.

Using cross-validation, we can see that the decision tree doesn't look so good anymore! It was
overfitting so badly, it actually did worse than the regression!
"""
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(tree_mse)
print(forest_rmse)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
# Using a RandomForest we get our best score yet! But it takes FOREVER to run. Welcome to ML.
