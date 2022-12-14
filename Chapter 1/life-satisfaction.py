import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors


# Given function
def prepare_country_stats(data1, data2):
    data1 = data1[data1["INEQUALITY"]=="TOT"]
    data1 = data1.pivot(index="Country", columns="Indicator", values="Value")
    data2.rename(columns={"2015": "GDP per capita"}, inplace=True)
    data2.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=data1, right=data2,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# Load the data
oecd_bli = pd.read_csv("../datasets/life-satisfaction/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("../datasets/life-satisfaction/gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding="latin1", na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()

# Select a model. We can use instance-based or model-based learning
# model = sklearn.linear_model.LinearRegression()
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus GDP per Capita
print(model.predict(X_new))
