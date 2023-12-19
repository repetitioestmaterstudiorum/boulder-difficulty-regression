import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time

RANDOM_SEED = 51

np.random.seed(RANDOM_SEED)

def get_data_item(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"./{name}")
    if df.columns[0] == 'Unnamed: 0':
        print('Removing first column')
        df.drop(columns=df.columns[0], axis=1, inplace=True)
    return df

X_train = get_data_item('X_train-holds-binary.csv')
y_train = get_data_item('y_train-holds-binary.csv')

X = X_train.values
y = y_train.values.ravel()

def get_random_percentage_of_data(X, y, percentage):
    num_samples = int(len(X) * percentage)
    indices = np.random.choice(len(X), num_samples, replace=False)
    return X[indices], y[indices]

amount_of_data = 1
X, y = get_random_percentage_of_data(X, y, amount_of_data)
print(f"X: {X.shape}, y: {y.shape}")

from xgboost import XGBRegressor

rgr = XGBRegressor(random_state=RANDOM_SEED)

# XGBRegressor hyperparameters: https://xgboost.readthedocs.io/en/stable/parameter.html#
param_grid = {
    'learning_rate': [0.1], # default 0.3
    'gamma': [0], # default 0
    'max_depth': [20], # default 6
    'min_child_weight': [2, 10], # default 1
    'subsample': [1], # default 1
    'colsample_bytree': [0.5], # default 1
    'lambda': [1], # default 1
    'alpha': [0.2, 1], # default 0
    'max_bin': [512], # default 256 - higher means more optimal splits but slower
}

unique_combinations = np.prod([len(param_grid[key]) for key in param_grid.keys()])
print('Unique combinations: ', unique_combinations)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
rmse = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
r2_scorer = make_scorer(r2_score)

scoring = {'MSE': mse_scorer, 'MAE': mae_scorer, 'RMSE': rmse, 'R2': r2_scorer}

search_type = 'grid' # 'grid' or 'random'
max_iterations = 1

if search_type == 'grid':
    search = GridSearchCV(rgr, param_grid=param_grid, cv=5, scoring=scoring, refit='MAE')
elif search_type == 'random':
    search = RandomizedSearchCV(rgr, param_distributions=param_grid, n_iter=max_iterations, cv=5, scoring=scoring, refit='MAE', random_state=RANDOM_SEED)
else:
    raise ValueError('Invalid search type')

time_start = time.time()
search.fit(X, y)
time_end = time.time()
duration = time_end - time_start

print(f"Search: '{search_type}' with {amount_of_data*100:.0f}% of data and {unique_combinations if search_type == 'grid' else max_iterations} of {unique_combinations} combintations. Random state: {RANDOM_SEED}. Duration in minutes: {duration/60:.1f}")
print(f"Best params: {search.best_params_}")
print(f"Best MSE: {-search.cv_results_['mean_test_MSE'][search.best_index_]}")
print(f"Best MAE: {-search.cv_results_['mean_test_MAE'][search.best_index_]}")
print(f"Best RMSE: {-search.cv_results_['mean_test_RMSE'][search.best_index_]}")
print(f"Best R2 score: {search.cv_results_['mean_test_R2'][search.best_index_]}")

results = search.cv_results_

df = pd.DataFrame(results["params"])
df["Mean MSE"] = -results["mean_test_MSE"]
df["Mean MAE"] = -results["mean_test_MAE"]
df["Mean RMSE"] = -results["mean_test_RMSE"]
df["Mean R2"] = results["mean_test_R2"]

end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")
df.to_csv(f"search_results--{end_timestamp}.csv", index=False)
