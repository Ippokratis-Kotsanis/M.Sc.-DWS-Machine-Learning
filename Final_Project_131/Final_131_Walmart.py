# Final Project - Ιπποκράτης Κοτσάνης 131
# (131 mod 45) + 1 = 42

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from IPython.display import FileLink

# pd.options.mode.chained_assignment = None  # default='warn'

# Load dataset
walmart_dataset = pd.read_csv('walmart_cleaned.csv')
df = pd.DataFrame(walmart_dataset)

# =====================================================================================
# ==========================Dataset exploration========================================
# =====================================================================================

# Select Store
Store_42 = df[df["Store"] == 42]
print(Store_42)
Store_42_clean = Store_42.iloc[:, 1:]

# input Variables Correlation with the output feature Weekly_Sales
corr = Store_42_clean.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.show()

# COMMENT
# By watching the correlation matrix, we can see that Weekly_Sales have a higher correlation with Store, Dept and Size.
# We will drop the variables with lower correlation in the Data Manipulation section.

# =====================================================================================
# ========================== Data Manipulation ========================================
# =====================================================================================
# Now, we will do the following steps:
#
#     - Remove null values from the markdown variables.
#     - Create variables for year, month and week, based on the date field.
#     - Remove the variables with low correlation.
#
# =====================================================================================

# Here we check the percentance of none values in every feature.
nan_percentage = walmart_dataset.isnull().mean() * 100
# print(nan_percentage)

# Create variable for year based on the date field.
Store_42_clean['Year'] = pd.to_datetime(Store_42_clean['Date']).dt.year

# Here we inert a column for week number
Store_42_clean['Date'] = pd.to_datetime(Store_42_clean['Date'], dayfirst=True)
Store_42_clean['Week'] = pd.to_datetime(Store_42_clean['Date']).dt.isocalendar().week

# We can move the target variable to the last column of the dataframe to ease the manipulation of the data.
df_1 = Store_42_clean.pop('Next week')
Store_42_clean['Next week'] = df_1

# Remove the variables with low correlation.
Store_42_clean = Store_42_clean.drop(columns=["Date", "CPI", "Fuel_Price", 'Temperature'])
# print(Store_42_clean)

nan_percentage_2 = Store_42_clean.isnull().mean() * 100
print(nan_percentage_2)

# Here, we identify inputs and target columns.
input_cols, target_col = Store_42_clean.columns[:-1], Store_42_clean.columns[-1]
inputs_df, targets = Store_42_clean[input_cols].copy(), Store_42_clean[target_col].copy()
print("Targets: ", targets)

# Now, we identify numeric and categorical columns.
numeric_cols = Store_42_clean[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = Store_42_clean[input_cols].select_dtypes(include='object').columns.tolist()
print('====Numeric columns===', numeric_cols)
print('====categorical columns===', categorical_cols)

# Here, we impute (fill) and scale numeric columns.
imputer = SimpleImputer().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])
scaler = MinMaxScaler().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])

# We can only use numeric data to train our models, that's why we have to use a technique called "one-hot encoding"
# for our categorical columns.

# One hot encoding involves adding a new binary (0/1) column for each unique category of a categorical column.

# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(inputs_df[categorical_cols])
# encoded_cols = list(encoder.get_feature_names(categorical_cols))
# inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])

# COMMENT: WE DO NOT HAVE CATEGORICAL COLUMNS.

# Select 3 Depts from Store 42
Store_42_Dept_1 = Store_42_clean[Store_42_clean["Dept"] == 1]
Store_42_Dept_10 = Store_42_clean[Store_42_clean["Dept"] == 10]
Store_42_Dept_11 = Store_42_clean[Store_42_clean["Dept"] == 11]

# select department
select_department = Store_42_Dept_11

# Now we must insert values into next week for training. Next week wil has the value of weekly sales from the next week.
weeklySales = []
for row in select_department['Weekly_Sales']:
    weeklySales.append(row)
weeklySales.pop(0)
weeklySales.append(None)
select_department['Next week'] = weeklySales

# divide dataset to train, test and validation data
Store_42_Dept_train = select_department[select_department["Year"] == 2010]
Store_42_Dept_validation = select_department[select_department["Year"] == 2011]
Store_42_Dept_test = select_department[select_department["Year"] == 2012]
Store_42_Dept_test = Store_42_Dept_test.iloc[:-1]  # here we delete the last week of the dataset beacuse it is none.

print("=========TRAIN SET===================")
print(Store_42_Dept_train)
print("=========VALIDATION SET===================")
print(Store_42_Dept_validation)
print("=========TEST SET===================")
print(Store_42_Dept_test)
print("===============================================")

# =====================================================================================
# ================================= TRAINING ==========================================
# =====================================================================================

# Finally, let's split the dataset into a training and validation set. Also, we'll use just the numeric and encoded
# columns, since the inputs to our model must be numbers. Split data to X and y(target).

Store_42_Dept_x_train = Store_42_Dept_train.drop(columns=['Next week'])
print(Store_42_Dept_x_train.columns)
Store_42_Dept_y_train = Store_42_Dept_train.iloc[:, -1:]
print(Store_42_Dept_x_train)
################################################################################################
Store_42_Dept_x_val = Store_42_Dept_validation.drop(columns=['Next week'])
Store_42_Dept_y_val = Store_42_Dept_validation.iloc[:, -1:]
print(Store_42_Dept_x_val)
################################################################################################
Store_42_Dept_x_test = Store_42_Dept_test.drop(columns=['Next week'])
Store_42_Dept_y_test = Store_42_Dept_test.iloc[:, -1:]
print(Store_42_Dept_x_test)
################################################################################################

# Now that we have our training and validation set, we will review two models of machine learning:
#
#     - Decision Tree
#     - Random Forest
#
# Based on the results, we will pick one of them.

# =====================================================================================
# ==========================Decision Tree==============================================
# =====================================================================================

tree = DecisionTreeRegressor(random_state=0)
tree.fit(Store_42_Dept_x_train, Store_42_Dept_y_train)

tree_val_preds = tree.predict(Store_42_Dept_x_val)
tree_val_rmse = mean_squared_error(Store_42_Dept_y_val, tree_val_preds, squared=False)

tree_test_preds = tree.predict(Store_42_Dept_x_test)
tree_test_rmse = mean_squared_error(Store_42_Dept_y_test, tree_test_preds, squared=False)

print('Validation RMSE: {}, Test RMSE: {}'.format(tree_val_rmse, tree_test_rmse))

# Decision tree visualization
sns.set_style('darkgrid')
plt.figure(figsize=(30, 15))
plot_tree(tree, feature_names=Store_42_Dept_x_test.columns, max_depth=3, filled=True)
plt.show()

tree_text = export_text(tree, feature_names=list(Store_42_Dept_x_train.columns))
# print(tree_text[:2000])

# Decision Tree feature importance
tree_importances = tree.feature_importances_
tree_importance_df = pd.DataFrame({'feature': Store_42_Dept_x_train.columns,
                                   'importance': tree_importances}).sort_values('importance', ascending=False)
# print(tree_importance_df)
plt.title('Decision Tree Feature Importance')
sns.barplot(data=tree_importance_df.head(10), x='importance', y='feature')
plt.show()

# =====================================================================================
# ==========================Random Forest==============================================
# =====================================================================================
rf1 = RandomForestRegressor(random_state=0, n_estimators=10)

# Fit our model
rf1.fit(Store_42_Dept_x_train, Store_42_Dept_y_train.values.ravel())

rf1_train_preds = rf1.predict(Store_42_Dept_x_train)
rf1_train_rmse = mean_squared_error(Store_42_Dept_y_train, rf1_train_preds, squared=False)

rf1_val_preds = rf1.predict(Store_42_Dept_x_val)
rf1_val_rmse = mean_squared_error(Store_42_Dept_y_val, rf1_val_preds, squared=False)

rf1_test_preds = rf1.predict(Store_42_Dept_x_test)
rf1_test_rmse = mean_squared_error(Store_42_Dept_y_test, rf1_test_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}, Test RMSE: {}'.format(rf1_train_rmse, rf1_val_rmse, rf1_test_rmse))


# COMMENT
# The random forest model shows better results for the validation RMSE, so we will use that model.

# =====================================================================================
# ========================== HYPERPARAMETER TUNING ====================================
# =====================================================================================

# Let's define a helper function test_params which can test the given value of one or more hyperparameters.
# For this new random forest model, I will use a number of estimators of 100.

def test_params(**params):
    model = RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=100, **params).fit(Store_42_Dept_x_train,
                                                                                             Store_42_Dept_y_train.
                                                                                             values.ravel())
    train_rmse = mean_squared_error(model.predict(Store_42_Dept_x_train), Store_42_Dept_y_train, squared=False)
    val_rmse = mean_squared_error(model.predict(Store_42_Dept_x_val), Store_42_Dept_y_val, squared=False)
    return train_rmse, val_rmse


def test_params_TEST_SET(**params):
    model1 = RandomForestRegressor(random_state=0, n_jobs=-1, n_estimators=100, **params).fit(Store_42_Dept_x_train,
                                                                                              Store_42_Dept_y_train.
                                                                                              values.ravel())
    train_rmse = mean_squared_error(model1.predict(Store_42_Dept_x_train), Store_42_Dept_y_train, squared=False)
    test_rmse = mean_squared_error(model1.predict(Store_42_Dept_x_test), Store_42_Dept_y_test, squared=False)
    return train_rmse, test_rmse


# Let's also define a helper function to test and plot different values of a single parameter.


def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], []
    for value in param_values:
        params = {param_name: value}
        train_rmse, val_rmse = test_params(**params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    plt.figure(figsize=(10, 6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])
    plt.show()


def test_param_and_plot_TEST_SET(param_name, param_values):
    train_errors, test_errors = [], []
    for value in param_values:
        params = {param_name: value}
        train_rmse, test_rmse = test_params_TEST_SET(**params)
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
    plt.figure(figsize=(10, 6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, test_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Test'])
    plt.show()


# FOR TESTING SET GO TO LINE: 330

print("Results with higher n_estimators", test_params())
# we can see better results with a higher number of estimators.

print(test_param_and_plot('min_samples_leaf', [1, 2, 3, 4, 5]))
print(test_params(min_samples_leaf=5))
# Here, we can see how the RMSE increases with the min_samples_leaf parameter, so we will use the default value (1).

print(test_param_and_plot('max_leaf_nodes', [2, 5, 10, 25, 40]))
print(test_params(max_leaf_nodes=20))
# The RMSE decreases with the max_leaf_nodes parameter, so we will use the default value (none).

print(test_param_and_plot('max_depth', [5, 10, 15, 20, 25]))
print(test_params(max_depth=5))
# The RMSE decreases with the max_depth parameter, so we will use the default value (none).

# =====================================================================================
# ========================== TRAINING THE BEST MODEL ====================================
# =====================================================================================

# We create a new Random Forest model with custom hyperparameters.
rf2 = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_leaf=1)

# Now we train the model.
rf2.fit(Store_42_Dept_x_train, Store_42_Dept_y_train.values.ravel())

rf2_train_preds = rf2.predict(Store_42_Dept_x_train)
rf2_train_rmse = mean_squared_error(Store_42_Dept_y_train, rf2_train_preds, squared=False)

rf2_val_preds = rf2.predict(Store_42_Dept_x_val)
rf2_val_rmse = mean_squared_error(Store_42_Dept_y_val, rf2_val_preds, squared=False)

print("===========================================================================")
print('Train RMSE: {}, Validation RMSE: {}'.format(rf2_train_rmse, rf2_val_rmse))
# Here we can see a decrease for the RMSE loss.

# RANDOM FOREST FEATURE IMPORTANCE
rf2_importance_df = pd.DataFrame({'feature': Store_42_Dept_x_train.columns,
                                  'importance': rf2.feature_importances_}).sort_values('importance', ascending=False)
# print(rf2_importance_df)
plt.title('Random Forest Feature Importance')
sns.barplot(data=rf2_importance_df.head(10), x='importance', y='feature')
plt.show()

# MAKING PREDICTIONS ON THE VALIDATION SET
val_preds = rf2.predict(Store_42_Dept_x_val)
Store_42_Dept_validation['Next week predictions'] = val_preds
print(Store_42_Dept_validation)

# =====================================================================================
# ========================== PREDICTIONS FOR TEST SET =================================
# =====================================================================================

print("Results with higher n_estimators", test_params_TEST_SET())
# we can see better results with a higher number of estimators.

print(test_param_and_plot_TEST_SET('min_samples_leaf', [1, 2, 3, 4, 5]))
print(test_params_TEST_SET(min_samples_leaf=5))
# Here, we can see how the RMSE increases with the min_samples_leaf parameter, so we will use the default value (1).

print(test_param_and_plot_TEST_SET('max_leaf_nodes', [2, 5, 10, 25, 40]))
print(test_params_TEST_SET(max_leaf_nodes=20))
# The RMSE decreases with the max_leaf_nodes parameter, so we will use the default value (none).

print(test_param_and_plot_TEST_SET('max_depth', [5, 10, 15, 20, 25]))
print(test_params_TEST_SET(max_depth=5))
# The RMSE decreases with the max_depth parameter, so we will use the default value (none).

# =====================================================================================
# ========================== TRAINING THE BEST MODEL ==================================
# =====================================================================================

# We create a new Random Forest model with custom hyperparameters.
rf2 = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_leaf=1)

# Now we train the model.
rf2.fit(Store_42_Dept_x_train, Store_42_Dept_y_train.values.ravel())

rf2_train_preds = rf2.predict(Store_42_Dept_x_train)
rf2_train_rmse = mean_squared_error(Store_42_Dept_y_train, rf2_train_preds, squared=False)

rf2_test_preds = rf2.predict(Store_42_Dept_x_test)
rf2_test_rmse = mean_squared_error(Store_42_Dept_y_test, rf2_test_preds, squared=False)

print("===========================================================================")
print('Train RMSE: {}, Validation RMSE: {}, Test RMSE: {}'.format(rf2_train_rmse, rf2_val_rmse, rf2_test_rmse))
# Here we can see a decrease for the RMSE loss.

# RANDOM FOREST FEATURE IMPORTANCE
rf2_importance_df = pd.DataFrame({'feature': Store_42_Dept_x_train.columns,
                                  'importance': rf2.feature_importances_}).sort_values('importance', ascending=False)
# print(rf2_importance_df)
plt.title('Random Forest Feature Importance')
sns.barplot(data=rf2_importance_df.head(10), x='importance', y='feature')
plt.show()

# MAKING PREDICTIONS ON THE TEST SET
test_preds = rf2.predict(Store_42_Dept_x_test)
Store_42_Dept_test['Next week predictions'] = test_preds
print(Store_42_Dept_test)

# =====================================================================================
# ========================== RESULT PLOTS =============================================
# =====================================================================================

# Validation set
plt.plot(Store_42_Dept_validation['Week'], Store_42_Dept_validation['Next week'], color='blue', marker='o')
plt.plot(Store_42_Dept_validation['Week'], Store_42_Dept_validation['Next week predictions'], color='red', marker='o')
plt.title('2011 PREDICTIONS - TRAINING SET 2010', fontsize=14)
plt.xlabel('week', fontsize=14)
plt.ylabel('Next week predictions rate', fontsize=14)
plt.grid(True)
plt.show()

# Test set
plt.plot(Store_42_Dept_test['Week'], Store_42_Dept_test['Next week'], color='blue', marker='o')
plt.plot(Store_42_Dept_test['Week'], Store_42_Dept_test['Next week predictions'], color='red', marker='o')
plt.title('2012 PREDICTIONS - TRAINING SET 2010', fontsize=14)
plt.xlabel('week', fontsize=14)
plt.ylabel('Next week predictions rate', fontsize=14)
plt.grid(True)
plt.show()

# =====================================================================================
# ========================== RESULT CSV for each DEPARTMENT ===========================
# =====================================================================================

# WE CREATE 2 CSVs FILES FOR EVERY DEPARTMENT. ONE FOR 2011 (VALIDATION SET) AND ONE FOR 2012 (TEST SET).
# 2011
Store_42_Dept_validation.to_csv('Department_11_2011.csv', index=False)
FileLink('Department_11_2011.csv')

# 2012
Store_42_Dept_test.to_csv('Department_11_2012.csv', index=False)
FileLink('Department_11_2012.csv')
