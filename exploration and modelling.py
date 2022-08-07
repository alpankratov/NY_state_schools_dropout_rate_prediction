#%%
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import tensorflow as tf
from keras import backend as K
import shap # 0.40 version is used
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import rankdata, spearmanr, pearsonr
from pingouin import ttest
import pickle

###############################################################################
# FINALIZING PREPARATION OF THE DATASET
###############################################################################
#%%
# loading prepared combined data before preprocessing
# all missing values are dropped, see details in data preparation.py
data_combined = pd.read_csv('data/processed/data_combined.csv',
                            index_col=['ind', 'aggregation_code', 'county_name'])\
    .dropna()\
    .drop(columns='aggregation_name')

data_combined[['cohort', 'gender']] = \
    data_combined[['cohort', 'gender']].astype('category')

#%%
# one-hot-encoding of categorical variables
continuous_predictors_name = list(data_combined.drop(columns=['cohort', 'gender', 'dropout_pct']).columns)
vars_categorical = pd.get_dummies(data_combined[['cohort', 'gender']]) # 'nrc_desc',
vars_continuous = data_combined[continuous_predictors_name]
data_preprocessed = pd.merge(vars_categorical, vars_continuous, left_index=True, right_index=True)
data_preprocessed = data_preprocessed.merge(data_combined['dropout_pct'], left_index=True, right_index=True).\
    drop(columns='gender_Male') # gender male column is dropped as variable gender is already binary (male/female)
                                # so 0 values in gender_Female variable already imply male students

#%%
# splittig the data into training, validation and test sets corresponding to 50%, 25% and 25%
# of the total dataset respectively
# Scaling is performed using Scikit-learns StandardScaler function to get all features within with mean of 0 and std of 1.
# It was particularly necessary in order to get the neural network to learn properly. For neural networks, the
# features should be measured in comparable ranges so that there is no bias towards improperly scaled columns.
train, test = train_test_split(data_preprocessed, test_size=0.5, random_state=123)
validation, test = train_test_split(test, test_size=0.5, random_state=123)


scaler = StandardScaler().fit(train[train.columns[:-1]]) # [continuous_predictors_name]
train[train.columns[:-1]] = scaler.transform(train[train.columns[:-1]])
validation[train.columns[:-1]] = scaler.transform(validation[train.columns[:-1]])
test[train.columns[:-1]] = scaler.transform(test[train.columns[:-1]])

x_train, y_train = train.drop(columns='dropout_pct'), train['dropout_pct']
x_validation, y_validation = validation.drop(columns='dropout_pct'), validation['dropout_pct']
x_test, y_test = test.drop(columns='dropout_pct'), test['dropout_pct']

#%%
# saving training, test and validation sets for GLM in R
train.to_csv("data/processed/train.csv")
validation.to_csv("data/processed/validation.csv")
test.to_csv("data/processed/test.csv")

#%%
# Load county data that is used to double check the correctness of correlation matrix
county_data = pd.read_csv('data/processed/county_data.csv').drop(columns='Unnamed: 0')


###############################################################################
# EXPLORATORY ANALYSIS
###############################################################################

#%%
# load exploratory data - not scaled training dataset
data_exloratory = data_combined.loc[train.index]

#%%
# Number of schools
len(data_exloratory.reset_index()['aggregation_code'].unique())
data_exloratory.groupby('gender').agg('count').reset_index()
data_exloratory.groupby('cohort').agg('count').reset_index()


#%%
# plot categorical variables
sns.set_style("whitegrid", {'grid.linestyle': '--'})
fig, ax = plt.subplots(2, figsize=(15, 15))
plt.suptitle('Distribution of observations \nby gender and cohort', fontsize=30)
sns.barplot(data=data_exloratory.groupby('gender').agg('count').reset_index(),
            x='gender', y='cohort', ax=ax[0])
ax[0].set_ylabel('Number of observations', fontsize=25)
ax[0].set_xlabel('Gender', fontsize=25)
sns.barplot(data=data_exloratory.groupby('cohort').agg('count').reset_index(),
            x='cohort', y='gender', ax=ax[1])
ax[1].set_ylabel('Number of observations', fontsize=25)
ax[1].set_xlabel('Cohort', fontsize=25)
plt.tight_layout()
plt.savefig('plots\exploratory\categortical_vars_plot.png')
plt.clf()

#%%
sns.set_style("whitegrid", {'grid.linestyle': '--'})
fig, ax = plt.subplots(1, 2, figsize=(20, 13))
fig.suptitle('Distribution of dropout rated by gender (left) and cohorts (right)')
sns.set(font_scale=2)
sns.histplot(data=data_exloratory, x='dropout_pct', bins=20, kde=True,
             hue='gender', palette='bright', multiple='layer', ax=ax[0])
sns.histplot(data=data_exloratory, x='dropout_pct', bins=20, kde=True,
             hue='cohort', palette='bright', multiple='layer', ax=ax[1])
plt.tight_layout
plt.savefig('plots\exploratory\dropout_rates_sitribution_by_gender_and_cohort.png')
plt.clf()


#%%
# correlation of dropout rate with other continuous variables
C_mat = pd.DataFrame(data_exloratory.corr()['dropout_pct'])
sns.set(font_scale=1.5)
fig = plt.figure(figsize=(6, 15))
sns.heatmap(C_mat, vmin=-1, vmax=1, cmap='vlag', annot=True, square=True)
plt.tight_layout()
plt.savefig('plots\exploratory\correlation_dropout.png')
plt.clf()

#%%
# continuous variables exploration
fig = plt.figure(figsize=(40, 40))
sns.set(font_scale=1.5)
sns.heatmap(data=train[continuous_predictors_name].corr(), vmin=-1, vmax=1, cmap='vlag', annot=True, square=True)
plt.yticks(rotation=0)
plt.savefig('plots\exploratory\continuous_correlation.png')
plt.clf()

#%%
# Histogram of correlation coefficient of continuous variables
train_corr = train[continuous_predictors_name].corr().\
    where(~np.tril(np.ones(train[continuous_predictors_name].corr().shape)).astype(bool))
train_corr = train_corr.unstack()
train_corr = train_corr.to_frame()
train_corr = train_corr.reset_index().rename(columns={'level_0': 'Variable 1',
                                                      'level_1': 'Variable 2',
                                                      0: 'Pearson correlation'})
train_corr = train_corr.dropna().reset_index().drop(columns=['index'])
train_corr['abs_corr'] = np.abs(train_corr['Pearson correlation'])

#%%
# Strong correlations among the continuous predictors
train_corr.query('abs_corr >= 0.8').to_csv('plots/exploratory/strong_correlations.csv', index=False)


###############################################################################
# MODEL BUILDING
###############################################################################
#%%
###############################################################################
# LOADING LASSO REGRESSION RESULTS FROM R
###############################################################################
# First creating the table where performance metrics of the models will be stored
model_validation_metrics = pd.DataFrame(columns=['Model', 'Mean absolute error', 'R2'])

# Loading predictions of lasso regression that was calculated in R
lasso_yhat_val = np.asarray(pd.read_csv('data/processed/lasso_yhat_val.csv'))

# Adding metrics for Lasso Regression in the summary table
validation_metrics_lasso = pd.DataFrame([['R1 regularized logistic regression',
    mean_absolute_error(y_validation, lasso_yhat_val),
    r2_score(y_validation, lasso_yhat_val)]],
                columns=['Model', 'Mean absolute error', 'R2'])

model_validation_metrics = model_validation_metrics.append(validation_metrics_lasso)

# Loading beta coefficients for variables in Lasso regression (for ranking)
lasso_betas = pd.read_csv('data/processed/lasso_betas.csv')
lasso_betas['variable'] = lasso_betas['variable'].replace(regex='`', value="")
lasso_betas = lasso_betas.eval('abs_value = abs(coefficient)')
lasso_betas['coeff_rank'] = lasso_betas['abs_value'].rank(method="max", ascending=False)


#%%
# keeping only variables considered important in Lasso regression
important_vars_lasso = list(lasso_betas.query('coefficient != 0')['variable'])
x_train = x_train[important_vars_lasso]
x_validation = x_validation[important_vars_lasso]
x_test = x_test[important_vars_lasso]

#%%
# check correlation of remaining continuous variables
continuous_predictors_selected = set(continuous_predictors_name).intersection(set(x_train.columns))

fig = plt.figure(figsize=(15, 15))
sns.set(font_scale=1.5)
sns.heatmap(data=train[continuous_predictors_selected].corr(), vmin=-1, vmax=1, cmap='vlag', annot=True, square=True)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots\exploratory\continuous_remaining_correlation.png')
plt.clf()

#%%
# Extracting correlation for remaining pais
train_selected_corr = train_corr[train_corr['Variable 1'].isin(continuous_predictors_selected)]
train_selected_corr = train_selected_corr[train_selected_corr['Variable 2'].isin(continuous_predictors_selected)]

# Plotting histogram of correlation among continuous predictors
# before and after variable selection
sns.set_style("whitegrid", {'grid.linestyle': '--'})
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.histplot(x='abs_corr', data=train_corr.query('abs_corr >= 0.3'), binwidth=0.1, ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_ylabel('Number of pairs of variables')
#ax[0].set_xticks(list(np.linspace(0.3, 0.9, 7)))
sns.histplot(x='abs_corr', data=train_selected_corr.query('abs_corr >= 0.3'), binwidth=0.1, ax=ax[1])
ax[1].set_xlabel('Absolute pearson correlation coefficient \n (only above 0.3)')
ax[1].set_ylabel('Number of pairs of variables')
#ax[1].set_xticks(list(np.linspace(0.3, 0.9, 7)))
plt.tight_layout()
plt.show()




#%%
###############################################################################
# RANDOM FOREST REGRESSION
###############################################################################
# Define the model with different hyperparameters and perform cross validation
param_grid = {
    'max_depth': np.linspace(10, 40, 5).astype(int),
    'max_leaf_nodes': np.linspace(500, 1500, 3).astype(int),
    'max_features': np.linspace(1, len(x_train.columns), 6).astype(int)

}

grid_random_forest = GridSearchCV(RandomForestRegressor(n_estimators=100, criterion="absolute_error", bootstrap=True,
                                            oob_score=True, random_state=123),
                                  param_grid=param_grid, refit='neg_mean_absolute_error', n_jobs=-1, verbose=2,
                                  cv=5, scoring=['r2', 'neg_mean_absolute_error'])
grid_random_forest.fit(x_train.values, y_train.values)

# NON-ESSENTIAL PART OF THE PROJECT
# Pickle the random forest model (for Flask API check)
model_file = "models/rf.pickle"
pickle.dump(grid_random_forest.best_estimator_, open(model_file, 'wb'))

#%%
# Plot change of metric across different model parameters
def grid_search_metrics_plot(model):
    rf_best_configuration_number = list(model.cv_results_['rank_test_neg_mean_absolute_error']).index(1)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('R-squared and mean absolute error (negative) metrics across different \n '
                 'combinations of random forest parameters. \n'
                 'Best configuration is #%s'% (rf_best_configuration_number))
    sns.lineplot(x=list(range(len(model.cv_results_['mean_test_neg_mean_absolute_error']))),
                 y=-model.cv_results_['mean_test_neg_mean_absolute_error'], ax=ax[0])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Mean Absolute Error')
    sns.lineplot(x=list(range(len(model.cv_results_['mean_test_r2']))),
                 y=model.cv_results_['mean_test_r2'], ax=ax[1])
    ax[1].set_xlabel('Combination of parameters')
    ax[1].set_ylabel('R-squared')
    plt.tight_layout()
    plt.show()


#%%
#Plot
grid_search_metrics_plot(grid_random_forest)

#%%
#Comment on the grid search plot
rf_param_table = pd.DataFrame(grid_random_forest.cv_results_['params'])
rf_param_table['MAE'] = -grid_random_forest.cv_results_['mean_test_neg_mean_absolute_error']
rf_param_table['R2'] = grid_random_forest.cv_results_['mean_test_r2']
rf_param_table['Index'] = list(range(len(pd.DataFrame(grid_random_forest.cv_results_['params']))))
rf_param_table.to_csv('data/output/rf_hyperparameters_tuning.csv', index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 7))
fig.suptitle('Random Forest. Distribution of MAE for each hyperparameter value')
sns.scatterplot(x='max_features', y='MAE', data=rf_param_table, ax=ax[0])
ax[0].set_xticks(np.linspace(1, len(x_train.columns), 6).astype(int))
sns.scatterplot(x='max_depth', y='MAE', data=rf_param_table, ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_xticks(np.linspace(10, 40, 5).astype(int))
sns.scatterplot(x='max_leaf_nodes', y='MAE', data=rf_param_table, ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xticks(np.linspace(500, 1500, 3).astype(int))
plt.tight_layout()
plt.show()

#%%
print(grid_random_forest.best_params_) # {'max_depth': 32, 'max_features': 6, 'max_leaf_nodes': 1000}

#%%
# calculating other score metrics on validation data
random_forest_yhat_val = grid_random_forest.best_estimator_.predict(x_validation.values)

validation_metrics_random_forest = pd.DataFrame([['Random forest',
    mean_absolute_error(y_validation, random_forest_yhat_val),
    r2_score(y_validation, random_forest_yhat_val)]],
                columns=['Model', 'Mean absolute error', 'R2'])

model_validation_metrics = model_validation_metrics.append(validation_metrics_random_forest)
#%%
###############################################################################
# EXTREME GRADIENT BOOSTING REGRESSION (XGBOOST)
###############################################################################
# Define the model with different hyperparameters and perform cross validation
params = {
    'max_depth': np.linspace(3, 9, 4).astype(int),
    'reg_lambda': np.linspace(4, 12, 5),
    'n_estimators': np.linspace(100, 500, 5).astype(int)
}

xgb_grid_search = GridSearchCV(XGBRegressor(random_state=123, objective='reg:logistic'),
                               param_grid=params, refit='neg_mean_absolute_error', verbose=2,
                               scoring=['r2', 'neg_mean_absolute_error'],
                               cv=5, n_jobs=-1)

xgb_grid_search.fit(x_train, y_train, eval_metric='mae', early_stopping_rounds=20,
                    eval_set=[(x_validation, y_validation)])

#%%
grid_search_metrics_plot(xgb_grid_search)

#%%
#Comment on the grid search plot
xgb_param_table = pd.DataFrame(xgb_grid_search.cv_results_['params'])
xgb_param_table['MAE'] = -xgb_grid_search.cv_results_['mean_test_neg_mean_absolute_error']
xgb_param_table['R2'] = xgb_grid_search.cv_results_['mean_test_r2']
xgb_param_table['Index'] = list(range(len(pd.DataFrame(xgb_grid_search.cv_results_['params']))))
xgb_param_table.to_csv('data/output/xgb_hyperparameters_tuning.csv', index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 7))
fig.suptitle('XGBoost. Distribution of MAE for each hyperparameter value')
sns.scatterplot(x='n_estimators', y='MAE', data=xgb_param_table, ax=ax[0])
ax[0].set_xticks(np.linspace(100, 500, 5).astype(int))
sns.scatterplot(x='max_depth', y='MAE', data=xgb_param_table, ax=ax[1])
ax[1].set_ylabel('')
ax[1].set_xticks(np.linspace(3, 9, 4).astype(int))
sns.scatterplot(x='reg_lambda', y='MAE', data=xgb_param_table, ax=ax[2])
ax[2].set_ylabel('')
ax[2].set_xticks(np.linspace(4, 12, 5))
plt.tight_layout()
plt.show()

#%%
xgb_grid_search.best_params_ # {'max_depth': 3, 'n_estimators': 500, 'reg_lambda': 8.0}

#%%
# calculating other score metrics on validation data
xgb_yhat_val = xgb_grid_search.best_estimator_.predict(x_validation)

validation_metrics_xgb = pd.DataFrame([['XGBoost Regression',
    mean_absolute_error(y_validation, xgb_yhat_val),
    r2_score(y_validation, xgb_yhat_val)]],
                columns=['Model', 'Mean absolute error', 'R2'])

model_validation_metrics = model_validation_metrics.append(validation_metrics_xgb)



#%%
###############################################################################
# NEURAL NETWORK BASED REGRESSION
###############################################################################
# Define the model architecture - Architecture #1 - 7 layers
model_NN_1 = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
    ]
)

#%%
# Define the model architecture - Architecture #2 - 5 layers
model_NN_2 = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
    ]
)

#%%
# Define the model architecture - Architecture #3 - 10 layers
model_NN_3 = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
    ]
)


#%%
# Creating R-squared coefficient of determination as a custom metric in the model
def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

#%%
# Creating a model checkpoint to save the instance of the training process where the model showed the best result
checkpoint_1 = tf.keras.callbacks.ModelCheckpoint("models/NN_best_model_1.h5", monitor='val_loss',
                                                verbose=0, save_best_only=True,
                                                mode="min", save_freq='epoch')
nn_early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=200)

checkpoint_2 = tf.keras.callbacks.ModelCheckpoint("models/NN_best_model_2.h5", monitor='val_loss',
                                                verbose=0, save_best_only=True,
                                                mode="min", save_freq='epoch')
nn_early_stopping_2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=300)

checkpoint_3 = tf.keras.callbacks.ModelCheckpoint("models/NN_best_model_3.h5", monitor='val_loss',
                                                verbose=0, save_best_only=True,
                                                mode="min", save_freq='epoch')
nn_early_stopping_3 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=40)


#%%
# Compiling and fitting the model #1
model_NN_1.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.mean_absolute_error,
                 metrics=coeff_determination)

model_NN_best_fit_1 = model_NN_1.fit(x_train, y_train, epochs=2000, batch_size=int(x_train.shape[0]/5)+1,
                                 validation_data=(x_validation, y_validation),
                                 callbacks=[checkpoint_1, nn_early_stopping_1], verbose=1)

#%%
# Compiling and fitting the model #2
model_NN_2.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.mean_absolute_error,
                 metrics=coeff_determination)

model_NN_best_fit_2 = model_NN_2.fit(x_train, y_train, epochs=3000, batch_size=int(x_train.shape[0]/5)+1,
                                 validation_data=(x_validation, y_validation),
                                 callbacks=[checkpoint_2, nn_early_stopping_2], verbose=1)

#%%
# Compiling and fitting the model #3
model_NN_3.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.mean_absolute_error,
                 metrics=coeff_determination)

model_NN_best_fit_3 = model_NN_3.fit(x_train, y_train, epochs=400, batch_size=int(x_train.shape[0]/5)+1,
                                 validation_data=(x_validation, y_validation),
                                 callbacks=[checkpoint_3, nn_early_stopping_3], verbose=1)


#%%
# Plot change in model metrics across training epochs
for model_plot in ['model_NN_best_fit_1', 'model_NN_best_fit_2', 'model_NN_best_fit_3']:
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f'Performance of the NN-Based regresssion')
    sns.lineplot(x=eval(model_plot).epoch, y=eval(model_plot).history['coeff_determination'],
                 label='Training R2', ax=ax[0])
    sns.lineplot(x=eval(model_plot).epoch, y=eval(model_plot).history['val_coeff_determination'],
                 label='Validation R2', ax=ax[0])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim([0, 0.8])
    sns.lineplot(x=eval(model_plot).epoch, y=eval(model_plot).history['loss'],
                 label='Training MAE (loss)', ax=ax[1])
    sns.lineplot(x=eval(model_plot).epoch, y=eval(model_plot).history['val_loss'],
                 label='Validation MAE (loss)', ax=ax[1])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylim([0.03, 0.05])
    plt.savefig(f'plots/ANN-based regression/{model_plot} plot.png')
    plt.clf()

#%%

# Loading saved model that showed the best validation loss
model_NN_best_selected = tf.keras.models.load_model('models/NN_best_model_2.h5', compile=False)



#%%
# calculating other score metrics on validation data
nn_yhat_val = model_NN_best_selected.predict(x_validation)

validation_metrics_nn = pd.DataFrame([['NN-based regression',
    mean_absolute_error(y_validation, nn_yhat_val),
    r2_score(y_validation, nn_yhat_val)]],
                columns=['Model', 'Mean absolute error', 'R2'])

model_validation_metrics = model_validation_metrics.append(validation_metrics_nn)

######################################################################
# Comparing performance of all models
######################################################################

#%%
# Comparing performance of all models
print(model_validation_metrics)
model_validation_metrics.to_csv('data/output/model_validation_metrics.csv')

#%%
# Showing predictions for N random observations
predictions_number = 25
rng = np.random.default_rng(123)
predictions = rng.choice(y_validation.shape[0], size=predictions_number, replace=False)
plot_predictions = pd.DataFrame({'Model': ['Observed'] * predictions_number,
                                 'Values': np.array(y_validation.iloc[predictions]),
                                 'Observed': np.array(y_validation.iloc[predictions])})\
    .append(pd.DataFrame({'Model': ['L1 Logistic Regression'] * predictions_number,
                          'Values': lasso_yhat_val[predictions].flatten(),
                          'Observed': np.array(y_validation.iloc[predictions])}))\
    .append(pd.DataFrame({'Model': ['Random Forrest'] * predictions_number,
                          'Values': random_forest_yhat_val[predictions].flatten(),
                          'Observed': np.array(y_validation.iloc[predictions])}))\
    .append(pd.DataFrame({'Model': ['XGBoost'] * predictions_number,
                          'Values': xgb_yhat_val[predictions].flatten(),
                          'Observed': np.array(y_validation.iloc[predictions])}))\
    .append(pd.DataFrame({'Model': ['NN-based Regression'] * predictions_number,
                          'Values': nn_yhat_val[predictions].flatten(),
                          'Observed': np.array(y_validation.iloc[predictions])}))\
    .reset_index().rename(columns={'index': 'Prediction #'})


sizes = dict({'Observed': 250, 'L1 Logistic Regression': 120, 'XGBoost': 120,
             'NN-based Regression': 120, 'Random Forrest': 120})

sns.set_style("whitegrid", {'grid.linestyle': '--'})
plt.figure(figsize=(10, 10))
plt.suptitle(f'Comparison of {predictions_number} of random prediction by different models')
sns.scatterplot(x='Observed', y='Values', hue='Model', size='Model', sizes=sizes, style='Model',
                size_order=['Observed'], data=plot_predictions, alpha=0.7, x_jitter=30)
#plt.xticks(ticks=list(range(predictions_number)), labels=list(range(1, predictions_number + 1)))
plt.legend(loc='best', borderaxespad=0, prop={'size': 14})
plt.ylabel('Predicted dropout rates')
plt.xlabel('Observed dropout rates')
plt.tight_layout()
plt.savefig('plots/random_predictions.png')
plt.show()


#%%
# Calculation of prediction precision:
# preparation of table with R2 and MAE for each model

# to load already prepared result
#bootstrap_results = pd.read_csv('data/output/bootstrap_results.csv')

bootstrap_results = pd.DataFrame(columns=['RF_R2', 'RF_MAE', 'XBG_R2', 'XBG_MAE', 'NN_R2', 'NN_MAE'])

# creating a list of random starts that will be used for bootstrapping
rng = np.random.default_rng(123)
random_starts = rng.choice(100000, size=5000, replace=False)

# taking the validation MAE and R squared for each validation by bootstrapping validation population
for i in range(random_starts.shape[0]):
    rng = np.random.default_rng(random_starts[i])
    indices = rng.choice(x_validation.shape[0], size=300, replace=False)
    x_validation_sample = x_validation.iloc[indices]
    y_validation_sample = y_validation.iloc[indices]

    if (i + 1) % 50 == 0:
        print(f'{i + 1}th iteration')

    # Random forest
    rf_predictions_sample = grid_random_forest.best_estimator_.predict(x_validation.iloc[indices].values)
    rf_r2 = r2_score(y_validation_sample, rf_predictions_sample)
    rf_mae = mean_absolute_error(y_validation_sample, rf_predictions_sample)

    # XGBoosting
    xgb_predictions_sample = xgb_grid_search.best_estimator_.predict(x_validation.iloc[indices])
    xgb_r2 = r2_score(y_validation_sample, xgb_predictions_sample)
    xgb_mae = mean_absolute_error(y_validation_sample, xgb_predictions_sample)

    # ANN-based regression
    nn_predictions_sample = model_NN_best_selected.predict(x_validation.iloc[indices])
    nn_r2 = r2_score(y_validation_sample, nn_predictions_sample)
    nn_mae = mean_absolute_error(y_validation_sample, nn_predictions_sample)

    bootstrap_results = bootstrap_results.append(pd.DataFrame({'RF_R2': [rf_r2], 'RF_MAE': [rf_mae],
                                                               'XBG_R2': [xgb_r2], 'XBG_MAE': [xgb_mae],
                                                               'NN_R2': [nn_r2], 'NN_MAE': [nn_mae]}))




#%%
# Loading GLM bootstrap results and adding to the table with other models
GLM_bootstrap_results = pd.read_csv('data/output/GLM_bootstrap_results.csv')
bootstrap_results['GLM_R2'] = np.array(GLM_bootstrap_results['GLM_R2'])
bootstrap_results['GLM_MAE'] = np.array(GLM_bootstrap_results['GLM_MAE'])
bootstrap_results.to_csv('data/output/bootstrap_results.csv', index=False)



#%%
# Estimating standard deviation of performance metrics for each model
bootstrap_results.std()
bootstrap_results.mean()


######################################################################
# Interpretation of the models
######################################################################
#%%
# Lasso regression
lasso_betas_top10 = lasso_betas.query('coeff_rank <= 30').sort_values('coeff_rank', ascending=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
#fig.subplots_adjust(left=2)
fig, ax = plt.subplots(1, 2)
fig.suptitle('Predictors in L1 regularized logistic regression \n sorted by its importance in descending order ')
sns.barplot(x='coefficient', y='variable', data=lasso_betas_top10, ci=None, ax=ax[0], color='#008BFB')
ax[0].set_xlabel('Coefficient value')
ax[0].set_ylabel('Variable name')
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
sns.barplot(x='abs_value', y='variable', data=lasso_betas_top10, ci=None, ax=ax[1], color='#008BFB')
ax[1].set_xlabel('Coefficient absolute value')
ax[1].set_ylabel('')
ax[1].set_yticklabels([])
fig.set_figheight(10)
fig.set_figwidth(16)
plt.tight_layout()
plt.savefig('plots/L1 logistic regression feature importance.png')
fig.show()



#%%
# BEFORE RUNNING SHAP PLEASE NOTE THAT IT HAS THE FOLLOWING ISSUE WITH PERMUTATION EXPLORER
# AttributeError: 'Explanation' object has no attribute '_old_format'
# CORRECTING THIS ISSUE CURRENTLY REQUIRES MANUAL MODIFICATION OF ONE LINE
# IN THE SHAP PACKAGE FILE "shap/explainers/_permutation.py"
# SEE DETAILS IN HERE:
# https://github.com/slundberg/shap/commit/f4e56e88d481111cab571012974fd38f5c81e53b
# https://github.com/slundberg/shap/pull/2369
# The issue was corrected on March 22, 2022, however new version of the package has not been released yet.


# SHAP values -Random forest based regression
# the explainer sends data as numpy array which has no column names, so we fix it
# https://gist.github.com/noleto/05dfa4a691ebbc8816c035b86d2d00d4
def rf_predict(data_asarray):
    data_asframe=pd.DataFrame(data_asarray, columns=list(x_train.columns))
    return grid_random_forest.best_estimator_.predict(data_asframe.values)

rf_explainer = shap.explainers.Permutation(rf_predict, x_train.values)
rf_shap_values_full = rf_explainer.shap_values(x_validation)

#%%
# Check additivity of base value and shap values for a single prediction
np.round(rf_shap_values_full.base_values[1]+rf_shap_values_full.values[1].sum(), 5)
np.round(grid_random_forest.best_estimator_.predict(x_validation.values)[1], 5)

#%%
# SHAP values - XGB regression
# the explainer sends data as numpy array which has no column names, so we fix it
# https://gist.github.com/noleto/05dfa4a691ebbc8816c035b86d2d00d4
def xgb_predict(data_asarray):
    data_asframe=pd.DataFrame(data_asarray, columns=list(x_train.columns))
    return xgb_grid_search.best_estimator_.predict(data_asframe)

xgb_explainer = shap.explainers.Permutation(xgb_predict, x_train)
xgb_shap_values_full = xgb_explainer.shap_values(x_validation)


# Check additivity of base value and shap values for a single prediction
np.round(xgb_shap_values_full.base_values[1]+xgb_shap_values_full.values[1].sum(), 5)
np.round(xgb_grid_search.best_estimator_.predict(x_validation)[1], 5)


#%%
nn_explainer = shap.explainers.Permutation(model_NN_best_selected.predict, x_train)
nn_shap_values_full = nn_explainer.shap_values(x_validation)

# Check additivity of base value and shap values for a single prediction
np.round(xgb_shap_values_full.base_values[1]+xgb_shap_values_full.values[1].sum(), 5)
np.round(xgb_grid_search.best_estimator_.predict(x_validation)[1], 5)


#%%
# Extracting only shap values
rf_shap_values = rf_shap_values_full.values
xgb_shap_values = xgb_shap_values_full.values
nn_shap_values = nn_shap_values_full.values

#%%
# save shap values
#np.save('data/shap values/xgb_shap_values.npy', xgb_shap_values)
#np.save('data/shap values/rf_shap_values.npy', rf_shap_values)
#np.save('data/shap values/nn_shap_values.npy', nn_shap_values)

#%%
# load shap values
xgb_shap_values = np.load('data/shap values/xgb_shap_values.npy')
rf_shap_values = np.load('data/shap values/rf_shap_values.npy')
nn_shap_values = np.load('data/shap values/nn_shap_values.npy')

#%%
def plot_shap(shap_values, x_validation, plot_type):
    shap.summary_plot(shap_values, x_validation, plot_type=plot_type, show=False)
    fig = plt.gcf() # gcf means "get current figure"
    fig.set_figwidth(12)
    plt.tight_layout()

#%%
# save all shap plots to files
for shap_values in ['nn_shap_values', 'rf_shap_values', 'xgb_shap_values']:
    for plot_type in ['dot', 'bar']:
        plot_shap(eval(shap_values), x_validation, plot_type=plot_type)
        plt.savefig(f'plots/shap values/{shap_values} {plot_type} plot.png')
        plt.clf()


#%%
# Ranking of all features' importance
feature_rank_table = pd.DataFrame()
feature_rank_table['Feature name'] = list(x_train.columns)
feature_rank_table['Regularized_logistic_regression'] = np.array(lasso_betas.query('coefficient !=0')['coeff_rank']).astype(int)
feature_rank_table['Random_forest_regression'] = rankdata(-np.abs(rf_shap_values).mean(0), method='max')
feature_rank_table['XGBoost_regression'] = rankdata(-np.abs(xgb_shap_values).mean(0), method='max')
feature_rank_table['NN_regression'] = rankdata(-np.abs(nn_shap_values).mean(0), method='max')
feature_rank_table = feature_rank_table.set_index('Feature name')
feature_rank_table = feature_rank_table.sort_values('XGBoost_regression', ascending=True)
feature_rank_table.to_csv('data/output/features_rank_table.csv')

#%%
# Calculating Spearman correlation of variable ranks in betwean each pair of models model
feature_rank_correlation = pd.DataFrame(spearmanr(feature_rank_table)[0], columns=feature_rank_table.columns).\
    set_index(feature_rank_table.columns)
feature_rank_correlation.to_csv('data/output/features_rank_correlation.csv')

#%%
# Comparison of feature importance and SHAP values for XGB and Random Forest
# Calculation of permutation feature importances
rf_feature_importance = permutation_importance(grid_random_forest.best_estimator_, x_validation.values,
                                               y_validation.values, n_repeats=100, random_state=123)

xgb_feature_importance = permutation_importance(xgb_grid_search.best_estimator_, x_validation,
                                               y_validation, n_repeats=100, random_state=123)

# Adding all in the table
feature_importance_vs_SHAP = pd.DataFrame()
feature_importance_vs_SHAP['Feature name'] = list(x_train.columns)
feature_importance_vs_SHAP['RF feature importance (MDI)'] = grid_random_forest.best_estimator_.feature_importances_
feature_importance_vs_SHAP['RF feature importance (permutation)'] = rf_feature_importance.importances_mean
feature_importance_vs_SHAP['RF feature importance (SHAP)'] = \
    np.abs(rf_shap_values).mean(0)/np.abs(rf_shap_values).mean(0).sum()
feature_importance_vs_SHAP['XGB feature importance (MDI)'] = \
    xgb_grid_search.best_estimator_.feature_importances_/rf_feature_importance.importances_mean.sum()
feature_importance_vs_SHAP['XGB feature importance (permutation)'] = \
    xgb_feature_importance.importances_mean/xgb_feature_importance.importances_mean.sum()
feature_importance_vs_SHAP['XGB feature importance (SHAP)'] = \
    np.abs(xgb_shap_values).mean(0)/np.abs(xgb_shap_values).mean(0).sum()
feature_importance_vs_SHAP = feature_importance_vs_SHAP.set_index('Feature name')

# extracting top 5 features of XGBoost (SHAP method)
feature_importance_vs_SHAP_top_5 = feature_importance_vs_SHAP.sort_values(by='XGB feature importance (SHAP)',
                                                                          ascending=False).iloc[0:4,:]
correlations = []
for importance_1 in feature_importance_vs_SHAP_top_5.columns:
    for importance_2 in feature_importance_vs_SHAP_top_5.columns:
        correlations.append(pearsonr(feature_importance_vs_SHAP_top_5[importance_1],
                                     feature_importance_vs_SHAP_top_5[importance_2])[0])

correlations = pd.DataFrame(np.array(correlations).reshape((6, 6)), columns=feature_importance_vs_SHAP.columns,
                            index=feature_importance_vs_SHAP.columns)

correlations.to_csv('data/output/feature importance pearson correlations.csv')

#%%
# Checking performance of ensemble predictions
# Calculating weights of each model in ensemble (based on training data)
training_predictions = pd.DataFrame()
training_predictions['XGB'] = xgb_grid_search.best_estimator_.predict(x_validation)
training_predictions['RF'] = grid_random_forest.best_estimator_.predict(x_validation.values)
training_predictions['NN'] = model_NN_best_selected.predict(x_validation.values)
training_predictions['GLM'] = lasso_yhat_val


evaluation_results = pd.DataFrame(columns=['XGB_Weight', 'RF_Weight', 'NN_Weight', 'GLM_Weight', 'MAE', 'R2'])

for xgb in np.linspace(0, 1, 21):
    for rf in np.linspace(0, 1, 21):
        for nn in np.linspace(0, 1, 21):
            for glm in np.linspace(0, 1, 21):
                if rf + xgb + nn + glm != 1:
                    pass
                else:
                    predicted = rf * training_predictions['RF'] + xgb * training_predictions['XGB'] \
                                + nn * training_predictions['NN'] + glm * training_predictions['GLM']
                    mae = mean_absolute_error(y_validation.values, predicted)
                    r2 = r2_score(y_validation.values, predicted)
                    evaluation_results = evaluation_results.append(pd.DataFrame({
                        'GLM_Weight': [glm],
                        'RF_Weight': [rf],
                        'XGB_Weight': [xgb],
                        'NN_Weight': [nn],
                        'MAE': [mae],
                        'R2': [r2]
                    }))

max_R2_overall = np.max(evaluation_results.query('GLM_Weight != 0 and NN_Weight != 0 and '
                                                 'RF_Weight != 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_RF_NN = np.max(evaluation_results.query('GLM_Weight == 0 and NN_Weight != 0 and '
                                                   'RF_Weight != 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_GLM_NN = np.max(evaluation_results.query('GLM_Weight != 0 and NN_Weight != 0 and '
                                                    'RF_Weight == 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_GLM_RF = np.max(evaluation_results.query('GLM_Weight != 0 and NN_Weight == 0 and '
                                                    'RF_Weight != 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_RF = np.max(evaluation_results.query('GLM_Weight == 0 and NN_Weight == 0 and '
                                                'RF_Weight != 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_NN = np.max(evaluation_results.query('GLM_Weight == 0 and NN_Weight != 0 and '
                                                'RF_Weight == 0 and XGB_Weight !=0')['R2'])
max_R2_XGB_GLM = np.max(evaluation_results.query('GLM_Weight != 0 and NN_Weight == 0 and '
                                                 'RF_Weight == 0 and XGB_Weight !=0')['R2'])
max_R2_XGB = np.max(evaluation_results.query('GLM_Weight == 0 and NN_Weight == 0 and RF_Weight == 0')['R2'])



best_combinations = evaluation_results.query('R2 == @max_R2_overall or R2 == @max_R2_XGB_RF_NN or '
                                             'R2 == @max_R2_XGB_GLM_NN or R2 == @max_R2_XGB_GLM_RF or '
                                             'R2 == @max_R2_XGB_RF or R2 == @max_R2_XGB_NN or R2 == @max_R2_XGB_GLM or '
                                             'R2 == @max_R2_XGB').reset_index().drop(columns='index')

best_combinations['Combination'] = ['XGB+RF+NN+GLM', 'XGB+RF+NN', 'XGB+RF', 'XGB+RF+GLM',
                                    'XGB+NN+GLM', 'XGB+NN', 'XGB+GLM', 'XGB']

best_combinations = best_combinations.set_index('Combination')
best_combinations.to_csv('data/output/weighted_average_ensemble_performance.csv')


#%%
# Calculation of whether the improvement is statistically significant
# Perform dependent t-test for paired samples

# Calculate R-squared for XGB and combination of XGB and RF
# for 5000 random samples (with replacement) from validation population
t_test_bootstrap_results = pd.DataFrame(columns=['XGB_R2', 'XGB_RF_R2'])

# creating a list of random starts that will be used for bootstrapping
rng = np.random.default_rng(123)
random_starts = rng.choice(100000, size=5000, replace=False)

# taking the validation MAE and R squared for each validation by bootstrapping validation population
for i in range(random_starts.shape[0]):
    rng = np.random.default_rng(random_starts[i])
    indices = rng.choice(x_validation.shape[0], size=300, replace=False)
    x_validation_sample = x_validation.iloc[indices]
    y_validation_sample = y_validation.iloc[indices]

    if (i + 1) % 50 == 0:
        print(f'{i + 1}th iteration')

    # XGB
    xgb_predictions_sample = xgb_grid_search.best_estimator_.predict(x_validation.iloc[indices])
    xgb_r2 = r2_score(y_validation_sample, xgb_predictions_sample)

    # XGB+RF
    rf_predictions_sample = grid_random_forest.best_estimator_.predict(x_validation.iloc[indices].values)
    xgb_rf_r2 = r2_score(y_validation_sample, 0.6 * xgb_predictions_sample + 0.4 * rf_predictions_sample)

    t_test_bootstrap_results = t_test_bootstrap_results\
        .append(pd.DataFrame({'XGB_R2': [xgb_r2], 'XGB_RF_R2': [xgb_rf_r2]}))


# Perform paired t-test and check Check p-vaue and confidence intervals
t_test_results = ttest(x=t_test_bootstrap_results['XGB_RF_R2'], y=t_test_bootstrap_results['XGB_R2'], paired=True)
# https://www.marsja.se/how-to-use-python-to-perform-a-paired-sample-t-test/
t_test_results

#%%
# Performance of final model (XGBoost) on test data
xgb_yhat_test = xgb_grid_search.best_estimator_.predict(x_test)

test_metrics_xgb = pd.DataFrame([['XGBoost Regression',
    mean_absolute_error(y_test, xgb_yhat_test),
    r2_score(y_test, xgb_yhat_test)]],
                columns=['Model', 'Mean absolute error', 'R2'])

test_metrics_xgb.to_csv('data/output/test_performance.csv', index=False)