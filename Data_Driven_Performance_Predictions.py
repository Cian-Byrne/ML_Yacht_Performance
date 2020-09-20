from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from Data_Clean_Functions import *
import jedi

import seaborn as sns
sns.set()


def print_scores(y_test, y_pred):

    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MAE = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    maximum_error = max_error(y_test, y_pred)

    print("RMSE: %.3f" % (RMSE))
    print("MAE: %.3f" % (MAE))
    print("R^2: %.3f" % (r2))
    print("Maximum Error: %.3f\n" % (maximum_error))


# set random seed for reproducability
SEED = 10
np.random.seed(SEED)

# Import data frame with the VPP predictions for use in PBLM/SciML models.
df = pd.read_csv("Labeled_all_VPP_waves100.csv")

df.label.unique()


UW_df = df[df['label'] == 'UW']
DW_df = df[df['label'] == 'DW']

# remove values that are have non-sensical Twa measurements.
UW_df = UW_df[UW_df["abs_Twa"] > 25]
UW_df = UW_df[UW_df["Bsp"] > 5.5]
DW_df = DW_df[DW_df["Bsp"] > 6.5]


steady_df = pd.concat([UW_df, DW_df])


# Standardise relevent features
features = ['abs_Twa', 'Trim', 'Forestay', 'Tws', 'abs_Rudder',
            'Trim_ampl', 'Trim_freq', 'Heel_freq', 'Heel_ampl', 'VPP_Bsp', 'VPP_Heel', 'VPP_Leeway', 'abs_Heel']

steady_df = Scale_df(steady_df, features, 'standard')

# select more advanced features.
# ,'Forestay_scaled', 'Trim_scaled', 'abs_Rudder_scaled', 'VPP_Bsp_scaled', 'VPP_Heel_scaled', 'VPP_Leeway_scaled' ]
use_features = ['abs_Twa_scaled', 'Tws_scaled']


Train_data = np.array(steady_df[use_features])
labels = np.transpose(np.array([steady_df['Bsp'], steady_df['abs_Heel']]))

# Split the data into training and testing data sets.
X_train, X_test, y_train, y_test = train_test_split(
    Train_data, labels, test_size=0.3, random_state=0)

# LinearRegression Model for baseline
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# MLP model (Hyperparams not tuned for these feats)
mlp = MLPRegressor(max_iter=1500, activation='tanh', hidden_layer_sizes=(
    50, 50, 50), learning_rate='adaptive', solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# RF model (Hyperparams not tuned for these feats)
rf = RandomForestRegressor(max_features='auto', min_samples_split=2, n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# SVM model (hyperparams not tuned for these feats)
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[0, 1])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)

# Pred heel first then feed prediction forward in the chain.
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[1, 0])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# select more advanced features.
use_features = ['abs_Twa_scaled', 'Tws_scaled', 'Forestay_scaled', 'Trim_scaled',
                'abs_Rudder_scaled']  # , 'VPP_Bsp_scaled', 'VPP_Heel_scaled', 'VPP_Leeway_scaled' ]


Train_data = np.array(steady_df[use_features])
labels = np.transpose(np.array([steady_df['Bsp'], steady_df['abs_Heel']]))

# Split the data into training and testing data sets.
X_train, X_test, y_train, y_test = train_test_split(
    Train_data, labels, test_size=0.3, random_state=0)

# LinearRegression Model for baseline
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# MLP model (Hyperparams not tuned for these feats)
mlp = MLPRegressor(max_iter=1500, activation='tanh', hidden_layer_sizes=(
    50, 50, 50), learning_rate='adaptive', solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# RF model (Hyperparams not tuned for these feats)
rf = RandomForestRegressor(max_features='auto', min_samples_split=2, n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# SVM model (hyperparams not tuned for these feats)
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[0, 1])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)

# Pred heel first then feed prediction forward in the chain.
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[1, 0])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# select more advanced features.
use_features = ['abs_Twa_scaled', 'Tws_scaled', 'Forestay_scaled', 'Trim_scaled',
                'abs_Rudder_scaled', 'VPP_Bsp_scaled', 'VPP_Heel_scaled', 'VPP_Leeway_scaled']


Train_data = np.array(steady_df[use_features])
labels = np.transpose(np.array([steady_df['Bsp'], steady_df['abs_Heel']]))

# Split the data into training and testing data sets.
X_train, X_test, y_train, y_test = train_test_split(
    Train_data, labels, test_size=0.3, random_state=0)

# LinearRegression Model for baseline
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# MLP model (Hyperparams not tuned for these feats)
mlp = MLPRegressor(max_iter=1500, activation='tanh', hidden_layer_sizes=(
    50, 50, 50), learning_rate='adaptive', solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# RF model (Hyperparams not tuned for these feats)
rf = RandomForestRegressor(max_features='auto', min_samples_split=2, n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# SVM model (hyperparams not tuned for these feats)
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[0, 1])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)

# Pred heel first then feed prediction forward in the chain.
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[1, 0])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# select more advanced features.
#use_features = ['abs_Twa_scaled', 'Tws_scaled','Forestay_scaled', 'Trim_scaled', 'abs_Rudder_scaled', 'Trim_ampl_scaled', 'Trim_freq_scaled', 'Heel_ampl_scaled', 'Heel_freq_scaled' ]
use_features = ['abs_Twa', 'Tws', 'Forestay', 'Trim', 'abs_Rudder',
                'Trim_ampl', 'Trim_freq', 'Heel_ampl', 'Heel_freq']


Train_data = np.array(steady_df[use_features])
labels = np.transpose(np.array([steady_df['Bsp'], steady_df['abs_Heel']]))

# Split the data into training and testing data sets.
X_train, X_test, y_train, y_test = train_test_split(
    Train_data, labels, test_size=0.3, random_state=0)

# LinearRegression Model for baseline
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# MLP model (Hyperparams not tuned for these feats)
mlp = MLPRegressor(max_iter=1500, activation='tanh', hidden_layer_sizes=(
    50, 50, 50), learning_rate='adaptive', solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# RF model (Hyperparams not tuned for these feats)
rf = RandomForestRegressor(max_features='auto', min_samples_split=2, n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# SVM model (hyperparams not tuned for these feats)
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[0, 1])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)

# Pred heel first then feed prediction forward in the chain.
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[1, 0])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# select more advanced features.
use_features = ['abs_Twa_scaled', 'Tws_scaled', 'Forestay_scaled', 'Trim_scaled', 'abs_Rudder_scaled', 'VPP_Bsp_scaled',
                'VPP_Heel_scaled', 'VPP_Leeway_scaled', 'Trim_ampl_scaled', 'Trim_freq_scaled', 'Heel_ampl_scaled', 'Heel_freq_scaled']


Train_data = np.array(steady_df[use_features])
labels = np.transpose(np.array([steady_df['Bsp'], steady_df['abs_Heel']]))

# Split the data into training and testing data sets.
X_train, X_test, y_train, y_test = train_test_split(
    Train_data, labels, test_size=0.3, random_state=0)

# LinearRegression Model for baseline
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# MLP model (Hyperparams not tuned for these feats)
mlp = MLPRegressor(max_iter=1500, activation='tanh', hidden_layer_sizes=(
    50, 50, 50), learning_rate='adaptive', solver='adam')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# RF model (Hyperparams not tuned for these feats)
rf = RandomForestRegressor(max_features='auto', min_samples_split=2, n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# SVM model (hyperparams not tuned for these feats)
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[0, 1])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)

Bsp_pred = y_pred[:, 0]
Heel_pred = y_pred[:, 1]
Bsp_test = y_test[:, 0]
Heel_test = y_test[:, 1]

print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)

# Pred heel first then feed prediction forward in the chain.
svm = SVR(C=1.5, kernel='rbf')
wrapper = RegressorChain(svm, order=[1, 0])
wrapper.fit(X_train, y_train)
y_pred = wrapper.predict(X_test)


print_scores(Bsp_test, Bsp_pred)
print_scores(Heel_test, Heel_pred)


# Compare the VPP data to true readings
y_pred = steady_df.VPP_Bsp
y_test = steady_df.Bsp

print_scores(y_test, y_pred)


# Compare the VPP data to true readings
y_pred = steady_df.VPP_Heel
y_test = steady_df.abs_Heel

print_scores(y_test, y_pred)


# Compare the VPP data to true readings
y_pred = steady_df.VPP_Bsp
y_test = steady_df.Bsp

print_scores(y_test, y_pred)
