from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.optimize import curve_fit
from sklearn.base import RegressorMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import Data_Clean_Functions
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd  # pandas for data handling
import os
import warnings
from sklearn.ensemble import RandomForestRegressor
%matplotlib qt
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

funcs = Data_Clean_Functions.Funcs()


def low_pass(sr, dt=10, Hz=100):
    'Apply lowpass filter to a data series'
    win = signal.hann(int(Hz*dt)+1)
    pad = np.zeros(len(win)//2) * np.NaN
    sig = signal.convolve(sr, win, mode='valid') / sum(win)
    return pd.Series(np.r_[pad, sig, pad])


# def read_power(folder):
#     df = pd.read_csv(folder+'/fort.9', delim_whitespace=True,
#                      names=["time", "CFL", "power", "phi"])
#     df.drop(df.index[:3], inplace=True)
#     df['power_smooth'] = low_pass(df.power, dt=0.03, Hz=len(df.power))
#     df.Theta = 0.14
#     return df

#
# def average(df, var):
#     return np.trapz(df[var], df.time)/(df.time.iloc[-1]-df.time.iloc[0])
#
#
# plt.figure(figsize=(6.3, 3), dpi=100)
# plt.xlabel(r'$z/L$', fontsize=12)
# plt.ylabel(r'$C_P$', rotation=0, labelpad=15, fontsize=12)
#
# for i in range(4):
#     folder = 'phase/case{:02d}'.format(i)
#     df = read_power(folder)
#     plt.plot(df.time, df.power_smooth/df.Theta**3, label=str(i)+'/8')
#
# bar, amp = average(df, 'power')/df.Theta**3, 1.5748*df.power.mad()/df.Theta**3
# plt.axhline(y=bar, color='grey', ls='--')
# plt.axhline(y=bar+amp, color='grey', ls='-.')
# plt.axhline(y=bar-amp, color='grey', ls='-.')
#
# plt.legend(title=r'$t/T$')
# plt.tight_layout()
# plt.savefig('phase_Cp.png')
# plt.show()
# # %% codecell
# df = pd.read_csv('grid_out.csv').query('r<1')
# df = pd.concat([df, pd.read_csv('grid_out_2.csv')])
#
# fig, axa = plt.subplots(1, 2, figsize=(7, 3), dpi=100, sharey=True)
# for ax in axa:
#     ax.set_yscale('log')
#     ax.set_ylim([1e-2, 1])
# axa[0].set_ylabel(r'$\overline{C_P}$', rotation=0, labelpad=15, fontsize=12)
# axa[0].scatter(df.r, df.bar_p/df.Theta**3, c=180/np.pi*df.Theta)
# axa[0].set_xlabel(r'$r/R$', fontsize=12)
# im = axa[1].scatter(180/np.pi*df.alpha, df.bar_p/df.Theta**3, c=180/np.pi*df.Theta)
# axa[1].set_xlabel(r'$\alpha$ (deg)', fontsize=12)
# fig.colorbar(im, label=r'$\Theta$ (deg)')
# plt.tight_layout()
# plt.savefig('grid_bar_Cp.png')
#
# fig, axa = plt.subplots(1, 2, figsize=(7, 3), dpi=100, sharey=True)
# for ax in axa:
#     ax.set_yscale('log')
#     ax.set_ylim([1e-3, 1])
# axa[0].set_ylabel(r'$\Theta |C_P|$', rotation=0, labelpad=15, fontsize=12)
# axa[0].scatter(df.r, df.amp_p/df.Theta**2, c=180/np.pi*df.Theta)
# axa[0].set_xlabel(r'$r/R$', fontsize=12)
# im = axa[1].scatter(180/np.pi*df.alpha, df.amp_p/df.Theta**2, c=180/np.pi*df.Theta)
# axa[1].set_xlabel(r'$\alpha$ (deg)', fontsize=12)
# fig.colorbar(im, label=r'$\Theta$ (deg)')
# plt.tight_layout()
# plt.savefig('grid_amp_Cp.png')
#
# fig = plt.figure(figsize=(8, 3), dpi=100)
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(df.r, 180/np.pi*df.alpha, df.bar_p/df.Theta**3, c=180/np.pi*df.Theta)
# ax.set_xlabel(r'$r/R$', fontsize=12)
# ax.set_ylabel(r'$\alpha$ (deg)', fontsize=12)
# ax.set_zlabel(r'$\overline{C_P}$', fontsize=12)
#
# ax = fig.add_subplot(122, projection='3d')
# ax.scatter(df.r, 180/np.pi*df.alpha, df.amp_p/df.Theta**2, c=180/np.pi*df.Theta)
# ax.set_xlabel(r'$r/R$', fontsize=12)
# ax.set_ylabel(r'$\alpha$ (deg)', fontsize=12)
# ax.set_zlabel(r'$\Theta |C_P|$', fontsize=12)
# plt.tight_layout()
# plt.savefig('grid_3D_Cp.png')
# plt.show()

strCWD = os.getcwd()

# ------------------------------------------------------------------------------
# Read in data

df = pd.read_csv(os.path.join(strCWD, 'Data', 'Jul28_Labeled.csv'))
df2 = pd.read_csv(os.path.join(strCWD, 'Data', 'Jul29_Labeled.csv'))
df3 = pd.read_csv(os.path.join(strCWD, 'Data', 'Jul30_Labeled.csv'))

df = df.append(df2)
df = df.append(df3)

df['Utc'] = df['Utc'].astype(str)
df['Utc'] = pd.to_datetime(df['Utc'])
df['Day'] = df.Utc.dt.day

df = df.reset_index().drop('index', axis=1)
index = list(df[(df['Segment'] == 9) & (df['Day'] == 30)].index)
df.loc[index, 'label'] = 'NR'
index = list(df[(df['Segment'] == 41) & (df['Day'] == 30)].index)
df.loc[index, 'label'] = 'NR'
index = list(df[(df['Segment'] == 5) & (df['Day'] == 28)].index)
df.loc[index, 'label'] = 'NR'


df['abs_Twa'] = abs(df.Twa)
df = df.reset_index().drop('index', axis=1)
df = funcs.manoeuver_label(df, man_len=50)
df.label.unique()

df = df[(df['label'] == 'UW') | (df['label'] == 'DW')]


# need to eliminate angles  and wind speeds that the VPP do not capture
df = df[(df['Twa'] > -150) & (df['Twa'] < 150)]
df = df[(df['Tws'] < 20)]
df = df[df['Bsp'] > 5]
df = df[df['abs_Twa'] > 30]

len(df)
df.head()


# -------------------------------------------------------------------------------
# Pipeline
poly_model = make_pipeline(StandardScaler(),
                           PolynomialFeatures(15),
                           LinearRegression())

rf_model = make_pipeline(StandardScaler(),
                         RandomForestRegressor(n_estimators=100))
# Model based on all the data
X = df[['Tws', 'abs_Twa']]
y = df.Bsp
poly_model.fit(X, y)
rf_model.fit(X, y)
score_poly = explained_variance_score(y, poly_model.predict(X))
score_rf = explained_variance_score(y, rf_model.predict(X))
print('4647 point polynomial fit score:{}'.format(score_poly))
print('4647 point polynomial fit score:{}'.format(score_rf))


# Data slice
# Theta_surf = df.Theta.unique()[5]
# X_surf = df[df.Theta == Theta_surf][['Theta', 'alpha', 'r']]
# y_dat = df[df.Theta == Theta_surf].bar_p/Theta_surf**3
# y_fit = poly_model.predict(X_surf)
# df_slice = df[(df['Tws'] < 10.2) & (df['Tws'] > 9.8)]
df_slice = df
X_surf = df_slice[['Tws', 'abs_Twa']]
y_dat = df_slice.Bsp
y_fit_poly = poly_model.predict(X_surf)
y_fit_rf = rf_model.predict(X_surf)

# Phi: Simplified models based on physical arguments
# X[0]=Theta, X[1]=alpha, X[2]=r

#
# def trig(X, a0=1, a1=1, a2=np.pi/2):
#     return a0*np.sin(a1*X[:, 1])*np.cos(a2*X[:, 2])
#
#
# def hyper(X, a0=1, a1=2, a2=4):
#     return a0*(1+np.tanh(a1*X[:, 1]))*np.tanh(a2*(1-X[:, 2]))

pickleIn = open('Models/ORC_Swan45_Rbf.pickle', 'rb')
rbf_swan = pickle.load(pickleIn)
pickleIn.close()


def ORC(X):
    return rbf_swan(X[:, 0], X[:, 1])


IM = ORC

y_IM = IM(X_surf.values)
score_IM = explained_variance_score(y, IM(X.values))
print('Phi model score:{}'.format(score_IM))


# plot
fig = plt.figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(121, projection='3d')
# ax.plot_trisurf(df.Tws, df.abs_Twa, y_fit_poly, alpha=0.8)
# ax.scatter(df.Tws, df.abs_Twa, y_dat, c=y_dat-y_fit_poly, alpha=1)
ax.plot_trisurf(df.Tws, df.abs_Twa, y_fit_rf, alpha=0.8)
ax.scatter(df.Tws, df.abs_Twa, y_dat, c=y_dat-y_fit_rf, alpha=1)
ax.set_xlabel('Tws (Knots)', fontsize=12)
ax.set_ylabel('abs_Twa (deg)', fontsize=12)
ax.set_zlabel('Bsp', fontsize=12)

ax = fig.add_subplot(122, projection='3d')
ax.plot_trisurf(df.Tws, df.abs_Twa, y_IM, color='C1', alpha=0.8)
ax.scatter(df.Tws, df.abs_Twa, y_dat, c=y_dat-y_IM, alpha=1)
ax.set_xlabel('Tws (Knots)', fontsize=12)
ax.set_ylabel('abs_Twa (deg)', fontsize=12)
ax.set_zlabel('Bsp', fontsize=12)
plt.show()
# plt.savefig('surf_3D_Cp.png')
# But the polynomial fit assumes we know all $N=1000$ data points. What if we only have $N=100$... or $N=10$? Lets sub-sample the data and to see how this approach generalizes.
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))

# Make the model "pipeline"
poly_model = make_pipeline(StandardScaler(),
                           PolynomialFeatures(4),
                           #                            LinearRegression()) # overfit until N~1000
                           Ridge())  # best for N>100
#                            RidgeCV()) # over-regularized unless N<50
rf_model = make_pipeline(StandardScaler(),
                         #                            LinearRegression()) # overfit until N~1000
                         RandomForestRegressor(n_estimators=100))
# Evaulate model performance


def subfit(n, model, state=1):  # split -> fit -> test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, random_state=state)
    model.fit(X_train, y_train)
    y_fit = model.predict(X_test)
    return {'score': explained_variance_score(y_test, y_fit), 'n': n, 'state': state}


def evaluate(model):  # loop through the data set sizes
    return pd.DataFrame([subfit(n, model, state)
                         for n in np.logspace(1, 3.6, 20).astype(int)
                         for state in range(20)])


poly_results = evaluate(poly_model)
rf_results = evaluate(rf_model)
rf_results
poly_results

fig = plt.figure(figsize=(6, 3), dpi=100)
ax = fig.add_subplot(111)
plt.axhline(y=score_poly, ls='--', c='k')
poly_results.plot.scatter(x='n', y='score', ax=ax, alpha=0.5)
poly_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, label='Polynomial Ridge')
rf_results.plot.scatter(x='n', y='score', ax=ax, alpha=0.5)
rf_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, label='RandomForest')

plt.ylim(0, 1)
plt.xscale('log')
plt.xlim(10, 4000)
plt.legend()
plt.title('Explained Bsp variance')
plt.xlabel('Training size')
plt.tight_layout()
plt.show()


# plt.savefig('poly-ridge.png')
# %% markdown
# If we only use `LinearRegression` the results are terrible until $N\approx 1000$. `Ridge` regression looks pretty good for $N\ge100$. `RidgeCV` adds cross-validation, but doesn't consistently improve results - it just avoid disasters for $N<40$.
#
# Lets fit the simple physics-based $\Phi$ model as a semi-emperical model instead.

# Use the IM as a SemiEmperical model

#
# class SemiEmperical(RegressorMixin):
#     def fit(self, X=None, y=None):
#         self.alpha, _ = curve_fit(IM, X.values, y.values, p0=[1, 1, np.pi/2])
#
#     def predict(self, X=None):
#         # Give back the mean of y, in the same
#         # length as the number of X observations
#         return IM(X.values, *self.alpha)
#
#
# semi_results = evaluate(SemiEmperical())
#
# fig = plt.figure(figsize=(6, 3), dpi=100)
# ax = fig.add_subplot(111)
# plt.axhline(y=score_1K, ls='--', c='k')
# poly_results.plot.scatter(x='n', y='score', ax=ax, alpha=0.5)
# poly_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, label='Polynomial Ridge')
# semi_results.plot.scatter(x='n', y='score', ax=ax, c='C1', alpha=0.5)
# semi_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, c='C1', label='Semi-emperical')
#
# plt.ylim(0, 1)
# plt.xscale('log')
# plt.xlim(10, 1e3)
# plt.legend()
# plt.title('Explained Bsp variance')
# plt.xlabel('Training size')
# plt.tight_layout()
# plt.show()
# plt.savefig('semi-emp.png')
# %% markdown
# So this is much better for low data sizes (at least on average), but it has no real learning capacity.
#
# Let's use $\Phi$ to create a basis instead - simple concatenation is enough in this case.
# %% codecell
# Concatenate input


def Phi_generator(X):
    return np.column_stack([X, IM(X)])
#     return np.column_stack([X,hyper(X)])


# PBLM pipeline
PBLM_Ridge_model = make_pipeline(FunctionTransformer(Phi_generator, validate=True),
                                 PolynomialFeatures(2),
                                 RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 1e2, 1e3]))

PBLM_rf_model = make_pipeline(FunctionTransformer(Phi_generator, validate=True),
                              RandomForestRegressor(n_estimators=50))


PBLM_Ridge_results = evaluate(PBLM_Ridge_model)
PBLM_rf_results = evaluate(PBLM_rf_model)

fig = plt.figure(figsize=(6, 3), dpi=100)
ax = fig.add_subplot(111)
plt.axhline(y=score_1K, ls='--', c='k')
poly_results.plot.scatter(x='n', y='score', ax=ax, alpha=0.5)
poly_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, label='Polynomial Ridge')
rf_results.plot.scatter(x='n', y='score', ax=ax, alpha=0.5)
rf_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, label='Random Forest')
# semi_results.plot.scatter(x='n', y='score', ax=ax, c='C1', alpha=0.5)
# semi_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, c='C1', label='Semi-emperical')
PBLM_Ridge_results.plot.scatter(x='n', y='score', ax=ax, c='C4', alpha=0.5)
PBLM_Ridge_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, c='C4', label='$\Phi$-basis ridge')
PBLM_rf_results.plot.scatter(x='n', y='score', ax=ax, c='C5', alpha=0.5)
PBLM_rf_results.groupby('n').median().reset_index().plot(x='n', y='score', ax=ax, c='C5', label='$\Phi$-basis rf')

plt.ylim(0, 1)
plt.xscale('log')
plt.xlim(10, 2e3)
plt.legend()
plt.title('Explained Bsp variance')
plt.xlabel('Training size')
plt.tight_layout()
# plt.savefig('phi-basis.png')
plt.show()


# So the results are **much** more robust than the black-box (polynomial ridge regression) to sparse data - both the median and variance are vastly improved. But there is also enough learning capacity to achieve a perfect fit when $N>100$. Note that *adaptive* regularization is key to the $\Phi$-basis working well across the full range.
