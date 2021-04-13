import seaborn as sns
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib qt
sns.set()


# --------------------------------------------------------------------------------------------------
# Import both datasets separately
dfTP = pd.DataFrame()
dfTP = dfTP.append(pd.read_csv(r'Data/Labeled_all_VPP_waves.csv'))

dfTP = dfTP[(dfTP['label'] == 'UW') | (dfTP['label'] == 'DW')]

dfSwan = pd.DataFrame()
dfSwan = dfSwan.append(pd.read_csv(r'Data/Swan45_DataCombined.csv'))

dfSwan = dfSwan[(dfSwan['label'] == 'UW') | (dfSwan['label'] == 'DW')]


# --------------------------------------------------------------------------------------------------
# Plotting functions
def plot_polar_raw(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.scatter(df.abs_Twa*np.pi/180, df.Bsp, c=df.Tws, cmap='Blues', label='Raw Data')

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_rmax(df.Bsp.max()+1.5)
    plt.legend()
    plt.show()

    return


def plot_polar_3d(df):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.Bsp, df.abs_Twa, df.Tws, c='b', label='Polar Plot')
    ax.set_xlabel('Bsp', fontsize=12)
    ax.set_ylabel('Twa', fontsize=12)
    ax.set_zlabel('Tws', fontsize=12)
    plt.show()

    return


def WindDist_Plot(df,):
    # Tp-Hs plot
    ax = sns.jointplot(x=df.Tws, y=df.Twa, kind='hex').set_axis_labels("Tws", "Twa")
    ax.ax_joint.grid()
    ax.ax_joint.set_xlim([5, 25])
    ax.ax_joint.set_ylim([-180, 180])
    plt.tight_layout()
    plt.show()

    return


# --------------------------------------------------------------------------------------------------
# Import ORC Swan model
pickleIn = open('Models/ORC_Swan45_Rbf.pickle', 'rb')
rbf_swan = pickle.load(pickleIn)
pickleIn.close()


def ORC(X):
    return rbf_swan(X[:, 0], X[:, 1])

# --------------------------------------------------------------------------------------------------
# Plots for the TP52 dataset
# Polar plot colored by TWS


plot_polar_raw(dfTP)


# 3d plot colored by wave parameters
fig = plt.figure(figsize=(8, 3), dpi=100)
ax = fig.add_subplot(141, projection='3d')
ax.scatter(dfTP.Bsp, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Heel_ampl, cmap='Blues', label='Colored by Heel_ampl')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(142, projection='3d')
ax.scatter(dfTP.Bsp, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Trim_ampl, cmap='Reds', label='Colored by Trim_ampl')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(143, projection='3d')
ax.scatter(dfTP.Bsp, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Heel_freq, cmap='Greens', label='Colored by Heel_Freq')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(144, projection='3d')
ax.scatter(dfTP.Bsp, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Trim_freq, cmap='Purples', label='Colored by Trim_freq')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)
plt.tight_layout()
plt.legend()
plt.show()


dfTP.columns
# plot model residuals
fig = plt.figure(figsize=(8, 3), dpi=100)
ax = fig.add_subplot(141, projection='3d')
ax.scatter(dfTP.VPP_Bsp_error, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Heel_ampl, cmap='Blues', label='Colored by Heel_ampl')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(142, projection='3d')
ax.scatter(dfTP.VPP_Bsp_error, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Trim_ampl, cmap='Reds', label='Colored by Trim_ampl')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(143, projection='3d')
ax.scatter(dfTP.VPP_Bsp_error, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Heel_freq, cmap='Greens', label='Colored by Heel_Freq')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)

ax = fig.add_subplot(144, projection='3d')
ax.scatter(dfTP.VPP_Bsp_error, dfTP.abs_Twa, dfTP.Tws, c=dfTP.Trim_freq, cmap='Purples', label='Colored by Trim_freq')
ax.set_xlabel('Bsp', fontsize=12)
ax.set_ylabel('Twa', fontsize=12)
ax.set_zlabel('Tws', fontsize=12)
plt.tight_layout()
plt.legend()
plt.show()

plt.hist(dfTP.VPP_Bsp_error)


# -------------
# Histogram plots of wave parameters
bins = 50
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(141)
ax.set_title('Heel_ampl')
ax.hist(dfTP.Heel_ampl, bins=bins)

ax = fig.add_subplot(142)
ax.set_title('Heel_freq')
ax.hist(dfTP.Heel_freq, bins=bins)

ax = fig.add_subplot(143)
ax.set_title('Trim_ampl')
ax.hist(dfTP.Trim_ampl, bins=bins)

ax = fig.add_subplot(144)
ax.set_title('Trim_freq')
ax.hist(dfTP.Trim_freq, bins=bins)

# ---------------------------------
# Distribution of wind conditions

WindDist_Plot(dfTP)


# --------------------------------------------------------------------------------------------------
# plots for Swan45 Data
plot_polar_raw(dfSwan)

plot_polar_3d(dfSwan)

WindDist_Plot(dfSwan)
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
