"""
Funtions to extract relevant data from a dataframe.
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
import xlrd as xl
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score


class Funcs():
    def __init__(self):

        return

    def Time_info(self, df):
        max(df.Utc)
        min(df.Utc)

        print("Start Time: %s" % (min(df.Utc)))
        print("End Time: %s" % (max(df.Utc)))
        print("Length of time elapsed: %.2f hours" % (len(df.Utc.unique())/60**2))

    def df_extract(self, start, stop, df):  # returns a df with time positions betweeen start and stop from original df
        df = df.set_index(df['Utc'])

        df = df.between_time(start, stop)

        df = df.rename_axis(None)
        df = df.reset_index()
        df = df.drop('index', axis=1)

        return df

    def unfiltered_plot(self, variable, df):  # Plots the unfiltered variable against time
        df = df[np.isfinite(df[variable])]
        y = df[variable]

        return plt.plot(df['Utc'], y)

    def filtered_plot(self, variable, df, N, Wn):  # Plots filtered variable data against time
        df = df[np.isfinite(df[variable])]
        y = df[variable]

        c, a = signal.butter(N, Wn)
        y = signal.filtfilt(c, a, y)
        return plt.plot(df['Utc'], y)

    def Var_filter_compare(self, variable, N, Wn, df):  # Plots both filtered and unfiltered data on the same plot
        plt.figure(figsize=(14, 8))
        self.unfiltered(variable, df)
        self.filtered(variable, df, N, Wn)
        plt.ylabel(variable)
        plt.xlabel('Time')
        plt.show()

    def Segment_points(self, df):  # Returns the Lat Lon positions when a manoeuver occurs.
        df = df[np.isfinite(df['Hdg'])]
        df = df.reset_index().drop('index', axis=1)

        i = 30
        Lat_arr = []
        Lon_arr = []
        Time_arr = []

        while i < len(df):
            angle = min(abs(df.at[i, 'Hdg'] - df.at[i-30, 'Hdg']),
                        360 - abs(df.at[i, 'Hdg'] - df.at[i-30, 'Hdg']))
            if angle > 50:

                Lon_arr = np.append(Lon_arr, df.at[i, 'Lon'])
                Lat_arr = np.append(Lat_arr, df.at[i, 'Lat'])
                Time_arr = np.append(Time_arr, df.at[i, 'Utc'])

                i += 40

            i += 1
        return Lon_arr, Lat_arr, Time_arr

    def Segment_times(self, df):    # Returns the times when a manoeuver occurs

        T_arr = []

        i = 1
        while i < len(df):

            current_sign = np.sign(df.loc[i, "Twa"])
            previous_sign = np.sign(df.loc[i - 1, "Twa"])

            if current_sign != previous_sign:
                T_arr = np.append(T_arr, df.loc[i, 'Utc'])

                i += 40  # Prevents some noise mid manoeb=ver registering as two separate manoevers
            i += 1

        return T_arr

    def time_to_str(self, time):  # Converts time to string format to be used in Segment_df()
        t = str(time.hour)+':'+str(time.minute)+':'+str(time.second)

        return t

    def Segment_plot(self, df):  # Plots each individual segment from a dataframe

        segment_df = self.Segment_df(df)
        i = 1
        while i <= len(segment_df['Segment'].unique()):
            plt.scatter(segment_df[segment_df['Segment'] == i].Lon, segment_df[segment_df['Segment'] == i].Lat, alpha=0.7, s=4)

            i += 1

    def Segment_df(self, df):  # Returns a df labeled to include each segment
        # time_arr = self.Segment_times(df)
        lonarr, latarr, time_arr = self.Segment_points(df)

        begin = self.time_to_str(df.at[0, 'Utc'])
        end = self.time_to_str(time_arr[0])

        Segment_df = self.df_extract(begin, end, df)
        Segment_df.insert(0, "Segment", 1)

        i = 1
        while i < len(time_arr):
            begin = self.time_to_str(time_arr[i-1])
            end = self.time_to_str(time_arr[i])
            df_temp = self.df_extract(begin, end, df)
            df_temp.insert(0, "Segment", i+1)

            Segment_df = pd.concat([Segment_df, df_temp])

            i += 1

        return Segment_df

    # Label the tacks and gybe contained in a

    def manoeuver_label(self, df, man_len=60):
        # Get the times that a manoever takes place
        Lon, Latt, times = self.Segment_points(df)

        if "Pred" in df.columns:
            my_label = "Pred"

        elif "label" in df.columns:
            my_label = "label"

        for time in times:

            index = df[df['Utc'] == time].index[0]

            label = df.loc[index - man_len/2:index + man_len/2, my_label].value_counts().idxmax()

            if label == 'UW':
                df.loc[index-man_len/2:index+man_len/2, my_label] = 'Tack'

            elif label == 'DW':
                df.loc[index-man_len/2:index+man_len/2, my_label] = 'Gybe'

        return df

    def Segment_scatter(self, df):  # Plots the positions of each manoeuver

        my_lonarr, my_latarr, my_timearr = self.Segment_points(df)
        plt.scatter(my_lonarr, my_latarr, alpha=0.8, s=45)

    def Segment_Label(self, UW_arr, DW_arr, NR_arr, df):  # Gives label to data in a df based on list of segments provided

        Seg_df = self.Segment_df(df)
        Seg_df.insert(0, 'label', '')
        Seg_df = Seg_df.reset_index().drop('index', axis=1)

        i = 0
        while i < len(Seg_df):
            for seg in UW_arr:
                if Seg_df.at[i, "Segment"] == seg:
                    Seg_df.at[i, 'label'] = 'UW'

            for seg in DW_arr:
                if Seg_df.at[i, "Segment"] == seg:
                    Seg_df.at[i, "label"] = 'DW'

            for seg in NR_arr:
                if Seg_df.at[i, "Segment"] == seg:
                    Seg_df.at[i, "label"] = 'NR'
            i += 1

        return Seg_df

    def Segment_time_label(self, df, Segment, TRANSITION, label_1, label_2):

        seg_df = df[df['Segment'] == Segment]
        START_index = seg_df.first_valid_index()
        TRANSITION_index = seg_df.index[seg_df['Utc'] == TRANSITION].to_list()[0]
        END_index = seg_df.last_valid_index()

        i = START_index
        while i < TRANSITION_index:
            df.at[i, 'label'] = label_1
            i += 1

        while i <= END_index:
            df.at[i, 'label'] = label_2
            i += 1

        return df

    def ARRAY_check(self, UW_arr, DW_arr, NR_arr, Interval_arr, df_segment):
        l1 = len(UW_arr)
        l2 = len(DW_arr)
        l3 = len(NR_arr)
        l4 = len(Interval_arr)
        len_tot = l1+l2+l3+l4

        len_true = len(df_segment.Segment.unique())

        if len_true == len_tot:
            print("Correct Number of segments labeled")

        else:

            print("Number of Segments: %d" % (len_true))
            print("Number of Segments Labeled: %d\n" % (len_tot))

            lin = np.linspace(1, len(df_segment.Segment.unique()), len(df_segment.Segment.unique()))
            my_arr = np.concatenate((UW_arr, DW_arr, NR_arr, lin, Interval_arr))
            unique, Counts = np.unique(my_arr, return_counts=True)

            i = 0
            while i < len(Counts):
                if Counts[i] != 2:
                    print("ERROR with value %d" % (i+1))

                i += 1

    def df_row_nan(self, df, drop_list):  # Returns a trimmed df that has no NaN values in first row
        i = 0
        while i < len(df):

            my_arr = np.isfinite(np.array(df.drop(drop_list, axis=1).loc[i]))
            sum_count = 0

            j = 0
            while j < len(my_arr):

                if my_arr[j] == True:
                    sum_count += 1

                j += 1

            if sum_count == len(df.columns)-1:
                break

            df = df.drop(i)
            i += 1

        df = df.reset_index().drop('index', axis=1)  # Reindexes new df

        return df

    def df_fill_av(self, df, drop_list):   # Fills nan values in df with the average of its neighbours values.

        df = self.df_row_nan(df, drop_list)  # Ensures first row has no NaN values

        for var in df.columns.drop(drop_list):
            i = 0
            while i < len(df)-2:

                if math.isnan(df.at[i, var]) == True and math.isnan(df.at[i+1, var]) == False:
                    df.at[i, var] = (df.at[i-1, var]+df.at[i+1, var])*0.5

                elif math.isnan(df.at[i, var]) == True:
                    df.at[i, var] = (df.at[i-1, var]+df.at[i+2, var])*0.5

                i += 1
        # Returns averaged data excluding the last two rows which could not be averaged
        return df.drop([len(df)-1, len(df)-2])

    def gps_clean(self, df):  # Removes erroneous gps recordings. NOTE: will not work if distance between points is actually large

        df = df[df['Lat'] < df.Lat.mean()+0.5]
        df = df[df['Lat'] > df.Lat.mean()-0.5]
        df = df[df['Lon'] < df.Lon.mean()+0.5]
        df = df[df['Lon'] > df.Lon.mean()-0.5]

        return df

    # Takes UW and Segment labeled df and returns which segments which have greater than 50% votes for UW.

    def Segment_keep(self, df):
        i = 1
        keep_list = []
        while i < len(df.Segment.unique()):

            Len = len(df[df['Segment'] == i])

            N_UW = len(df[df['Segment'] == i][df[df['Segment'] == i]['UW'] == 1])

            if N_UW > 0.5*Len and len(df[df['Segment'] == i]) > 45:

                keep_list = np.append(keep_list, i)

            i += 1

        return keep_list

    def df_UW(self, df):  # Returns a data frame with only the segments that we want to keepdd

        keep = self.Segment_keep(df)

        df_UW = df[df["Segment"] == keep[0]]
        for seg in keep:

            df_UW = pd.concat([df_UW, df[df["Segment"] == seg]])

        return df_UW

    def arrow_plot(self, df, BBox):

        Lon_len = BBox[1]-BBox[0]
        Lat_len = BBox[3]-BBox[2]

        a1 = np.arctan((Lon_len)/(Lat_len))
        a2 = a1+2*np.arctan((Lat_len)/(Lon_len))
        a3 = a2+2*a1
        a4 = a3+a2-a1

        TWD = df.Twd.mean()*np.pi/180

        x_centre = (BBox[1]+BBox[0])/2
        y_centre = (BBox[3]+BBox[2])/2

        length = 0.03*max(Lon_len, Lat_len)

        if TWD <= a1:
            x = x_centre+np.tan(TWD)*Lat_len/2
            y = y_centre+Lat_len/2

            dx = length*np.sin(TWD)
            dy = length*np.cos(TWD)

            plt.arrow(x, y, -dx, -dy, color='green')

        elif TWD > a1 and TWD <= np.pi/2:
            x = x_centre+Lon_len/2
            y = y_centre+Lon_len/2*np.tan(np.pi/2-TWD)

            dx = length*np.cos(np.pi/2-TWD)
            dy = length*np.sin(np.pi/2-TWD)

            plt.arrow(x, y, -dx, -dy, color='green')

        elif TWD > np.pi/2 and TWD <= a2:
            x = x_centre+Lon_len/2
            y = y_centre-((Lon_len/2)/(np.tan(np.pi-TWD)))

            dx = length*np.sin(np.pi-TWD)
            dy = length*np.cos(np.pi-TWD)

            plt.arrow(x, y, -dx, +dy, color='green')

        elif TWD > a2 and TWD <= np.pi:
            x = x_centre+Lat_len/2*np.tan(np.pi-TWD)
            y = y_centre-Lat_len/2

            dx = length*np.sin(np.pi-TWD)
            dy = length*np.cos(np.pi-TWD)

            plt.arrow(x, y, -dx, +dy, color='green')

        elif TWD > np.pi and TWD <= a3:
            x = x_centre-Lat_len/2*np.tan(TWD-np.pi)
            y = y_centre-Lat_len/2

            dx = length*np.sin(TWD-np.pi)
            dy = length*np.cos(TWD-np.pi)

            plt.arrow(x, y, +dx, +dy, color='green')

        elif TWD > a3 and TWD <= 3/2*np.pi:
            x = x_centre-Lon_len/2
            y = y_centre-Lon_len/2*np.tan(3/2*np.pi-TWD)

            dx = length*np.cos(3/2*np.pi-TWD)
            dy = length*np.sin(3/2*np.pi-TWD)

            plt.arrow(x, y, +dx, +dy, color='green')

        elif TWD > 3/2*np.pi and TWD <= a4:
            x = x_centre-Lon_len/2
            y = y_centre+Lon_len/2*np.tan(TWD-np.pi*3/2)

            dx = length*np.cos(TWD-3/2*np.pi)
            dy = length*np.sin(TWD-3/2*np.pi)

            plt.arrow(x, y, +dx, -dy, color='green')

        elif TWD > a4:
            x = x_centre-Lat_len/2*np.tan(2*np.pi-TWD)
            y = y_centre+Lat_len/2

            dx = length*np.sin(2*np.pi-TWD)
            dy = length*np.cos(2*np.pi-TWD)

            plt.arrow(x, y, +dx, -dy, color='green')

    def Norm_Pos(self, window_df):

        new_df = pd.DataFrame(columns=window_df.columns)
        new_df['Lat_norm'] = 0
        new_df['Lon_norm'] = 0

        for window in window_df.window.unique():

            # Create temporary df for each window to normalise position data
            temp_df = window_df[window_df['window'] == window].reset_index().drop('index', axis=1)

            # Find Twd for given window
            TWD = temp_df.Twd.mean()
            # Convert to radians
            TWD = TWD*np.pi/180

            # Find the max/min positions of Lat/Lon in the temporary df
            x_max = temp_df.Lon.max()
            x_min = temp_df.Lon.min()
            y_max = temp_df.Lat.max()
            y_min = temp_df.Lat.min()

            i = 0
            while i < len(temp_df):

                # Scale the Lat/Lon data at each point wrt the given window.
                Lon_norm = (temp_df.at[i, 'Lon']-x_min)/(x_max-x_min)
                Lat_norm = (temp_df.at[i, 'Lat']-y_min)/(y_max-y_min)

                # Translate the axis so that the y axis is pointing in the direction of TWD for given window.
                x = Lon_norm*np.cos(-TWD) + Lat_norm*np.sin(-TWD)
                y = -Lon_norm*np.sin(-TWD) + Lat_norm*np.cos(-TWD)

                if np.isfinite(Lon_norm) == True and np.isfinite(Lat_norm) == True:

                    # Scale each Lon/Lat reading and save in seperate column for simplicity of comparisson
                    temp_df.at[i, 'Lon_norm'] = x
                    temp_df.at[i, 'Lat_norm'] = y
                    # Append the new df to include all new information, this is slow.
                    new_df = new_df.append(temp_df.loc[i, :])

                else:

                    # If normalised values Nan then set values in this window to be 0,0
                    temp_df.at[i, 'Lon_norm'] = 0
                    temp_df.at[i, 'Lat_norm'] = 0
                    # Append the new df to include all new information, this is slow.
                    new_df = new_df.append(temp_df.loc[i, :])

                i += 1

        return new_df

    # Function comparing labeled and predicted results

    def compare(self, Labeled_df, Pred_df):

        correct = 0
        wrong = 0
        total = 0

        # Only compare the times that are present in both data frames to gve more accurate percentage.
        l1 = list(Labeled_df.Utc)
        l2 = list(Pred_df.Utc)
        for time in list(set(l1).intersection(l2)):
            # for time in Labeled_df.Utc.unique():

            Pred_label = Pred_df[Pred_df['Utc'] == time].Pred.to_list()[0]
            Corr_label = Labeled_df[Labeled_df['Utc'] == time].label.to_list()[0]

            if Pred_label == Corr_label:
                correct += 1

            else:
                wrong += 1

            total += 1

        correct_perc = correct/total*100

        return correct_perc

    def Window(self, df, window_len, overlap):

        # Array of unique dates contained in df so that windowing does not go over 2 days.
        df['date'] = df['Utc'].map(lambda x: x.strftime('%Y-%m-%d'))
        date_arr = df['date'].unique()

        # create the new dataframe which the windowed segments will be added to
        window_df = pd.DataFrame(columns=df.columns)
        window_df['window'] = 0

        # Global window count
        k = 1

        for date in date_arr:
            # create dataframe of just given date to work with
            date_df = df[df['date'] == date].reset_index().drop('index', axis=1)
            date_df['window'] = 0
            len_data = len(date_df)

            # Set the maximum number of complete time series with given inputs to be n
            n = int((len_data-overlap)/(window_len-overlap))

            j = 1  # window count
            i = 0  # row count

            while j <= n:
                while i < window_len+(j-1)*(window_len-overlap):
                    date_df.at[i, 'window'] = k
                    window_df = window_df.append(date_df.loc[i, ])

                    i += 1

                # Step back the overlap length for indices before continuing to next window
                i -= overlap
                # Increase the window count for this date
                j += 1
                # Increase the global window count
                k += 1

        return window_df

    def Labeled_cluster_plot(self, df, UW_arr, DW_arr, NR_arr, Map_arr, Labeled_df_arr, red=100000):

        # Array of unique dates contained in df so that windowing does not go over 2 days.
        df['date'] = df['Utc'].map(lambda x: x.strftime('%Y-%m-%d'))
        date_arr = df['date'].unique()

        i = 0
        while i < len(df.date.unique()):

            Pred_df = df[df['date'] == df.date.unique()[i]]
            Map = Map_arr[i]

            Labeled_df = Labeled_df_arr[i]

            # LHS: Individual cluster plot
            BBox = (Pred_df.Lon.min(), Pred_df.Lon.max(), Pred_df.Lat.min(), Pred_df.Lat.max())
            plt.figure(figsize=(14, 10))

            plt.subplot(1, 2, 1)
            plt.imshow(plt.imread(Map), extent=BBox, aspect='equal')

            for pred in df.Pred.unique():
                if pred == red:
                    plt.scatter(Pred_df[Pred_df['Pred'] == pred].Lon, Pred_df[Pred_df['Pred']
                                                                              == pred].Lat, alpha=0.3, s=5, label='0', color='r')

                elif pred in UW_arr:
                    plt.scatter(Pred_df[Pred_df['Pred'] == pred].Lon, Pred_df[Pred_df['Pred']
                                                                              == pred].Lat, alpha=0.3, s=5, label='0', color='b')

                elif pred in DW_arr:
                    plt.scatter(Pred_df[Pred_df['Pred'] == pred].Lon, Pred_df[Pred_df['Pred']
                                                                              == pred].Lat, alpha=0.3, s=5, label='0', color='orange')

                elif pred in NR_arr:
                    plt.scatter(Pred_df[Pred_df['Pred'] == pred].Lon, Pred_df[Pred_df['Pred']
                                                                              == pred].Lat, alpha=0.3, s=5, label='0', color='g')

          # plt.legend()

            plt.subplot(1, 2, 2)
            # RHS: Labeled UW data
            BBox = (Labeled_df.Lon.min(), Labeled_df.Lon.max(),
                    Labeled_df.Lat.min(), Labeled_df.Lat.max())

            plt.imshow(plt.imread(Map), extent=BBox, aspect='equal')
            plt.scatter(Labeled_df[Labeled_df['label'] == 'UW'].Lon,
                        Labeled_df[Labeled_df['label'] == 'UW'].Lat, alpha=0.3, s=5, label='UW')
            plt.scatter(Labeled_df[Labeled_df['label'] == 'DW'].Lon, Labeled_df[Labeled_df['label']
                                                                                == 'DW'].Lat, alpha=0.3, s=5, label='DW', color='orange')
            plt.scatter(Labeled_df[Labeled_df['label'] == 'NR'].Lon,
                        Labeled_df[Labeled_df['label'] == 'NR'].Lat, alpha=0.3, s=5, label='NR', color='g')
            plt.legend()

            plt.show()

            i += 1

            # Automate process of assigning labels to clusters, the parameters are subject to user guidance.

    def auto_labels(self, df):

        UW_arr = []
        DW_arr = []
        NR_arr = []

        for pred in df.Pred.unique():

            pred_df = df[df['Pred'] == pred]

            if pred_df.Bsp.mean() > 6 and pred_df.abs_Leeway.mean() > 2.5:  # and pred_df.abs_Heel.mean() > 10 :
                UW_arr.append(pred)

            # and pred_df.abs_Heel.mean() > 5.0 and pred_df.Bsp_Tws > 0.75 :
            elif pred_df.Bsp.mean() > 8 and pred_df.abs_Twa.mean() > 110*np.pi/180 and pred_df.Bsp_Tws.mean() > 0.67 and pred_df.abs_Heel.mean() > 5:
                DW_arr.append(pred)

            else:
                NR_arr.append(pred)

        return UW_arr, DW_arr, NR_arr

    # Add predicted cluster labels(y_vals) to a given dataframe which may be windowded.

    def pred_label_df(self, df, y_vals):

        if 'Pred' in df.columns:
            df = df.drop('Pred', axis=1)

        # Adding prediction column to df
        df['Pred'] = 0

        if 'window' in df.columns:
            # length of window in df
            length = len(df[df['window'] == 1])

        else:
            length = 1

        # index count
        i = 0

        # label count
        k = 0

        while i < len(df):
            df.iloc[i:i+length, -1] = y_vals[k]

            i += length
            k += 1

        return df

    def stats(self, df):
        # Statistical desciption
        desc_df = df.describe()

        desc_df.loc["+3_std"] = desc_df.loc['mean'] + (desc_df.loc['std'] * 3)
        desc_df.loc["-3_std"] = desc_df.loc['mean'] - (desc_df.loc['std'] * 3)

        return desc_df

    def Scale_df(self, df, features, scaling_type):

        # Define each scaler
        min_max_scaler = MinMaxScaler()
        std_scaler = StandardScaler()
        robust_scaler = RobustScaler()

        if scaling_type == 'min_max':
            for feature in features:
                df[feature +
                    '_scaled'] = min_max_scaler.fit_transform(np.array(df[feature]).reshape(-1, 1))

        elif scaling_type == 'standard':
            for feature in features:
                df[feature + '_scaled'] = std_scaler.fit_transform(np.array(df[feature]).reshape(-1, 1))

        elif scaling_type == 'robust':
            for feature in features:
                df[feature +
                    '_scaled'] = robust_scaler.fit_transform(np.array(df[feature]).reshape(-1, 1))

        else:
            print("ERROR in scaling_type input")

        return df

    def Series_label_df(self, df):
        # Give a series number to separate each consecutive time series for later use with RNN etc.

        df['Series_num'] = 0

        index_arr = df.index
        start = index_arr[0]

        i = 0
        k = 1
        while i < len(index_arr)-1:

            if abs(index_arr[i] - index_arr[i+1]) > 1:
                end = index_arr[i]

                df.loc[start:end, 'Series_num'] = k

                start = index_arr[i+1]

                k += 1

            i += 1

        return df

    def Labeled_df_plot(self, df, Map_arr, Labeled_df_arr):

        # Array of unique dates contained in df so that windowing does not go over 2 days.
        df['date'] = df['Utc'].map(lambda x: x.strftime('%Y-%m-%d'))
        date_arr = df['date'].unique()

        i = 0
        while i < len(df.date.unique()):

            Pred_df = df[df['date'] == df.date.unique()[i]]
            Map = Map_arr[i]

            Labeled_df = Labeled_df_arr[i]

            # LHS: Individual cluster plot
            BBox = (Pred_df.Lon.min(), Pred_df.Lon.max(), Pred_df.Lat.min(), Pred_df.Lat.max())
            plt.figure(figsize=(14, 10))
            plt.imshow(plt.imread(Map), extent=BBox, aspect='equal')

            for label in df.label.unique():
                if label == 'UW':
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='b')

                elif label == 'DW':
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='orange')

                elif label == 'NR':
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='g')

                elif label == 'Tack':
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='r')

                elif label == 'Gybe':
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='purple')

                else:
                    plt.scatter(Pred_df[Pred_df['label'] == label].Lon,
                                Pred_df[Pred_df['label'] == label].Lat, alpha=0.3, s=5, color='black')

            i += 1

    def mean_absolute_percentage_error(self, y_true, y_pred):

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def VPP_compare(self, df):

        feat_list = ['Bsp', 'Heel', 'Leeway']

        for feat in feat_list:

            VPP_feat = 'VPP_' + feat

            if feat == 'Heel' or feat == 'Leeway':

                feat = 'abs_' + feat

            RMSE = mean_squared_error(df[feat], df[VPP_feat], squared=False)
            MAE = mean_absolute_error(df[feat], df[VPP_feat])
            MAPE = self.mean_absolute_percentage_error(df[feat], df[VPP_feat])
            r2 = r2_score(df[feat], df[VPP_feat])
            explained_variance = explained_variance_score(df[feat], df[VPP_feat])
            maximum_error = max_error(df[feat], df[VPP_feat])

            print('-' * 100)
            print("RMSE %s: %.3f" % (feat, RMSE))
            print("MAE %s: %.3f" % (feat, MAE))
            print("MAPE %s: %.3f" % (feat, MAPE))
            print("R^2 %s: %.3f" % (feat, r2))
            print("Explained variance %s: %.3f" % (feat, explained_variance))
            print("Maximum Error %s: %.3f\n" % (feat, maximum_error))

    def print_scores(self, y_test, y_pred):

        RMSE = mean_squared_error(y_test, y_pred, squared=False)
        MAE = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        maximum_error = max_error(y_test, y_pred)

        print("RMSE: %.3f" % (RMSE))
        print("MAE: %.3f" % (MAE))
        print("R^2: %.3f" % (r2))
        print("Maximum Error: %.3f\n" % (maximum_error))
