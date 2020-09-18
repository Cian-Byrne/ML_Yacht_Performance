# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
#import pandas.rpy.common as com
import rpy2.robjects.packages as rpackages
from Data_Clean_Functions import *
import time



# get R random forest function for unsupervised learning
randomForest = importr('randomForest')

Dir = "C://Users//cdb1g19//Downloads//"


important_feats = ['Utc','Bsp','Awa','Aws','Twa','Tws','Twd','Leeway','Hdg',
                   'Lat','Lon','Cog','Sog','Rudder','Heel','Trim','Forestay',
                   'Rake']

d1 = pd.read_csv(Dir+'2015Sep13_0.clean.csv',usecols = important_feats)
d2 = pd.read_csv(Dir+'2015Sep14_0.clean.csv',usecols = important_feats)
d3 = pd.read_csv(Dir+'2015Sep15_0.clean.csv',usecols = important_feats)
d4 = pd.read_csv(Dir+'2015Sep16_0.clean.csv',usecols = important_feats)
d5 = pd.read_csv(Dir+'2015Sep17_0.clean.csv',usecols = important_feats)
d6 = pd.read_csv(Dir+'2015Sep18_0.clean.csv',usecols = important_feats)
d7 = pd.read_csv(Dir+'2015Sep19_0.clean.csv',usecols = important_feats)
d8 = pd.read_csv(Dir+'2015Sep20_0.clean.csv',usecols = important_feats)

# Convert time column to date-time.
d_arr=[d1,d2,d3,d4,d5,d6,d7,d8]

for day in d_arr:
    day['Utc'] = day['Utc'].apply(lambda x: xl.xldate_as_tuple(x, 0))
    a = pd.DataFrame(day['Utc'].values.tolist(), columns=['year', 'month', 'day', 'hour', 'm', 'second'])
    b = pd.to_datetime(a)
    day['Utc']=b

# Fill missing values day by day.
i=0
while i < len(d_arr):
    
    d_arr[i]=df_fill_av(d_arr[i],'Utc')
    
    i+=1

#Combine data to one data frame
df = pd.concat(d_arr)
df = df[df['Lon']<-9.4]

#Reset index to avoid repeating index
df = df.reset_index().drop('index',axis=1)



# Load in Labeled data
day1 = pd.read_csv(Dir+"//Labeled_TP52_day1.csv")
day2 = pd.read_csv(Dir+"//Labeled_TP52_day2.csv")
day3 = pd.read_csv(Dir+"//Labeled_TP52_day3.csv")
day4 = pd.read_csv(Dir+"//Labeled_TP52_day4.csv")
day5 = pd.read_csv(Dir+"//Labeled_TP52_day5.csv")
day6 = pd.read_csv(Dir+"//Labeled_TP52_day6.csv")
day7 = pd.read_csv(Dir+"//Labeled_TP52_day7.csv")
day8 = pd.read_csv(Dir+"//Labeled_TP52_day8.csv")

# Convert time column to date-time.
day_arr=[day1,day2,day3,day4,day5,day6,day7,day8]

for day in day_arr:
    day["Utc"] = pd.to_datetime(day['Utc'])
    
    

#Combine data to one data frame
Labeled_df=pd.concat(day_arr)
Labeled_df.label.fillna('NR',inplace=True)

#Reset index to avoid repeating index
Labeled_df=Labeled_df.reset_index().drop('index',axis=1)

#Labeled_df = df
#df = df.drop('label', axis = 1)




# Fill na values to be 0, NEEDS IMPROVEMENT
df.fillna(0,inplace=True)

#Create abs value columns for Awa, Twa, leeway, rudder and heel 
abs_arr = ['Awa','Twa','Leeway','Rudder','Heel']
for variable in abs_arr:
    
    df['abs_'+variable]=abs(df[variable])
    
# Create new combination of features
df['Bsp_Tws'] = df['Bsp']/(df['Tws']+0.0001)
df['AwaxBsp'] = df['Bsp']*df['abs_Awa']
df['Bsp2'] = df['Bsp']**2
df['Tws_Heel'] = np.log((df['Tws']+0.0001)/(df['abs_Heel']+0.001))

#Important features for classification
scale_features = ['Bsp2','abs_Awa','abs_Twa','Trim','Forestay','abs_Heel',
                  'Tws','abs_Leeway','abs_Rudder','Bsp_Tws','Tws_Heel',
                  'AwaxBsp']

df = Scale_df(df,scale_features,'standard')

features=['Bsp2_scaled','abs_Awa_scaled','Trim_scaled','Forestay_scaled',
          'abs_Heel_scaled','Tws_scaled','abs_Leeway_scaled',
          'abs_Rudder_scaled','Bsp_Tws_scaled','Tws_Heel_scaled',
          'AwaxBsp_scaled']

# Creating Training array
temp_df = df[features]

Train_arr = []
length = 40
i = 0
while i < len(df)-length:
    
    temp_arr = []
    
    for feat in features: 
        val = temp_df.iloc[i:i+length][feat].mean()
        temp_arr.append(val)
        
    Train_arr.append(np.array(temp_arr))
    
    i+=length

X_train = np.array(Train_arr)
# Convert to R object for unsupervised learning
rpy2.robjects.numpy2ri.activate()


### Convert Train array into r object for use in URF function ##
#
nr, nc = X_train.shape
r_train = robjects.r.matrix(X_train, nrow=nr, ncol=nc)

t1 = time.time()
# Create function to obtain proximity matrix from R function randomForest
robjects.r('''
           f <- function(X_train) {

                    library('randomForest')
                    
                    rf <- randomForest(x = X_train, mtry = 2, ntree = 2000, proximity = TRUE)
                    prox <- rf$proximity

            }
            ''')
            
URF = robjects.globalenv['f']

# Calculate the proximity matrix
proximity = (URF(r_train))
proximity = np.array(proximity)


## Fit KMeans cluster model to the proximity matrix generated in URF
kmeans = KMeans(n_clusters=30,n_init=20)
kmeans.fit(proximity)


y_vals= kmeans.predict(proximity)

## Fit KMedoids model to proximity matrix
#Kmedoids = KMedoids(n_clusters = 20)
#Kmedoids.fit(proximity)
#y_vals= Kmedoids.predict(proximity)

### Add the predicted cluster label to the correct points ###
df['Pred']=0
# index count
i=0

# label count
k=0

while i < len(df)-length:
    df.loc[i:i+length,'Pred'] = y_vals[k]

    i+=length
    k+=1


# Automatically label each cluster
UW_arr, DW_arr, NR_arr = auto_labels(df)


df = df.astype({'Pred':'str'})

# Re-Label the clustered data.
df['Pred'].replace(np.array(UW_arr).astype('str'),'UW',inplace=True)
df['Pred'].replace(np.array(DW_arr).astype('str'),'DW',inplace=True)
df['Pred'].replace(np.array(NR_arr).astype('str'),'NR',inplace=True)

t2 = time.time()
print("Fit Time: %.4f s"%(t2-t1))

print("Length: %d"%(length))
print("Accuracy of Unsupervised RF model: %.3f"%(compare(Labeled_df, df)))
        


        
        
