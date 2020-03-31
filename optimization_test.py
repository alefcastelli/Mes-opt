## %% Part Zero: Options
# importing modules
from pandas import read_excel as read_excel
import numpy as np
from sklearn.cluster import KMeans as KMeans
from matplotlib import pyplot as plt
from collections import Counter

options = {
    # Number of clusters
    'n_clusters': 3,
    # Time interval. Unit: [Hr]. If fraction of a hour, then it must be lower than 1
    # Time interval  must be inserted after looking at the excel datasheet
    'dt': 1,
    }

# Matrix containing the first and last position of the week
cluster_flags = np.zeros( (options["n_clusters"], 2) )

for i in range(0,options["n_clusters"]):
    cluster_flags[i,:] = [ i*int(24/options["dt"]*7), (i+1)*int(24/options["dt"]*7) - 1 ]

## Part One: Upload timetable and generate typical weeks
# Read demand profiles from xlsx file (insert the folder path where file is located)
demands = read_excel('/Users/Lorenzo/Desktop/Prova modello python/Book1.xlsx')
PV_output = read_excel('/Users/Lorenzo/Desktop/Prova modello python/PVdata_2013.xlsx')

# Change columns names
columns = ['Q', 'EE', 'Cold', 'Time', 'TimeStep']
demands.columns = columns
PV_output.columns = ['Sensor ID', 'Time', 'Irr', 'Temp', 'T cell', 'Eff cell', 'El output']

# Change in matrix format
#Demands = demands.as_matrix()

# Creating dictionary to collect demand profiles
# Profiles are called with demands['good_name']
# Profiles are column vectors
demands = {
    'Q': np.transpose(demands['Q'][:].values),
    'EE': np.transpose(demands['EE'][:].values),
    'Cold': np.transpose(demands['Cold'][:].values)
    }

PV_data = {
    'El_out': np.transpose(PV_output['El output'][:].values)
}
# Creating weekly profiles for clustering
# Weekly demands
weekly_demands = {
    'Q': np.reshape(demands['Q'][:-int(24/options["dt"])], [int(24/options["dt"]*7),52], order = 'F'),
    'EE': np.reshape(demands['EE'][:-int(24/options["dt"])], [int(24/options["dt"]*7),52], order = 'F'),
    'Cold': np.reshape(demands['Cold'][:-int(24/options["dt"])], [int(24/options["dt"]*7),52], order = 'F')
    }
# 24/dt*7 = periods in a week, 52 weeks in a year
weekly_PV_data = {
    'PV_out': np.reshape(PV_data['El_out'][:-int(24/options["dt"])], [int(24/options["dt"]*7),52], order = 'F')
}

# Creating cluster profiles as input to k-means
cluster_profiles = np.concatenate(
    [ weekly_demands['EE'], weekly_demands['Q'], weekly_demands['Cold'], weekly_PV_data['PV_out'] ]
    )

# Cluster generation
# class clusters():
#     def __init__(self, centroids, labels, weights):
#         self.centroids = centroids
#         self.labels = labels
#         self.weights = weights
# kmeans clusterung
kmeans_clusters = KMeans( n_clusters = options['n_clusters'] ).fit(np.transpose(cluster_profiles))
# Define dictionary for clusters, add here centroids and week labels (can use tuple in future so to avoid overwriting)
clusters = {
    'centroids': kmeans_clusters.cluster_centers_,
    'labels': kmeans_clusters.labels_,
    }
# Time period weights
temp_a = Counter(clusters["labels"])
weights_temp = np.empty((0,))
for i in range(options["n_clusters"]):
    weights_temp = np.append( weights_temp, temp_a[i]*np.ones( (1, int(24/options["dt"]*7)) ) )
clusters["weights"] = weights_temp
del temp_a, weights_temp

# %% Plot centroid profiles for the different clusters
fig_clusters = plt.figure()
typ_profiles = np.zeros((1,4)) # just for initiation
for ii in range(clusters['centroids'].shape[0]):
    temp = clusters['centroids'][ii,:].reshape(4,int(1/options['dt']*24*7))
    typ_profiles = np.concatenate([typ_profiles, np.transpose(temp)])
    for jj in range(4):
        plt.subplot(1,3,ii+1)
        plt.plot(temp[jj,:])
    plt.xticks( np.arange(0, 24*8, 24), list(range(1,8+1)) )
    plt.yticks( np.linspace(0, np.max(clusters['centroids']), 15 ))
    plt.grid(which='both')

# Add labels
plt.subplot(1,3,1)
plt.ylabel('Power [kW]')
plt.subplot(1,3,2)
plt.xlabel('Days')
plt.title('Typical weeks demand profiles')
plt.subplot(1,3,1)
plt.legend(['Electricity', 'Heat', 'Cold', 'PV out'], loc='upper left')
plt.show()

# PV electricity generated plot
fig_PVout = plt.figure()
for ii in range(clusters['centroids'].shape[0]):
    temp = clusters['centroids'][ii,:].reshape(4,int(1/options['dt']*24*7))
    plt.subplot(1,3,ii+1)
    plt.plot(temp[3,:], color='r')
    plt.xticks( np.arange(0, 24*8, 24), list(range(1,8+1)) )
    plt.yticks( np.linspace(0, 150, 15 ))
    plt.grid(which='both')

# Add labels
plt.subplot(1,3,1)
plt.ylabel('Electricity generated [kW/m2]')
plt.subplot(1,3,2)
plt.xlabel('Days')
plt.title('PV typical weeks profiles')
plt.show()

# Link together the typical weeks so to have a unique typical profile
typ_profiles = typ_profiles[1:,:]
