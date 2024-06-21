import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.kmedoids import kmedoids
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('valence_arousal_dataset.csv')

sns.scatterplot(x='valence', y='energy', data = df).set(title='Energy vs Valence of All songs')

distortions = []
num_clusters = range(1, 20)

for i in num_clusters:
    cluster_centers, distortion = kmeans(df[['valence', 'energy']], i)
    distortions.append(distortion)
    
# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
elbow = sns.factorplot(x='num_clusters', y='distortions',data = elbow_plot, size=9).set(title='Elbow Plot - Within-cluster sum of squares (WSS) vs k', xlabel='Number of cluster (k)', ylabel='Within Clusters SSE (WSS)/Distortions')

# Generate cluster centers
cluster_centers, distortion = kmeans(df[['valence','energy']], 6)

# Assign cluster labels
df['k-means_moods'], distortion_list = vq(df[['valence','energy']], cluster_centers)

mood_centroids = {'depressed':[0.19066008, 0.21390282],'misery':[0.29176613, 0.55255357], 'contentment': [0.67930098, 0.46349509], 'distressed':[0.21726171, 0.89302689], 'excitement':[0.83923924, 0.80557057], 'energetic':[0.53893319, 0.83543712]}

# Plot clusters
sns.scatterplot(x='valence', y='energy', hue='k-means_moods', palette='bright', data = df).set(title="Moods clusters with K-means, k* = 6")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='o', s = 100)
#plt.legend(title="Moods", loc=4, fontsize='small', labels=['0 - Depressed', '1 - Energetic', '2 - Misery', '3 - Excitement', '4 - Distressed', '5 - Contentment'])
plt.show()

# Create instance of K-Medians algorithm.
K = 6
df_position = df[['valence', 'energy']]
initial_medians = [[0.19066008, 0.21390282], [0.29176613, 0.55255357], [0.67930098, 0.46349509], [0.21726171, 0.89302689], [0.83923924, 0.80557057], [0.53893319, 0.83543712]]
kmedians_instance = kmedians(df_position.values.tolist(),initial_medians)

# Run cluster analysis and obtain results.
kmedians_instance.process()
cluster_instances = kmedians_instance.get_clusters() # index of data points in each cluster
centers = np.array(kmedians_instance.get_medians())

clusters=np.zeros(len(df_position.index)) # initialize the membership of each point
for k in range(K):
    clusters[cluster_instances[k]]=k # label cluster membership for each point
    
sns.scatterplot(x=df['valence'], y=df['energy'], hue=clusters, palette='bright').set(title="Mood clusters with K-Medians, k* = 6")
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=100)
plt.show()

# Set random initial medoids.
initial_medoids = [1, 50, 75, 90, 2, 45]
df_valence_energy = df[['valence', 'energy']]

# Create instance of K-Medoids algorithm.
kmedoids_instance = kmedoids(df_valence_energy.values, initial_medoids)
kmedoids_instance.process()

# Run cluster analysis and obtain results.
cluster_instances = kmedoids_instance.get_clusters() # index of data points in each cluster
center_index = kmedoids_instance.get_medoids()
centers=df_position.values[center_index,:]

clusters=np.zeros(len(df_position.index)) # initialize the membership of each point
for k in range(K):
    clusters[cluster_instances[k]]=k # label cluster membership for each point

sns.scatterplot(x=df['valence'], y=df['energy'], hue=clusters, palette='bright').set(title="Mood clusters with K-Medoids, k* = 10")
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=100);
plt.show()

def distance(mood1, mood2):
    mean_valence = np.average([mood_centroids[mood1][0], mood_centroids[mood2][0]])
    mean_energy = np.average([mood_centroids[mood1][1], mood_centroids[mood2][1]])
    
    selected_moods = df.loc[df['k-means_moods_label'].isin([mood1, mood2])]
    selected_moods['euclidian_distance'] = ((selected_moods['valence'] - mean_valence)**2 + (selected_moods['energy'] - mean_energy)**2)**(1/2)
    
    #Select 1 song from 10 songs that have closest Euclidian Distance
    selected_track = selected_moods.sort_values('euclidian_distance').iloc[0:10]['track_name'].sample(n=1)
    
    return selected_track

distance('excitement', 'energetic')
mood_centroids
moods = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised']
mood_df = pd.DataFrame()

for main in moods:
    mood_df['main_mood'] = main
    for secondary in moods:
        mood_df['secondary_mood'] = secondary

data = [['Angry', 'none', 'Distressed', 'Energetic', 'Contentment', 'none'], 
        ['Disgust', 'none', 'Distressed', 'none', 'Contentment', 'none'],
        ['Fear', 'none', 'Distressed', 'Misery', 'Contentment', 'Excitement'],
        ['Happy', 'none', 'Excitement', 'none', 'Excitement', 'none'],
        ['Sad', 'none', 'Depressed', 'none', 'Contentment', 'Excitement'],
        ['Surprised', 'Energetic', 'none', 'Misery', 'Contentment', 'none'],
        ['Neutral','Energetic','Contentment',]
       ]
mood_df = pd.DataFrame(data = data, columns=['main_mood', 'secondary_mood', 'same_cluster1', 'same_cluster2', 'different_cluster1', 'different_cluster2'])

df.loc[df['k-means_moods'] == 0, 'moods_label'] = 'Distressed'
df.loc[df['k-means_moods'] == 1, 'moods_label'] = 'Excitement'
df.loc[df['k-means_moods'] == 2, 'moods_label'] = 'Misery'
df.loc[df['k-means_moods'] == 3, 'moods_label'] = 'Energetic'
df.loc[df['k-means_moods'] == 4, 'moods_label'] = 'Depressed'
df.loc[df['k-means_moods'] == 5, 'moods_label'] = 'Contentment'
df.to_csv('valence_arousal_dataset_labeled.csv')
