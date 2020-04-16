# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# #%% [markdown]
# ## Clustering of unlabeled data with sklearn.
# ----
# 
# ### Each Clustering algorithm comes in two variants: 
# * a class, that implements the fit method to learn the clusters on train data, 
# * a function, that, given train data, returns an array of integer labels corresponding to the different clusters. 
# 
# ## Clustering performance evaluation
# ----
# Evaluating the performance of a clustering algorithm is not as trivial as counting the number of errors or the precision and recall of a supervised classification algorithm. 
# 
# * In particular any evaluation metric should not take the absolute values of the cluster labels into account but rather if this clustering define separations of the data similar to some ground truth set of classes or satisfying some assumption such that members belong to the same class are more similar than members of different classes according to some similarity metric
# 
# ## Adjusted Rand Index
# 
# - Given the knowledge of the ground truth class assignments labels_true and our clustering algorithm assignments of the same samples labels_pred, the adjusted Rand index is a function that measures the similarity of the two assignments, ignoring permutations and with chance normalization:
# 
# - Perfect labeling is scored 1.0; Bad (e.g. independent labelings) have negative or close to 0.0 scores. 
# 
# - If C is a ground truth class assignment and K the clustering, let us define  and  as:
#     * $a$, the number of pairs of elements that are in the same set in C and in the same set in K
#     * $b$, the number of pairs of elements that are in different sets in C and in different sets in K
#     The raw (unadjusted) Rand index is then given by:<br>
#     $\text{RI} = \frac{a + b}{C_2^{n_{samples}}}$ <br>
#     Where $C_2^{n_{samples}}$ is the total number of possible pairs in the dataset (without ordering)
# 
# ## Adjusted Mutual Information 
# - Given the knowledge of the ground truth if MIA is a function that measures the agreement of the two assignments, ignoring permutations
# - Perfect labeling is scored 1.0; Bad (e.g. independent labelings) has negative score 
# 
# ## Homogeneity, completeness and V-measure
# - All rely on the ground trith info 
# - **Homogeneity:** each cluster contains only members of a single class
# - **Completeness:** all members of a given class are assigned to the same cluster
# - Both are bounded below by $0.0$ and above by $1.0$ (higher is better)
# 
# - **V-measure**: the harmonic mean of Homogeneity and Completeness: $v = \frac{(1 + \beta) \times \text{homogeneity} \times \text{completeness}}{(\beta \times \text{homogeneity} + \text{completeness})}$
# 
# ## Silhouette Coefficient
# - Applicable when ground truth is not given 
# - Is defined for each sample and is composed of two scores:
#     * a: The mean distance between a sample and all other points in the same class.
#     * b: The mean distance between a sample and all other points in the next nearest cluster.
# 
# - The Silhouette Coefficient $s$ for a single sample is then given as: $s = \frac{b - a}{max(a, b)}$
# 
# ## Inertia
# - Applicable when ground truth is not given 
# - Sum of squared distances of samples to their closest cluster center
# 

# %%
## stdlib
from time import time

## PyPI
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# %% [markdown]
# #%% [markdown]
# ## Load data

# %%
## load dataset
df = pd.read_pickle("./dataFrames/testDF.pkl")
features = ["mjj", "dEtajj", "dPhijj", "dRjj","dPtjj",  "sumPtAllJets", "sumPtjj",  "vecSumPtAllJets",  "vecSumPtjj"]
target = ["isSignal"]
print(df.head())
print(df.describe())

## keep 20% for validation 
X_train, X_val, y_train, y_val = train_test_split(df[features].values, df[target].values, test_size=0.20, random_state=42)

print(X_train.shape)

# Get the numerical data only
# background_array = df[df['isSignal'] == 0].to_numpy()[:,3:].astype(float)
# signal_array = df[df['isSignal'] == 1].to_numpy()[:,3:].astype(float)

# %% [markdown]
# #%% [markdown]
# ### The k-Means clustering
# - tries to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.
# - the number of clusters should be specified 
# - The k-means algorithm divides a set of $N$ samples $X$ into $K$  disjoint clusters $C$, each described by the mean $\mu_j$ 
#  of the samples in the cluster. The means are commonly called the cluster “centroids”
# 
#  #### k-Means algorithm: 
#     1. choose the initial centroids, with the most basic method being to choose  samples from the dataset. 
#     2. assign each sample to its nearest centroid, i.e create new centroids by taking the mean value of all of the samples assigned to each previous centroid. 
#     3. calculate the difference between the old and the new centroids 
#     4. repeat until the centroids do not move significantly
# 
# * k-means might end up in local minimas depending on the initialization of the centroids
# * As a result, the computation is often done several times, with different initializations of the centroids. 
# * k-means++ initialization scheme ensures generally distant initial centroids
# 

# %%
## preprocessing 
data = scale(X_train)
sample_size = X_train.shape[0]
n_classes = 2
labels = y_train.ravel()

def bench_k_means(estimator, name, data):
    """ model benchmarking 
    """
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

bench_k_means(KMeans(init='k-means++', n_clusters=n_classes, n_init=200, n_jobs=-1),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_classes, n_init=20),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_classes).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_classes, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

#%% [markdown] 
## To perform PCA:
# - shift the means of all the columns to 0, so the data is centered around the origin
# - compute the covariance matrix, $\Sigma$ of the data
# - calculate the eigenvalues and (normalized) eigenvectors of  $\Sigma$

#%% 
def PCA(data, num_components=2):
    X = torch.from_numpy(data) # Create pytorch tensor
    X_mean = torch.mean(X, 0) # Calculate mean of all the columns
    X = X - X_mean.expand_as(X) # Broadcast to subtract the row of means from each row

    # Perform a singular value decomposition, and return the linear transform with
    # the original matrix and num_components eigenvectors
    U, S, V = torch.svd(torch.t(X))
    return torch.mm(X, U[:, :num_components])

#%%
num_components = 4

fig, ax = plt.subplots(num_components - 1, num_components - 1, figsize=(5*num_components, 5*num_components))

background_pca = PCA(background_array, num_components)
signal_pca = PCA(signal_array, num_components)

for component_1 in range(num_components):
    for component_2 in range(component_1 + 1, num_components):
            this_ax = ax[component_1, component_2 - 1]
            this_ax.scatter(background_pca[:, component_1], background_pca[:, component_2], alpha=0.5, label='background')
            this_ax.scatter(signal_pca[:, component_1], signal_pca[:, component_2], alpha=0.5, label='signal')
            this_ax.set_xlabel(f"PCA {component_1}", fontsize=14)
            this_ax.set_ylabel(f"PCA {component_2}", fontsize=14)
            this_ax.xaxis.set_tick_params(labelsize=14)
            this_ax.yaxis.set_tick_params(labelsize=14)
            
fig.suptitle("PCA for background vs. signal data", fontsize=20)
ax[0, 0].legend(fontsize=16)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
#plt.savefig("pca.png")
plt.show()


# %% [markdown]
# #%% [markdown]
# ---
# ## Visualize the results on PCA-reduced data
# 
# 
# 
# 

# %%
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=2)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# %%
from sklearn.cluster import MeanShift, estimate_bandwidth

# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=sample_size)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)



#%%
# #############################################################################
# Plot result
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# %%
