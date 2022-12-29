# Like get_embeddign_matrix bute removes first PC (SIF)
#def get_embedding_matrix_sif(word_list,nlp):
#    X = []
#    for word in word_list:
#        X.append(nlp(word).vector.tolist())
#        
#    X = np.array(X)
#    # now remove first PC to get SIF
#    from sklearn.decomposition import TruncatedSVD
#    svd = TruncatedSVD(n_components=1, random_state=0, n_iter=20)
#    svd.fit(X)    
#    svd = svd.components_
#
#    X = X - X.dot(svd.transpose()) * svd
#    return X  

# Like original function but allows custom clusterin algorithms
def cluster_words(word_list,k,nlp, algorithm):
    X                        = get_embedding_matrix(word_list,nlp)
    labels,extra,centroids = algorithm(X, k, seed=0)
    #print(f"Clustering metric = {metric}")
    
    df_word     = pd.DataFrame(data={'word':word_list,'embedding':X.tolist(),'label':labels})
    df_cluster  = pd.DataFrame(data={'label':list(range(k)), 'centroid':centroids.tolist(),'extra':extra})
    df_cluster  = find_central_word(df_word,df_cluster)
    return df_word,df_cluster

# default choice, works well with k = 75
def sklearn_KMeans(X, num_clusters, seed=0):
    kmeans = cluster.KMeans(n_clusters=num_clusters,random_state=seed,)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    return (labels,inertia,centroids)

# Difficult to tune correctly, comes up with a lot of non-clustered items
def cluster_hdbscan(X, num_clusters, seed=0):
    import hdbscan    
    from sklearn.metrics.pairwise import cosine_similarity
    
    #distance = cosine_similarity(X) #, metric='precomputed')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    clusterer.fit(X)
    
    return clusterer.labels_, None, np.random.randn(num_clusters)

# Intuitive for rest of analysis but clusters arent' that good ...
def cluster_gaussian(X, num_clusters, seed=0):
    from sklearn.mixture import GaussianMixture 
    clusterer = GaussianMixture(n_components = num_clusters, random_state =seed)
    clusterer.fit(X)
    return clusterer.predict(X), None, clusterer.means_

# Automatic selection by choosing Bayesian alternative
# Works quite nicely, just like the kNN but a bit better
def cluster_gaussian2(X, num_clusters, seed=0):
    from sklearn.mixture import BayesianGaussianMixture 
    clusterer = BayesianGaussianMixture(n_components = num_clusters, random_state =seed)
    clusterer.fit(X)        
    return clusterer.predict(X), clusterer.weights_, clusterer.means_

def cluster_agglo_NOTREADY():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib import pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    from lib.addons_enric import get_agglo_distances

    model = AgglomerativeClustering(n_clusters=20)
    model = model.fit(X)
    # This takes a while ... 
    # Subsample
    #idx = np.array(list(range(100)))
    #plotL = list(np.array(word_list)[idx])
    #plotX = X[idx,:]
    #distance, weight = get_agglo_distances(X,model)
    #linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
    #plt.figure(figsize=(20,20),dpi=200)
    #dendrogram(linkage_matrix,truncate_mode="level",p=3,orientation='right', distance_sort='descending', labels=word_list)
    #dendrogram(linkage_matrix,orientation='right', distance_sort='descending')#,labels=plotL)
    #plt.show()
    #plt.savefig(f'output/tree.png')
    
    
#untested, paste in notebook
def skree_plot():
    from IPython.display import display, clear_output
    from sklearn.metrics import silhouette_score

    # plot the inertia for different k values
    X = get_embedding_matrix(word_list,nlp)

    fig,axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].set_title("Inertia")
    axs[1].set_title("Silhouette")
    inertias,silhs,ks = [],[],[]
    for k in np.arange(5,200,5):   
        labels,inertia,_ = weighted_sklearn_KMeans(X, k)
        silh = silhouette_score(X , labels)

        inertias.append(inertia)
        ks.append(k)
        silhs.append(silh)


        axs[0].cla()
        axs[0].plot(ks,inertias)
        axs[1].cla()
        axs[1].plot(ks,silhs)

        display(fig)    
        clear_output(wait = True)
        
        
def weighted_sklearn_KMeans(X, num_clusters, seed=0):
    kmeans = cluster.KMeans(n_clusters=num_clusters,random_state=seed,)
    kmeans.fit(X, sample_weight=weights)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    return (labels,inertia,centroids)
