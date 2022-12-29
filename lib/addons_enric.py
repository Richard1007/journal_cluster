import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.global_var import *


def vis_2d_influencer_with_scaled_dots(df_inf,df_brand,df_cluster,brandname, influencer, dim_list,xy_lims=None, scaledim="typicality"):
    from lib.hard_launch import set_up_quadrant_spine, get_membership_col
    membership_col = get_membership_col(df_cluster)
    
    # same as vis_2d_influencer but now sizing by typicality
    nrow = (len(dim_list)-1)//3+1
    ncol = 3
    fig,axes = plt.subplots(nrow,ncol, figsize=(ncol*5,nrow*4), facecolor="white")
    axes = axes.flatten()
    
    # series to plot
    s_inf_mem     = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']!=influencer)][membership_col]
    s_inf_name    = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']!=influencer)]['handle']
    s_inf_special = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']==influencer)][membership_col]
    s_treat = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='yes')][membership_col]
    s_control = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='no')][membership_col]    

        
    # typicality/scaledim sizes
    from sklearn.preprocessing import MinMaxScaler        
    s_inf_sizes          = df_inf[(df_inf['brand']==brandname)][scaledim]
    inf_names            = df_inf[(df_inf['brand']==brandname)]['handle']
    typicality           = MinMaxScaler().fit_transform(s_inf_sizes.values.reshape(1,-1).T)
    focus_inf_typicality = typicality[np.where(inf_names == influencer)[0][0]]
    other_inf_typicality = typicality[np.where(inf_names != influencer)[0]] 

    
    # this version adds sizing
    def vis_2d_membership(df,df_cluster,dim1,dim2,text,c='b',s=30,marker='o',ax=None):
        label1 = df_cluster.loc[df_cluster['central_word'] == dim1, 'label'].values[0]
        label2 = df_cluster.loc[df_cluster['central_word'] == dim2, 'label'].values[0]
        col1,col2 = membership_col[label1], membership_col[label2]
        x,y = df[col1], df[col2]

        if(type(s) != int):
            s = 10 + 15*s
            s = s**2

        ax.scatter(x,y,c=c,s=s,marker=marker,alpha=.5)
        xmin, xmax = ax.get_xlim()
        xdisplace = (xmax-xmin)*0.08
        for xi,yi,texti in zip(x,y,text):
            ax.text(xi+xdisplace,yi,texti,verticalalignment="center",alpha=.8)
        return ax
 

    for i in range(len(dim_list)):
        dim1,dim2 = dim_list[i]
        ax = axes[i]
        
        # set ax lim, use the same lim as the plot for Control group brand
        if(xy_lims is not None):
            xmin,xmax,ymin,ymax = xy_lims[i]
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
     
        # plot points
        vis_2d_membership(s_inf_mem,df_cluster,dim1,dim2,s_inf_name,c='b',ax=ax,s=other_inf_typicality) # influencer
        vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='orange',s=focus_inf_typicality,ax=ax) # special influencer
        vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
        vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax,marker="o") # control
        
        # set up spine
        set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig


def clean_clusters(df_cluster):
    for k,v in blacklist.items():
        idx        = np.where(df_cluster.central_word == k)[0][0]
        words      = df_cluster.at[idx,"word"]
        words      = set(words) - set(v)    
        df_cluster.at[idx,"word"] = words

    return df_cluster



def plot_concept_change(brandname, df_brand, df_inf, df_cluster, topk=10):
    from lib.hard_launch import set_up_quadrant_spine, get_membership_col
    membership_col = get_membership_col(df_cluster)
    
    fig, axs = plt.subplots(1,3, figsize=(14,5), facecolor="white")

    # the series to plot ...
    s_treat       = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='yes')][membership_col]
    s_control     = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='no')][membership_col]
    s_influencers = pd.DataFrame(df_inf[(df_inf['brand']==brandname)][membership_col].mean(axis=0)).T

    # calculate top 10 labels for brand based on control condition
    idxs = np.argsort(s_control.values.ravel())[::-1][:topk][::-1]
    lbls = df_cluster.central_word.values[idxs]    


    xlim = (0, max(s_control.max().max(), s_treat.max().max()))
    ref_vals = s_control.values[0,idxs].ravel()

    def plot_bar_membership(ax, series, title, color="#555555"):
        vals = series.values[0,idxs].ravel()

        ax.barh(lbls, vals,color=color)
        ax.barh(lbls, ref_vals, color="black", alpha=0.2)

        ax.set_title(title)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlim(xlim)

    # in general
    plot_bar_membership(axs[0], s_control, "Control")
    plot_bar_membership(axs[1], s_treat, "Treatment")
    plot_bar_membership(axs[2], s_influencers, "Mean Influencer") # ~ this should be the same but there's a slight discrepancy ...

    # for non-typical influencers
    influencer_typicality = df_inf[(df_inf['brand']==brandname)].typicality
    s_typical  = df_inf[(df_inf['brand']==brandname)][influencer_typicality >= influencer_typicality.median()]
    s_atypical = df_inf[(df_inf['brand']==brandname)][influencer_typicality <= influencer_typicality.median()]

    plot_bar_membership(axs[0], s_control, "Control")
    plot_bar_membership(axs[1], pd.DataFrame(s_atypical.mean()).T, "Less Typical",color="red")
    plot_bar_membership(axs[2], pd.DataFrame(s_typical.mean()).T, "More Typical", color="green")

def plot_concept_change_manybrands(brandname, df_brand, df_inf, df_cluster, topk=10):
    from lib.hard_launch import get_membership_col
    membership_col = get_membership_col(df_cluster)
    fig, axs = plt.subplots(1,3, figsize=(14,5), facecolor="white")

    # the series to plot ...
    inbrand       = [brand in brandname for brand in df_brand['brand'].values]
    infinbrand    = [brand in brandname for brand in df_inf['brand'].values]
    
    s_treat       = pd.DataFrame(df_brand[inbrand & (df_brand['treatment']=='yes')][membership_col].mean(axis=0)).T
    s_control     = pd.DataFrame(df_brand[inbrand & (df_brand['treatment']=='no')][membership_col].mean(axis=0)).T
    s_influencers = pd.DataFrame(df_inf[infinbrand][membership_col].mean(axis=0)).T

    # calculate top 10 labels for brand based on control condition
    idxs = np.argsort(s_control.values.ravel())[::-1][:topk][::-1]
    lbls = df_cluster.central_word.values[idxs]    


    xlim = (0, max(s_control.max().max(), s_treat.max().max()))
    ref_vals = s_control.values[0,idxs].ravel()

    def plot_bar_membership(ax, series, title, color="#555555"):
        vals = series.values[0,idxs].ravel()

        ax.barh(lbls, vals,color=color)
        ax.barh(lbls, ref_vals, color="black", alpha=0.2)

        ax.set_title(title)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlim(xlim)

    # in general
    #plot_bar_membership(axs[0,0], s_control, "Control")
    #plot_bar_membership(axs[0,1], s_treat, "Treatment")
    #plot_bar_membership(axs[0,2], s_influencers, "Mean Influencer") # ~ this should be the same but there's a slight discrepancy ...

    # for non-typical influencers
    
    influencer_typicality = df_inf[infinbrand].typicality
    s_typical  = df_inf[infinbrand][influencer_typicality >= influencer_typicality.median()]
    s_atypical = df_inf[infinbrand][influencer_typicality <= influencer_typicality.median()]

    plot_bar_membership(axs[0], s_control, "Control")
    plot_bar_membership(axs[1], pd.DataFrame(s_atypical.mean()).T, "Less Typical",color="red")
    plot_bar_membership(axs[2], pd.DataFrame(s_typical.mean()).T, "More Typical", color="green")
    
    
    
def vis_2d_justone(dim1, dim2, df_inf,df_brand,df_cluster,brandname,  dim_list,xy_lims, scaledim="typicality"):
    from lib.hard_launch import get_membership_col
    membership_col = get_membership_col(df_cluster)

    from lib.hard_launch import set_up_quadrant_spine
    
    fig,ax = plt.subplots(1,1, figsize=(5,4), facecolor="white",dpi=100)
    
    # series to plot
    s_inf_mem     = df_inf[(df_inf['brand']==brandname)][membership_col]
    s_inf_name    = df_inf[(df_inf['brand']==brandname)]['handle']
    
    s_treat = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='yes')][membership_col]
    s_control = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='no')][membership_col]    

        
    # typicality/scaledim sizes
    from sklearn.preprocessing import MinMaxScaler        
    s_inf_sizes          = df_inf[(df_inf['brand']==brandname)][scaledim]
    inf_names            = df_inf[(df_inf['brand']==brandname)]['handle']
    typicality           = MinMaxScaler().fit_transform(s_inf_sizes.values.reshape(1,-1).T)
    other_inf_typicality = typicality

    
    # this version adds sizing
    def vis_2d_membership(df,df_cluster,dim1,dim2,text,c='b',s=30,marker='o',ax=None):
        label1 = df_cluster.loc[df_cluster['central_word'] == dim1, 'label'].values[0]
        label2 = df_cluster.loc[df_cluster['central_word'] == dim2, 'label'].values[0]
        col1,col2 = membership_col[label1], membership_col[label2]
        x,y = df[col1], df[col2]

        if(type(s) != int):
            s = 10 + 400*s

        ax.scatter(x,y,c=c,s=s,marker=marker,alpha=.5)
        xmin, xmax = ax.get_xlim()
        xdisplace = (xmax-xmin)*0.08
        for xi,yi,texti in zip(x,y,text):
            ax.text(xi+xdisplace,yi,texti,verticalalignment="center",alpha=.8)
        return ax
 


    # set ax lim, use the same lim as the plot for Control group brand
    #xmin,xmax,ymin,ymax = xy_lims[i]
    #ax.set_xlim(xmin,xmax)
    #ax.set_ylim(ymin,ymax)

    # plot points
    vis_2d_membership(s_inf_mem,df_cluster,dim1,dim2,s_inf_name,c='b',ax=ax,s=other_inf_typicality) # influencer
    #vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='orange',s=focus_inf_typicality,ax=ax) # special influencer
    vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
    vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax,marker="o") # control

    # set up spine
    set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig



def vis_2d_manybrands(dim1, dim2, df_inf,df_brand,df_cluster,brandname, influencer, xy_lims=None, scaledim="typicality"):
    from lib.hard_launch import get_membership_col
    membership_col = get_membership_col(df_cluster)
    
    from lib.hard_launch import set_up_quadrant_spine
    
    fig,ax = plt.subplots(1,1, figsize=(5,4), facecolor="white",dpi=100)
    
    # series to plot
    infinbrand    = [brand in brandname for brand in df_inf['brand'].values]
    brinbrand     = [brand in brandname for brand in df_brand['brand'].values]
    
    if(influencer is None or influencer == ''):
        influencer = df_inf[infinbrand].handle.iloc[0]
    
    s_inf_mem     = df_inf[infinbrand & (df_inf['handle']!=influencer)][membership_col]
    s_inf_name    = df_inf[infinbrand & (df_inf['handle']!=influencer)]['handle']
    s_treat       = df_brand[brinbrand & (df_brand['treatment']=='yes')][membership_col]
    s_control     = df_brand[brinbrand & (df_brand['treatment']=='no')][membership_col]    

        
    # typicality/scaledim sizes
    from sklearn.preprocessing import MinMaxScaler        
    s_inf_sizes          = df_inf[infinbrand][scaledim]
    inf_names            = df_inf[infinbrand]['handle']

    typicality           = MinMaxScaler().fit_transform(s_inf_sizes.values.reshape(1,-1).T)
    focus_inf_typicality = typicality[np.where(inf_names == influencer)[0][0]]
    other_inf_typicality = typicality[np.where(inf_names != influencer)[0]] 

    
    # this version adds sizing
    def vis_2d_membership(df,df_cluster,dim1,dim2,text,c='b',s=30,marker='o',ax=None):
        label1 = df_cluster.loc[df_cluster['central_word'] == dim1, 'label'].values[0]
        label2 = df_cluster.loc[df_cluster['central_word'] == dim2, 'label'].values[0]
        col1,col2 = membership_col[label1], membership_col[label2]
        x,y = df[col1], df[col2]

        if(type(s) != int):
            s = 1 + 15*s
            #https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
            s = s**2

        ax.scatter(x,y,c=c,s=s,marker=marker,alpha=.5)
        xmin, xmax = ax.get_xlim()
        xdisplace = (xmax-xmin)*0.08
        for xi,yi,texti in zip(x,y,text):
            ax.text(xi+xdisplace,yi,texti,verticalalignment="center",alpha=.8)
        return ax
 


    # set ax lim, use the same lim as the plot for Control group brand
    #xmin,xmax,ymin,ymax = xy_lims[i]
    #ax.set_xlim(xmin,xmax)
    #ax.set_ylim(ymin,ymax)

    # plot points

    vis_2d_membership(s_inf_mem,df_cluster,dim1,dim2,s_inf_name,c='b',ax=ax,s=other_inf_typicality) # influencer
    #vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='orange',s=focus_inf_typicality,ax=ax) # special influencer
    #vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
    #vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax,marker="o") # control
    
    # set up spine
    set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig



############################
## Word Clustering
############################
def cluster_words_hdbscan(word_list,k,nlp):
    from lib.hard_launch import get_embedding_matrix
    X                        = get_embedding_matrix(word_list,nlp)
    labels,inertia,centroids = sklearn_KMeans(X, k, seed=0)
    print(f"Kmeans clustering mean inertia = {inertia/len(X)}")
    
    df_word     = pd.DataFrame(data={'word':word_list,'embedding':X.tolist(),'label':labels})
    df_cluster  = pd.DataFrame(data={'label':list(range(k)), 'centroid':centroids.tolist()})
    df_cluster  = find_central_word(df_word,df_cluster)
    return df_word,df_cluster



# For plotting dendrograms you need this function first
# when specifying the #of clusters
# https://stackoverflow.com/questions/26851553/sklearn-agglomerative-clustering-linkage-matrix
def get_agglo_distances(X,model,mode='l2'):
    distances = []
    weights   = []
    
    
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5 
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d


        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append( wNew)
    return distances, weights


#######################################################
# Code to plot wordclouds for an influencer that is typical
# and atypical. You can specify to only show 2 dims or everything
def plot_inf_brand_cloud(brand, inf1, inf2, df, df_cluster, df_word, dims):
    from wordcloud import WordCloud

    def cluster_label(word):
        return df_cluster[[word in w for w in df_cluster.word]].label.values[0]

    def words_for_x(all_words,only_cluster=None):
        all_words = [item for sublist in all_words for item in sublist]
        all_words = [item for sublist in all_words for item in sublist]

        if(only_cluster is not None):
            all_words_filtered =[]
            for word in all_words:
                if(cluster_label(word) in only_cluster):
                    all_words_filtered.append(word)
            return all_words_filtered
        return all_words
    # all words for focal brand, that are in ctrl
    def words_for_brand(df, name, only_cluster=None):
        all_words = df[(df.brand == name) & (df.treatment=='no')][P1_lemma+P2_lemma].values
        return words_for_x(all_words,only_cluster)
    def words_for_influencer(df, handle, only_cluster=None):
        all_words = df[df.handle == handle][P1_lemma+P2_lemma].values
        return words_for_x(all_words,only_cluster)

    def vis_wordcloud(word_list,title,ax=None):
        from matplotlib import cm

        if(ax is None):
            ax = plt.figure().gca()
        '''Given central_word, show a wordcloud'''
        all_word_dict = dict(df_word[['word','count']].to_numpy())
        word_freq = {w:all_word_dict[w] for w in word_list}

        def my_tf_color_func():
            return my_tf_color_func_inner
        def my_tf_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            #return "rgb(%d, 255, 255)" % ( cluster_label(word))
            color = cm.tab20(cluster_label(word)/100)
            return f"rgb({255*color[0]:.0f},{255*color[1]:.0f},{255*color[2]:.0f})"



        wordcloud = WordCloud(width = 400, height = 200, random_state=1, background_color='white', collocations=False,
                              colormap=cm.inferno, color_func=my_tf_color_func)
        wordcloud.fit_words(word_freq)
        plt.imshow(wordcloud) 
        plt.axis("off")
        plt.title(title)

    if(dims is not None):
        allowed = [ cluster_label(dims[0]), cluster_label(dims[1])]
    else:
        allowed = None
    vis_wordcloud(words_for_brand(df,brand, allowed), "Brand")
    vis_wordcloud(words_for_influencer(df,inf1, allowed), "Typical")
    vis_wordcloud(words_for_influencer(df,inf2, allowed), "Atypical")


    
def censor_ips(df):
    bad_ips = ["73.177.78.145"]
    goodidx = [ not ip in bad_ips for ip in df.IPAddress]
    return df.loc[goodidx]



# Weighted clustering using Bayesian Guassian Mixture Model
def cluster_words(word_list,weights, k,nlp):
    print(f"Clustering all words with given weights into {k} clusters using spacy.")
    
    from lib.hard_launch import get_embedding_matrix, find_central_word
    # https://github.com/ktrapeznikov/dpgmm    
    #from sklearn.mixture import BayesianGaussianMixture 
    from lib.dpm.dpgmm import WeightedDPGMM
    X                        = get_embedding_matrix(word_list,nlp)

    clusterer = WeightedDPGMM(
        n_components=k, max_iter=100, verbose=1, random_state=0,
        init_params = 'kmeans',
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_distribution',
        #weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1
    )
    labels    = clusterer.fit_predict(X, sample_weight=weights)
    centroids = clusterer.means_                                # mu = cluster centers
    stdevs    = [np.trace(cv) for cv in clusterer.covariances_] # sigma = the trace of the individual covariance matrices

    #print(f"Clustering metric = {metric}")
    k = len(np.unique(labels))
    #k = len(centroids)
    print(k, len(centroids), len(stdevs), len(clusterer.weights_))
    df_word     = pd.DataFrame(data={'word':word_list,'embedding':X.tolist(),'label':labels}, index=word_list)
    df_cluster  = pd.DataFrame(data={
        'label':list(range(k)), 
        'centroid':centroids.tolist(),
        'stdev': stdevs,
        'weight': clusterer.weights_
    })
    df_cluster  = find_central_word(df_word,df_cluster)
    
    

    
    return df_word,df_cluster, clusterer


def get_jaccard_similarity_participant_vs_brand_words(df):
    def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    def calc_mem_dist(row):
        b = row.brand
        member_words = row[P1_lemma+P2_lemma].sum()
        return jaccard_similarity(member_words, brand_words[b])
    ctrl = df.treatment == 'no'
    brand_words = df[ctrl][P1_lemma+P2_lemma].set_index(df[ctrl].brand).sum(axis=1)
    brand_words = brand_words.groupby('brand').sum()
    
    df = df.copy()
    return df.apply(calc_mem_dist,axis=1)
# using the labels instead
def get_jaccard_similarity_participant_vs_brand_lbls(df):
    def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    def calc_mem_dist(row):
        b = row.brand
        member_words = row[P1_lemma+P2_lemma].sum()
        member_lbls  = words_to_lbls(member_words)
        return jaccard_similarity(member_lbls, brand_lbls[b])

    word2lbl = dict(zip(df_word.word, df_word.label))
    def words_to_lbls(words):
        return [word2lbl[w] for w in words]

    ctrl = df.treatment == 'no'
    brand_words = df[ctrl][P1_lemma+P2_lemma].set_index(df[ctrl].brand).sum(axis=1)
    brand_words = brand_words.groupby('brand').sum()
    brand_lbls  = brand_words.apply(words_to_lbls)
    
    df = df.copy()
    return df.apply(calc_mem_dist,axis=1)



def plot_radar(df, yrange = np.linspace(.12,.5,4)):
    from math import pi

    # ------- PART 1: Create background
    # number of variable
    categories = list(df.index)
    N          = df.shape[0]
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]

    # Initialise the spider plot
    plt.figure(facecolor="white",dpi=150)
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(yrange, [f"{y:.2f}" for y in yrange], color="grey", size=7)
    plt.ylim(0,yrange.max())
    angles += angles[:1]


    # ------- PART 2: Add plots
    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values  = df.iloc[:,0].values.flatten().tolist()
    values += values[:1]    
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[0])
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df.iloc[:,1].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.columns[1])
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
def calc_radar_data(df, df_word, df_cluster, brand, influencer, n_dim):
    """
        Given a brand name and an influencer, it makes a df of word counts (normalized)
        for the brand and the influencer. N_dim specifies the number of output dimensions
        cut off at whichever are the top n_dim for the influencer.
    """
    # get all relevant labels
    brand_words  = df.loc[(df.brand == brand) & (df.treatment == 'no')][P1_lemma+P2_lemma].sum().values.sum()
    brand_labels = df_word.set_index(df_word.word).loc[brand_words].label.values
    inf_words      = df.loc[(df.handle == influencer)][P1_lemma+P2_lemma].sum().values.sum()
    inf_labels     = df_word.set_index(df_word.word).loc[inf_words].label.values

    # calculate top counts
    ivalues, icounts = np.unique(inf_labels, return_counts=True)
    bvalues, bcounts = np.unique(brand_labels, return_counts=True)
    
    
    chosen_lbls, chosen_cnts = ivalues, icounts   # reference = influencer 
    #chosen_lbls, chosen_cnts = bvalues, bcounts    # reference = brand
    top_idx        = np.argsort(chosen_cnts)[::-1][:n_dim]
    top_lbls       = chosen_lbls[top_idx]

    included       = [i in top_lbls for i in ivalues]
    inf_top        = pd.Series(name="Influencer",
        index = ivalues[included], 
        data  = icounts[included] / np.sum(icounts),
    )
    included       = [b in top_lbls for b in bvalues]
    brand_top      = pd.Series(name="Control",
        index = bvalues[included],
        data  = bcounts[included] / np.sum(bcounts)
    )
    df_radar = pd.concat([inf_top,brand_top], axis=1)
    df_radar.index = df_cluster.iloc[df_radar.index].central_word.values
    df_radar = df_radar.fillna(0)
    
    # merging of similar categories, start these with a capital to 
    # avoid naming conflicts
    remap = {
        'Athleticism':["athletic","athlete","player","fitness","fit"],
        'Fashionable':["fashionable","stylish","fashionista"],
        "Upper_class":["upper_class","status"],
        "Cool":["cool","hip"]
    }

    for concept,subconcepts in remap.items():
        # check if it's relevant first ...
        if(np.sum([s in df_radar.index for s in subconcepts]) > 0):

            df_radar.loc[concept,:] = 0
            for subconcept in subconcepts:
                if(subconcept in df_radar.index):
                    df_radar.loc[concept,:] += df_radar.loc[subconcept]
                    df_radar.drop(subconcept,axis=0, inplace=True)
    
    df_radar.index = [i.lower() for i in df_radar.index]
    
    print("Final dimensios:", df_radar.shape)
    return df_radar
