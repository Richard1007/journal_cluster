############################
'''
╔╗─╔╗──────╔╗╔╗────────────╔╗
║║─║║──────║║║║────────────║║
║╚═╝╠══╦═╦═╝║║║╔══╦╗╔╦═╗╔══╣╚═╗
║╔═╗║╔╗║╔╣╔╗║║║║╔╗║║║║╔╗╣╔═╣╔╗║
║║─║║╔╗║║║╚╝║║╚╣╔╗║╚╝║║║║╚═╣║║║
╚╝─╚╩╝╚╩╝╚══╝╚═╩╝╚╩══╩╝╚╩══╩╝╚╝
'''
############################
import re
import csv
import spacy
import enchant
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import Counter
from pandas import ExcelWriter
from scipy.stats.stats import pearsonr
from sklearn import cluster
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from lib.global_var import *
from scipy.spatial import distance
from wordcloud import WordCloud

#********************
# (In a R shell run): install.packages("crossmatch")
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import BoolVector, FloatVector
from rpy2.robjects import r
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_distances
importr('crossmatch')




############################
## Data Preprocessing 
############################
def load_data(path):
    '''
    Remove extra starting rows and save as tmp.csv
    Replace some text data with numerical values
    '''
    # delete the 1st row from the original file
    with open(path, 'r') as inp, open('cache/tmp.csv', 'w') as out:
        writer = csv.writer(out)
        n = 0
        for row in csv.reader(inp):
            if n!=0:
                writer.writerow(row)
            n+=1
    df = pd.read_csv('cache/tmp.csv')

    # replace some text to numbers
    adj_cols = P3+brand_conn+post_conn+intention_to_sell+\
    [idea_change,typicality,user_similarity,resp_conf,follow_likelihood,\
    sponsor_by,featured_product_typicality]  # cols need to be adjusted
    df[adj_cols] = df[adj_cols].replace(regex=r'\D*', value='').astype(float)
    
    df = df.rename(columns=rename_col)
    return df

def remove_anomaly(df):
    '''To add more anomaly detection rules here
    return a df with abnormal rows deleted
    '''
    # DistributionChannel = anonymous
    # IP_block = 0    
    df = df[(df['DistributionChannel']=='anonymous') & (df['IP_block']==0)]
    # keep responses with >=8 non-repetitive answers
    print(f"Original data have {df.shape[0]} rows.")
    df = df[df[P1+P2].apply(set,axis=1).apply(len)>=8]
    print(f"After deletion of rows with too few non-repetitive answers: {df.shape[0]} rows")
    return df

def series_to_matrix(s):
    '''series of vector => 2d matrix
    arg:
        s - np.Series
    '''
    return np.array(s.apply(list).to_list())


############################
## Text Processing 
############################
def process_text(df,nlp):
    '''
    args:
        df  - loaded participant level df
        nlp - loaded number-batch model
    return:
    df[pd.DataFrame] - processed participant level dataframe
    invalid_word[set] - set of invalid words
    val_word_cnt[Counter] - Counter of valid words
    corrected_words[pd.DataFrame] - df of words before and after correction
    '''
    def preprocess(text):
        '''
        Lowercase, lemmatize, remove stopwords & len(word)<3
        return a list of lemmatized words
        '''
        if pd.isna(text):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z\s]','',text)
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        lemmas = []

        for word,tag in tags:
            if tag[0]=='A':
                pos='a'
                lemma = lemmatizer.lemmatize(word,pos)                
            # DONT LEMMATIZE TO AVOID MAPPING OF ACCEPT-> ACCEPTED?            
            #elif tag[0]=='V':
            #    pos='v'
            elif tag[0]=='R':
                pos='r'
                lemma = lemmatizer.lemmatize(word,pos)                    
            else:
                pos='n'
            lemma = lemmatizer.lemmatize(word,pos)
            
            if lemma not in english_stopwords and len(lemma)>2:
                lemmas.append(lemma)
        return lemmas

    def correct_lemma_words(row):
        '''Use PyEnchant to correct words if the word has appeared elsewhere 
        (i.e. exists in val_word_cnt)
        '''
        for col in P1_lemma+P2_lemma:
            word_list = row[col].copy()

            for word in word_list:
                vec = nlp(word).vector            
                
                if all(v == 0 for v in vec): # invalid
                    before = word
                    after = []

                    # remove invalid word from the blank
                    tmp = row[col].remove(word)
                    # get the best suggestion (s can be a phrase of multiple words)
                    s = d.suggest(word)
                    if len(s)>0:
                        s=s[0]
                    else:
                        s=''
                    # check whether each word has appeared elsewhere
                    for w in s.split():
                        if w in val_word_cnt:
                            row[col].append(w)
                            after.append(w)
                    
                    corrected_words.append({'before':before,'after':' '.join(after)})

        return row

    from functools import reduce
    def get_avg_embeddings(word_list):
        #print("get_avg:",word_list)
        '''
        Given a list of words, return avg embedding.
        Applied on every survey entry
        '''
        # First we try to do the word in one go ...        
        if(len(word_list) > 1):       
            # can't do spaces or dashes
            wordgroups = ["".join(word_list) ]
            for attempt in wordgroups:
                vec  = nlp(attempt).vector
                if(not all(v == 0 for v in vec)):
                    val_word_cnt[attempt]+=1
                    return vec
        
        # If we're here it didn't work, try mean embeddign instead
        embeddings = []
        for word in word_list:
            vec  = nlp(word).vector
            if all(v == 0 for v in vec):
                invalid_word.add(word)
            else:
                val_word_cnt[word]+=1
                embeddings.append(vec)
                
        if len(embeddings)==0:
            return float('nan')
        return np.array(embeddings).mean(axis=0)

    # we remove these words because they don't add any meaning
    blacklist = ['well','somewhat','look','something','thing','sort','know','liker','maybe','none','everyone','dont','yes','idk','always','fairly','everybody','nothing','someone','quite','slightly','okay','sorry','belive','absolutely','expect','happen','seem','alright','simply','really','alot','believe','seriously','totally','kinda','possible','one','name','really','person','reply']
    def remove_censored(row):
        words = []
        for i, word in enumerate(row):
            if(not word in blacklist):
                words.append(word)
        return words
    # checks if the bigram exists, if it does we replace it in the dataset
    def replace_bigrams(row):
        if(len(row) <= 1):
            return row
        newword = "".join(row)
        vec  = nlp(newword).vector        
        if(not all(v == 0 for v in vec)):
            #print(f"replacing {row} with {newword}")
            return [newword]
        return row
    
    def remove_dup(row):
        '''Change later words to nan if the participant has entered the same word for previous blanks on this page'''
        col_list = row.keys()
        for i in range(len(col_list)):
            col2 = col_list[i]
            for j in range(i):
                col1 = col_list[j]
                try:
                    if set(row[col2])==set(row[col1]):
                        row[col2]=[]
                except:# ignore cells containing nan value
                    pass
        return row

    # Text process
    invalid_word = set() # words not recognized by ConceptNet
    val_word_cnt = Counter()
    corrected_words = []

    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words("english"))
    d = enchant.Dict("en_US")
   
    # text ==> list of lemmatized words
    df[P1_lemma+P2_lemma] = df[P1+P2].applymap(preprocess)
    
    # lemma --> nonblacklist lemma
    df[P1_lemma+P2_lemma] = df[P1_lemma+P2_lemma].applymap(remove_censored)       
    df[P1_lemma+P2_lemma] = df[P1_lemma+P2_lemma].applymap(replace_bigrams)
    
    # remove infrequent words -> slow implementations  
    (uq, cnts) = np.unique(output[P1_lemma+P2_lemma].values.sum(), return_counts=True)

    def remove_infrequent(row):
        return row
    df[P1_lemma+P2_lemma] = df[P1_lemma+P2_lemma].applymap(remove_infrequent)

    
    # list of lemmatized words ==> list of word embeddings 
    # (this step also generates set of valid and invalid words)
    df[P1_emb+P2_emb] = df[P1_lemma+P2_lemma].applymap(get_avg_embeddings)
    # correct spelling of lemmatized words
    df[P1_lemma+P2_lemma] = df[P1_lemma+P2_lemma].apply(correct_lemma_words,axis=1) 

    # remove repetitive words
    df[P1_lemma+P2_lemma] = df[P1_lemma+P2_lemma].apply(remove_dup,axis=1)
    # recalculate average embeddings with non-repetitive and corrected words
    df[P1_emb+P2_emb] = df[P1_lemma+P2_lemma].applymap(get_avg_embeddings)

    # nan ratio
    na_ratio = df[P1_emb+P2_emb].isna().to_numpy().sum()/(df[P1_emb+P2_emb].shape[0]*df[P1_emb+P2_emb].shape[1])
    print(f"Unrecognized word ratio (# nan / # cells): {na_ratio:.2%}")
    
    return df,invalid_word,val_word_cnt,pd.DataFrame(corrected_words)


############################
## Func to Calculate Participant Level Quantities
############################
def construct_participant_level_df(df):
    '''
    Adjust some participant level survey data
    '''
    df = df.copy()
    df[P3_adjust]         = df[P1_emb+P2_emb+P3].apply(adjust_association,axis=1)  # adjust association rating for invalid response to nan
    df['mean_emb']        = df[P1_emb+P2_emb].apply(avg_vec, axis=1)
    df['avg_association'] = df[P3_adjust].apply(row_avg_association,axis=1)  # participant average association score
    df['conn_with_brand'] = df[brand_conn].mean(axis=1)  # participant can relate to the brand
    df['conn_with_inf']   = df[post_conn].mean(axis=1)   # participant can relate to the influencer
    df['intentionality']  = df[intention_to_sell].mean(axis=1)
    return df

def avg_vec(row):
    '''Get the average vector of each row. 
    Return nan if no valid blank-level embedding
    arg:
        row - each cell contains a np vector
    '''
    res = np.array([val.tolist() for val in row.values if not type(val)==float])
    if len(res)==0:
        return np.nan
    else:
        return res.mean(axis=0)
    
def get_vec_var(row):
    '''
    Given a row of vectors, return 'vector variance', 
    i.e. avg distance of each vector agains their mean vec
    arg:
        row - each cell contains a np vector
    '''
    X = np.array([val.tolist() for val in row.values if not type(val)==float])
    if len(X)==0:
        return np.nan
    norm = np.sqrt(np.sum((X - X.mean(axis=0))**2, axis=1))
    return norm.mean()

def adjust_association(row):
    '''
    If embedding is nan, change corresponding association rating to nan
    '''
    for idx,col in enumerate(P1_emb+P2_emb):
        if type(row[col])==float:
            row[P3[idx]]=np.nan
    return row[P3]

def row_avg_association(row):
    '''Calculate avg association score ignoring nan values'''
    if np.isnan(row.values).all():
        return np.nan
    return np.nanmean(row.values)

def control_group_brand_avg(df):
    '''Return a series of control group average embeddings of each brand
    Brand avg embedding is calculated by:
    1. based on participant level average embedding
    2. Calculate control group mean vec of each brand
    '''
    def group_vec_avg(group):
        return group.sum()/group.count()
    C_brand_mean = df[df[treatment]=='no'].groupby(brand)[mean_emb].apply(group_vec_avg)
    C_brand_mean.name='C_brand_mean'
    return C_brand_mean

def dist_between_2_col(row):
    '''Given row of 2 cells with vectors, output distance between the vectors.'''
    assert len(row)==2
    return np.linalg.norm(row[0]-row[1])

def vis_pearsonr(s1,s2,ax=None):
    '''input 2 np.Series, visualize the correlation
    args: s1,s2 - 2 Series with nan values removed.
    return:
        (corr,p,ax)
    '''
    corr,p = pearsonr(s1,s2)
    print(f"{s1.name}, {s2.name} || Correlation Coefficient = {corr}, pval={p}")
    ax = sns.regplot(x=s1, y=s2,ax=ax)
    return (corr,p,ax)

def get_membership_dist_from_C(df, df_brand, metric='euclidean'):
    '''For each participant, calculate its distance from C group mean membership.
    args:
        df - participant level df
        df_brand - brand level df
        metric[str]: {'euclidean','cosine'}
    return:
        pd.Series[n_participant,]
            Each participant's distance from C group mean
    '''
    def calc_mem_dist(row):
        b = row[brand]
        C_mean = df_brand[(df_brand[brand]==b) & (df_brand[treatment]=='no')][membership_col].to_numpy().flatten().astype(np.float64) 
        par_vec = row[membership_col].to_numpy().flatten().astype(np.float64)
        
        if metric=='euclidean':
            dist = np.linalg.norm(par_vec-C_mean)
        elif metric=='cosine':
            dist = distance.cosine(par_vec,C_mean)
        return dist
    
    df = df.copy()
    return df.apply(calc_mem_dist,axis=1)

def lr(df,verbose=False):
    '''Linear regression
    args:
        df - first n-1 cols: independent variables
             the last col: dependent variables
        verbose - whether to print regression summary table
    return:
        params(np.array) - array of coefficients. The first one is for the intercept
        pvals(np.array) - array of pvals. The first one is for the intercept
    '''
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2).fit()
    
    if verbose:
        print(est.summary())    
    pvals = est.pvalues
    params = est.params
    return (params,pvals)


############################
## Func to Calculate Brand Level Quantities
############################
# def get_brand_avg_vec


############################
## Word Clustering
############################
def cluster_words(word_list,k,nlp):
    '''Clustering on valid words
    args:
        word_list[List] - list of valid words
        k - number of clusters
        nlp - number-batch model
    return:
        df_word[word,]  - word level df. Columns are:
            * embedding - word embedding
            * label     - Each word's cluster label
        df_cluster[cluster,] - cluster level df. Columns are:
            * label: cluster label
            * centroid: cluster centroid
            * central_word: word in cluster that is closest to cluster centroid
            * word[list(str)]: list of words within the cluster
            * embedding[list[list(float)]]: list of all word embeddings in the cluster
    '''
    X                        = get_embedding_matrix(word_list,nlp)
    labels,inertia,centroids = sklearn_KMeans(X, k, seed=0)
    print(f"Kmeans clustering mean inertia = {inertia/len(X)}")
    
    df_word     = pd.DataFrame(data={'word':word_list,'embedding':X.tolist(),'label':labels})
    df_cluster  = pd.DataFrame(data={'label':list(range(k)), 'centroid':centroids.tolist()})
    df_cluster  = find_central_word(df_word,df_cluster)
    return df_word,df_cluster

def get_embedding_matrix(word_list,nlp):
    '''
    arg:
        word_list[list(str)]: list of words
    return:
        np.array(word,embedding)
    '''
    X = []
    for word in word_list:
        X.append(nlp(word).vector.tolist())
    return np.array(X)

def sklearn_KMeans(X, num_clusters, seed=0):
    kmeans = cluster.KMeans(n_clusters=num_clusters,random_state=seed,)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    return (labels,inertia,centroids)

def find_central_word(df_word,df_cluster):
    '''Find central word for each cluster
    args:
        df_word: word level df
        df_cluster: cluster_level df
    return:
        cluster level df
            * label: cluster label
            * centroid: cluster centroid
            * central_word: word in cluster that is closest to cluster centroid
            * word[list(str)]: list of words within the cluster
            * embedding[list[list(float)]]: list of all word embeddings in the cluster
    '''
    
    def get_c_word(row):
        '''Find the central word which is the nearest to the cluster centroid'''
        dist = np.linalg.norm(np.array(row['embedding'])-np.array(row['centroid']), axis=1)  # (n_word in cluster, ) distance of each word against cluster centroid
        return row['word'][np.argmin(dist)]
        
    # group words to list for each cluster
    res              = pd.DataFrame(df_word.groupby('label')['word'].apply(list))
    res['embedding'] = pd.DataFrame(df_word.groupby('label')['embedding'].apply(list))
    res = res.reset_index()
    res = res.merge(df_cluster, left_on='label', right_on='label')
    
    # find central_word for each cluster
    res['central_word'] = res.apply(get_c_word,axis=1)
    return res

def vis_cluster_wordcloud(df_word,df_cluster,cword):
    '''Given central_word, show a wordcloud'''
    word_list = df_cluster[df_cluster['central_word']==cword]['word'].values[0]
    all_word_dict = dict(df_word[['word','count']].to_numpy())
    word_freq = {w:all_word_dict[w] for w in word_list}
    
    wordcloud = WordCloud(width = 400, height = 200, random_state=1, background_color='white', collocations=False)
    wordcloud.fit_words(word_freq)
    plt.imshow(wordcloud) 
    plt.axis("off")


############################
## Cluster membership
############################
def calc_word_membership(df_word,df_cluster,dist_func):
    '''Calcuate distance of words against each centroid.
    Append distances as cols to df_word.
    '''
    df_word = df_word.copy()
    for i in range(len(df_cluster)):
        centroid = np.array(df_cluster[df_cluster['label']==i]['centroid'].tolist()[0])
        df_word['dist_from_centroid_'+str(i)] = df_word['embedding']\
                                                .apply(np.array)\
                                                .apply(lambda vec: dist_func(vec,centroid))
    return df_word

def calc_part_membership(df,df_word):
    '''Calc participant membership by averaging word membership.
    Add membership col to df.
    '''
    def get_avg_membership(row):
        words = [w for blank in row[P1_lemma + P2_lemma].values for w in blank]
        dist_matrix = []
        for w in words:
            dist_matrix.append(df_word[df_word['word']==w][membership_col].to_numpy())
        mean_dist = np.concatenate(dist_matrix,axis=0).mean(axis=0)
        return pd.Series(mean_dist)
    
    df = df.copy()
    k = df_word['label'].nunique()
    membership_col = ['dist_from_centroid_'+str(i) for i in range(k)]
    df[membership_col] = df.apply(get_avg_membership,axis=1)
    return df

def calc_mem_var(df,df_word):
    '''Calculate the membership variation for each participant'''
    def func(row):
        avg_mem  = row[membership_col].to_numpy().astype(float)
        val_word = row[P1_lemma+P2_lemma].apply(Counter).sum().keys()
        distances = []
        
        # calculate cosine distance of each word membership score with participant's avg membership score
        for w in val_word:
            if w=='beautyful':
                print(row[['ResponseId']+P1_lemma+P2_lemma])
            w_mem = word_mem_series[w]
            distances.append(distance.cosine(w_mem,avg_mem))
        
        # averaging cosine distances
        return sum(distances)/len(distances)
            
    word_mem_series = df_word.set_index('word')[membership_col].apply(lambda row: row.to_numpy(), axis=1)
    return df.apply(func,axis=1)


############################
## Crossmatch Test
############################
def crossmatch_2_groups(X,Y,metric='euclidean'):
    '''Given 2 groups of multi-dimensional vectors, conduct crossmatch test.
    Distance metric: Euclidean TODO: other metrics
    args:
        X - np.array[n_obs, n_dim] Groups 0 of vec
        Y - np.array[n_obs, n_dim] Groups 1 of vec
        metric - ['euclidean','cosine'] The metric chosen to compute the distance matrix
    return:
        dict - result dictionary from R crossmatchtest function
        - a1: The number of cross-matches
        - Ea1 The expected number of cross-matches under the null
        - Va1 The variance of number of cross-matches under the null
        - dev The observed difference from expectation under null in SE units
        - pval The p-value based on exact null distribution (NA for datasets with 340 observations or more)
        - approxpval The approximate p-value based on normal approximation
    '''
    # construct binary label array and distance matrix
    # label 0: Control Group
    # label 1: Treatment Group
    N_vec = np.vstack((X,Y))
    z_py = np.array([0]*len(X) + [1]*len(Y))
    if metric=='euclidean':
        D_py = distance_matrix(N_vec,N_vec,2)
    elif metric=='cosine':
        D_py = cosine_distances(N_vec,N_vec)
    else:
        print("Invalid metric type!")
        return

    # convert py dtype to R dtype
    z_r = BoolVector(z_py)
    nr, nc = D_py.shape
    Dvec = FloatVector(D_py.transpose().reshape((D_py.size)))
    D_r = r.matrix(Dvec, nrow=nr, ncol=nc)

    # call R function: crossmatchtest. Results represent the following:
    res = r['crossmatchtest'](z_r,D_r)
    return dict(zip(['a1','Ea1','Va1','dev','pval','approxpval'],[val[0] for val in res]))

def get_brand_top_membership(df_cluster, df_brand, brandname, Tgroup, n=10):
    '''Print top 10 cluster with highest membership score for the given brand'''
    membership_col = list(df_brand.filter(regex=("dist_from_centroid_\d")).columns)
    label_name_mapper = dict(zip(df_cluster['label'],df_cluster['central_word']))
    mapper = {'dist_from_centroid_'+str(k):v for k,v in label_name_mapper.items()}

    Treat = 'yes' if Tgroup else 'no'
    mem_score = df_brand[(df_brand[brand]==brandname) & (df_brand[treatment]==Treat)][membership_col].squeeze()
    return mem_score.nlargest(n).rename(mapper)
    
def crossmatch_test_for_brand(df, df_cluster, df_brand, brandname, verbose=False):
    '''Conduct crossmatch test for a given brand
    Print membership changes for a given brand
    args:
        df - participant level df
        df_cluster - cluster level df
        df_brand - brand level df
        brandname(str)
        verbose - (default False) Print out top 10 word cluster membership for C and T group.
    return:
        dict - crossmatch results 
    '''
    T = df[(df[treatment]=='yes') & (df[brand]==brandname)][membership_col].to_numpy()
    C = df[(df[treatment]=='no') & (df[brand]==brandname)][membership_col].to_numpy()
    res = crossmatch_2_groups(T,C,metric='cosine')
    
    if verbose:
        print(f"Crossmatch test for {brandname}")
        print(res)
        print("==========================================")
        print('Treatment group top 10 membership:')
        print(get_brand_top_membership(df_cluster,df_brand,brandname,Tgroup=True))
        print()
        print('Control group top 10 membership:')
        print(get_brand_top_membership(df_cluster,df_brand,brandname,Tgroup=False))
    return res

def brand_TC_membership_crossmatch_test(df,df_cluster,df_brand):
    '''Get crossmatch test restults on membership changes from C to T for each brand in df
    args:
        df - participant-level df
    return:
        pd.DataFrame[brand,]: each row contains crossmatch result for a certain brand
    '''
    data = {}
    for brandname in set(df[brand].values):
        res = crossmatch_test_for_brand(df,df_cluster,df_brand,brandname,verbose=False)
        data[brandname] = res
    return pd.DataFrame.from_dict(data, orient='index')


############################
## Correlation & Mediation & Moderation
############################
def mediation_analysis(df,mediators):
    res = []

    for mediator in mediators:
        for iv,dv in [('avg_association','mem_var'),('idea_change','mem_cos_dist_from_C')]:
            # (c) total effect
            model = lr(df[[iv,dv]].dropna())
            t_c,p_c = model[0][iv], model[1][iv]

            # (c',b) direct effect
            model = lr(df[[iv,mediator,dv]].dropna())
            t_c_prime,p_c_prime,t_b,p_b = model[0][iv], model[1][iv],model[0][mediator], model[1][mediator]

            # (a)
            model = lr(df[[iv, mediator]].dropna())
            t_a,p_a = model[0][iv], model[1][iv]

            res.append([mediator,iv,dv,t_a,p_a,t_b,p_b,t_c_prime,p_c_prime,t_c,p_c,])
    
    res = pd.DataFrame(data=res,columns=['moderator','iv','dv','a (t stats)', 'a (pval)', 'b (t stats)', 'b (pval)', 'c\' (t stats)', 'c\' (pval)', 'c (t stats)', 'c (pval)'])
    res.set_index(['moderator', 'iv','dv'], inplace=True)
    return res

def mediation_replace_typicality(df,mediators):
    res = []
    
    # baseline
    model = lr(df[['avg_association','mem_var']].dropna())
    t_c1,p_c1 = model[0]['avg_association'], model[1]['avg_association']
    res.append(['avg_association','mem_var',t_c1,p_c1])
    
    model = lr(df[['idea_change','mem_cos_dist_from_C']].dropna())
    t_c2,p_c2 = model[0]['idea_change'], model[1]['idea_change']
    res.append(['idea_change','mem_cos_dist_from_C',t_c2,p_c2])
    
    for mediator in mediators:
        for dv in ['avg_association','mem_var','idea_change','mem_cos_dist_from_C']:
            model = lr(df[[mediator,dv]].dropna())
            t_c,p_c = model[0][mediator], model[1][mediator]
            res.append([mediator,dv,t_c,p_c])
    
    res = pd.DataFrame(data=res,columns=['iv','dv','t', 'pval'])
    res.set_index(['iv','dv'], inplace=True)
    return res

def moderation_analysis(df,moderators):
    res = []
    
    for moderator in moderators:
        # (1a) Typicality Strengthening of Associations
        _1a = lr(df[['typicality', moderator,'avg_association']].dropna())
        t_iv,p_iv,t_m,p_m = _1a[0]['typicality'], _1a[1]['typicality'], _1a[0][moderator],_1a[1][moderator]
        res.append([moderator,'1a', t_iv,p_iv,t_m,p_m])

        # (1b) Typicality  Vector Variance
        _1a = lr(df[['typicality', moderator,'mem_var']].dropna())
        t_iv,p_iv,t_m,p_m = _1a[0]['typicality'], _1a[1]['typicality'], _1a[0][moderator],_1a[1][moderator]
        res.append([moderator,'1b', t_iv,p_iv,t_m,p_m])

        # (2a) Typicality  Changing of Associations
        _1a = lr(df[['typicality', moderator,'idea_change']].dropna())
        t_iv,p_iv,t_m,p_m = _1a[0]['typicality'], _1a[1]['typicality'], _1a[0][moderator],_1a[1][moderator]
        res.append([moderator,'2a', t_iv,p_iv,t_m,p_m])

        # (2b) Typicality  Vector Change
        _1a = lr(df[['typicality', moderator,'mem_cos_dist_from_C']].dropna())
        t_iv,p_iv,t_m,p_m = _1a[0]['typicality'], _1a[1]['typicality'], _1a[0][moderator],_1a[1][moderator]
        res.append([moderator,'2b', t_iv,p_iv,t_m,p_m])

    res = pd.DataFrame(data=res,columns=['moderator','model','t_typicality', 'p_typicality', 't_moderator', 'p_moderator'])
    res.set_index(['moderator', 'model'], inplace=True)
    return res.sort_index(inplace=False)

def two_arrow_moderation_analysis(df,moderator,n_slices=[3,3]):
    '''Moderating effect on the 2 major arrows:
    1. avg_associaiton => membership variance
    2. idea_change => dist from C avg

    args:
        df - participant level df
        moderator(str): moderator variable
        n_slices(list[int]): number of slices each moderator should be divided to. Default [3,3]
    return:
        (df1[interval,], df2[interval,])
            left - left interval boundary
            right - right interval boundary
            cnt - number of observations in the interval
            corr - correlation for the observations within the interval
            p - pval
    '''
    IV = 'avg_association'
    DV = 'mem_var'
    df1 = moderation_analysis(df,IV,DV,moderator,n_slices[0])

    IV = idea_change
    DV = 'mem_cos_dist_from_C'
    df2 = moderation_analysis(df,IV,DV,moderator,n_slices[1])
    
    return(df1,df2)

def save_xls(list_dfs, sheet_names, xls_path):
    '''Save a list of dfs to sheets of the same xls file, one df per sheet.'''
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, sheet_names[n])
        writer.save()


############################
## Case Study: Visualize Changes
############################
def emb_from_membership_score(df, df_cluster):
    '''Reconstruct embedding as linear combinations of word cluster centroid embeddings using membership dist rating.
    args:
        df - df that contains membership distance. Dist towards each cluster centroid as a separate col.
        df_cluster: cluster level df.
    '''
    def calc_emb_fr_mem(membership,centroid):
        '''Reconstruct embedding as linear combinations of word cluster centroid embeddings using membership dist rating (larger means closer to).
        args:
            centroid(np.array[n_cluster, 300]) - centroid embedding matrix
            membership(np.array[n_observation, n_cluster]) - membership rating matrix
        '''
        return np.matmul(membership, centroid)
    
    M = df[membership_col].to_numpy()
    C = np.array(df_cluster['centroid'].to_list())
    return calc_emb_fr_mem(M,C)

def vis_2d_brand_membership_C(df_brand,df_cluster,dim_list):
    '''visualize brand control group only in a set of quadrant plots'''
    nrow = (len(dim_list)-1)//3+1
    ncol = 3
    fig,axes = plt.subplots(nrow,ncol, figsize=(ncol*5,nrow*4))
    axes = axes.flatten()
    xy_lims = []
    
    for i in range(len(dim_list)):
        dim1,dim2 = dim_list[i]
        ax = axes[i]
        s_brand_C = df_brand[df_brand['treatment']=='no']
        vis_2d_membership(s_brand_C,df_cluster,dim1,dim2,s_brand_C['brand'],c='r',ax=ax)
        set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        xy_lims.append((xmin,xmax,ymin,ymax))
    plt.tight_layout(pad=3.0)
    return fig,xy_lims

def vis_2d_brand_membership_TC(df_brand,df_cluster,dim_list):
    '''visualize brand treatment and control group in a set of quadrant plots'''
    nrow = (len(dim_list)-1)//3+1
    ncol = 3
    fig,axes = plt.subplots(nrow,ncol, figsize=(ncol*5,nrow*4))
    axes = axes.flatten()
    xy_lims = []
    
    for i in range(len(dim_list)):
        dim1,dim2 = dim_list[i]
        ax = axes[i]
        s_brand_C = df_brand[df_brand['treatment']=='no']
        s_brand_T = df_brand[df_brand['treatment']=='yes']

        vis_2d_membership(s_brand_C,df_cluster,dim1,dim2,s_brand_C['brand'],c='r',ax=ax)
        vis_2d_membership(s_brand_T,df_cluster,dim1,dim2,s_brand_T['brand'],c='g',ax=ax)
        
        set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        xy_lims.append((xmin,xmax,ymin,ymax))
    plt.tight_layout(pad=3.0)
    return fig,xy_lims

def vis_2d_influencer(df_inf,df_brand,df_cluster,brandname, influencer, dim_list,xy_lims):
    '''plot influencer for the chosen brand in a set of quadrant plots.
    This plot is adjust to brand scale (same xmin,xmax,ymin,ymax for axes as the brand plot)
    For the given brand:
    1. Plot each influencer's mem score
    2. Plot brand treatment avg (green)
    3. Plot brand control avg (red)
    args:
        df_inf: influencer level df
        df_brand: brand level df, needed for treat/control avg
        df_cluster: cluster level df, needed to map central words to cluster label
        brandname(str)
        dim_list: a list of dims
        influencer: str, influencer who will receive a special marker
        xy_lims(list[len(dim_list),]): list of tuples (xmin,xmax,ymin,ymax)
    '''
    nrow = (len(dim_list)-1)//3+1
    ncol = 3
    fig,axes = plt.subplots(nrow,ncol, figsize=(ncol*5,nrow*4))
    axes = axes.flatten()
    
    # series to plot
    s_inf_mem = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']!=influencer)][membership_col]
    s_inf_name = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']!=influencer)]['handle']
    s_inf_special = df_inf[(df_inf['brand']==brandname) & (df_inf['handle']==influencer)][membership_col]
    s_treat = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='yes')][membership_col]
    s_control = df_brand[(df_brand['brand']==brandname) & (df_brand['treatment']=='no')][membership_col]
    
    for i in range(len(dim_list)):
        dim1,dim2 = dim_list[i]
        ax = axes[i]
        
        # set ax lim, use the same lim as the plot for Control group brand
        xmin,xmax,ymin,ymax = xy_lims[i]
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
 
        # plot points
        vis_2d_membership(s_inf_mem,df_cluster,dim1,dim2,s_inf_name,c='b',ax=ax) # influencer
        vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='b',s=80,marker='*',ax=ax) # spceial influencer
        vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
        vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax) # control
        
        # set up spine
        set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig

# helper func
def set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None):
    '''Set up spines of quadrant plot
    args:
        dim1(str): central word for dim1
        dim2(str): central word for dim2
        ax: which ax to adjust spine
        x_loc: x location of yaxis
        y_loc: y location of xaxis
    '''
    # set spline location
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    if not x_loc:
        x_loc = (xmax+xmin)/2
    if not y_loc:
        y_loc = (ymax+ymin)/2
        
    # set spines
    ax.spines['left'].set_position(('data', x_loc))
    ax.spines['bottom'].set_position(('data', y_loc))
    ax.xaxis.set_ticks([]); ax.yaxis.set_ticks([])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # set text
    ax.text(xmax, y_loc, "MORE "+ dim1.upper(),horizontalalignment='left',verticalalignment='center',weight = 'bold',bbox=dict(facecolor='white', alpha=1))
    ax.text(x_loc, ymax, "MORE "+dim2.upper(),horizontalalignment='center',verticalalignment='bottom',weight = 'bold',bbox=dict(facecolor='white', alpha=1))
    ax.text(xmin, y_loc,"LESS "+dim1.upper(),horizontalalignment='right',verticalalignment='center',weight = 'bold',bbox=dict(facecolor='white', alpha=1))
    ax.text(x_loc, ymin,"LESS "+dim2.upper(),horizontalalignment='center',verticalalignment='top',weight = 'bold',bbox=dict(facecolor='white', alpha=1))

def vis_2d_membership(df,df_cluster,dim1,dim2,text,c='b',s=30,marker='o',ax=None):
    '''Plot membership score as coordinates in 2d space
    args:
        df - df with membership_cols
        df_cluster - cluster level df to obtain mapping between cluster label and central words
        dim1(str) - central words for x axis
        dim2(str) - central words for y axis
        c(str) - color for points
        text(list[str]) - text label for each data point
    '''
    label1 = df_cluster.loc[df_cluster['central_word'] == dim1, 'label'].values[0]
    label2 = df_cluster.loc[df_cluster['central_word'] == dim2, 'label'].values[0]
    col1,col2 = membership_col[label1], membership_col[label2]

    x,y = df[col1], df[col2]
    
    # plot
    if not ax:
        fig, ax = plt.subplots(1,1)
    
    # influencer
    ax.scatter(x,y,alpha=1,c=c,s=s,marker=marker)
    for xi,yi,texti in zip(x,y,text):
        ax.text(xi,yi,texti)
    
    return ax

def get_word_count(df,brand,influencer=None):
    '''get word count dictionary for the chosen brand and influencer'''
    if influencer:
        df_filtered = df[(df['brand']==brand) & (df['handle']==influencer)]
    else:
        df_filtered = df[(df['brand']==brand)]
    df_filtered = df_filtered[P1_lemma+P2_lemma].applymap(Counter)
    return dict(df_filtered.values.sum())
    
def vis_wordcloud(word_freq):
    wordcloud = WordCloud(width = 400, height = 200, random_state=1, background_color='white', collocations=False)
    wordcloud.fit_words(word_freq)
    plt.imshow(wordcloud) 
    plt.axis("off")
    return plt.gcf()

def get_major_dim(df_brand,brandname,df_cluster,n):
    '''Return the central words of the top n dimensions that have received the largest word count'''
    cluster_cnt = df_brand[df_brand['brand']==brandname].iloc[0,]['cluster_counter']
    indices = np.argsort(cluster_cnt)[-1:-n-1:-1]
    
    res = []
    for idx in indices:
        cword = df_cluster[df_cluster['label']==idx]['central_word'].values[0]
        res.append(cword)
    return res



