import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.global_var import *

def vis_2d_influencer_with_scaled_dots(df_inf,df_brand,df_cluster,brandname, influencer, dim_list,xy_lims, scaledim="typicality"):
    from lib.hard_launch import set_up_quadrant_spine
    
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
    blacklist = {
        'chic': ['unstylish', 'unfashionable', 'frumpy', 'untrendy','frump'],
        'hipster' : ['unhip'],
        'opulent' : ['poorer']
    }
    for k,v in blacklist.items():
        idx        = np.where(df_cluster.central_word == k)[0][0]
        words      = df_cluster.at[idx,"word"]
        words      = set(words) - set(v)    
        df_cluster.at[idx,"word"] = words

    return df_cluster



def plot_concept_change(brandname, df_brand, df_inf, df_cluster, topk=10):
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
    #plot_bar_membership(axs[0,0], s_control, "Control")
    #plot_bar_membership(axs[0,1], s_treat, "Treatment")
    #plot_bar_membership(axs[0,2], s_influencers, "Mean Influencer") # ~ this should be the same but there's a slight discrepancy ...

    # for non-typical influencers
    influencer_typicality = df_inf[(df_inf['brand']==brandname)].typicality
    s_typical  = df_inf[(df_inf['brand']==brandname)][influencer_typicality >= influencer_typicality.median()]
    s_atypical = df_inf[(df_inf['brand']==brandname)][influencer_typicality <= influencer_typicality.median()]

    plot_bar_membership(axs[0], s_control, "Control")
    plot_bar_membership(axs[1], pd.DataFrame(s_atypical.mean()).T, "Less Typical",color="red")
    plot_bar_membership(axs[2], pd.DataFrame(s_typical.mean()).T, "More Typical", color="green")

def plot_concept_change_manybrands(brandname, df_brand, df_inf, df_cluster, topk=10):
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
    
    
    
def vis_2d_justone(dim1, dim2, df_inf,df_brand,df_cluster,brandname, influencer, dim_list,xy_lims, scaledim="typicality"):
    from lib.hard_launch import set_up_quadrant_spine
    
    fig,ax = plt.subplots(1,1, figsize=(5,4), facecolor="white",dpi=100)
    
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
    vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='orange',s=focus_inf_typicality,ax=ax) # special influencer
    vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
    vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax,marker="o") # control

    # set up spine
    set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig



def vis_2d_manybrands(dim1, dim2, df_inf,df_brand,df_cluster,brandname, influencer, dim_list,xy_lims, scaledim="typicality"):
    
    from lib.hard_launch import set_up_quadrant_spine
    
    fig,ax = plt.subplots(1,1, figsize=(5,4), facecolor="white",dpi=100)
    
    # series to plot
    infinbrand    = [brand in brandname for brand in df_inf['brand'].values]
    brinbrand     = [brand in brandname for brand in df_brand['brand'].values]
    
    print("fixed?")

    if(influencer is None or influencer == ''):
        influencer = df_inf[infinbrand].handle[0]
        print("fixed")
    
    s_inf_mem     = df_inf[infinbrand & (df_inf['handle']!=influencer)][membership_col]
    s_inf_name    = df_inf[infinbrand & (df_inf['handle']!=influencer)]['handle']
    s_inf_special = df_inf[infinbrand & (df_inf['handle']==influencer)][membership_col]
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
    vis_2d_membership(s_inf_special,df_cluster,dim1,dim2,[influencer],c='orange',s=focus_inf_typicality,ax=ax) # special influencer
    #vis_2d_membership(s_treat,df_cluster,dim1,dim2,['TREAT'],c='g',s=100,ax=ax)   # treat
    #vis_2d_membership(s_control,df_cluster,dim1,dim2,['CTRL'],c='r',s=100,ax=ax,marker="o") # control
    
    # set up spine
    set_up_quadrant_spine(dim1,dim2,ax,x_loc=None,y_loc=None)
    plt.tight_layout(pad=3.0)
    return fig

