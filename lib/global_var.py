###########################
##### Variable columns correspondence ##########
###########################

# Text response related cols
P1 = ['Q133_10','Q133_11','Q133_12','Q133_13','Q133_14'] #nouns
P2 = ['Q121_10','Q121_11','Q121_12','Q121_13','Q121_14'] #adjectives
P3 = ['Q171_1','Q171_2','Q171_3','Q171_4','Q171_5','Q171_6','Q171_7','Q171_8','Q171_9','Q171_10'] #
P1_lemma = [c+'_lemma' for c in P1]
P2_lemma = [c+'_lemma' for c in P2]
P1_emb = [c+'_emb' for c in P1]
P2_emb = [c+'_emb' for c in P2]
P3_adjust = [c+'_adjust' for c in P3]


# Other cols
treatment = 'treatment'
brand = 'brand'
lemmas = 'lemmas'
idea_change = 'Q168'
typicality = 'Q173'
featured_product_typicality = 'Q198' # typicality rating of the product featured in the post

user_similarity = 'Q177'
resp_conf = 'Q179' #confidence of responses
intention_to_sell = ['Q195_1','Q195_2','Q195_3']  # intentionality
sponsor_by = 'Q137'
follow_likelihood = 'Q83'

brand_conn = ['Q172_1','Q172_2','Q172_3'] #personal connection with the brand: reflect/identify/connect
post_conn = ['Q132_1','Q132_2','Q132_3'] #personal connection with the influencer: like/similar/share commonality/1
user_perc = 'Q120_1' #percentage of the population who are regular users of the brand
hour_pw = 'Q112'
age = 'Q18'
gender = 'Q19'
race = 'Q172'


# generated col name
#k = 25
#membership_col = ['dist_from_centroid_'+str(i) for i in range(k)]


# vec_var         = 'vec_var'   #vector variance for each participant
# avg_association = 'avg_association'
# mean_emb        = 'mean_emb' # mean embedding of all response of each participant
# dist_from_C     = 'dist_from_C'
# conn_with_brand = 'conn_with_brand' # how much participants can relate to the brand
# conn_with_inf   = 'conn_with_inf'   # how much participants can relate to the influencer shown
# intentionality  = 'intentionality'
# mem_cos_dist_from_C

# # global var
# word_label_dict = None  # word-label mapping as a result of kmeans clustering
############################
CTA_keywords = ["http","https"," ad ","sponsor", "endorse","collaboration","{AD}","#ad ",
                "sponsor","buy","click","learn more","act now", "apply today","promo ",
                "teaming up","liketoknowit","ltk","free gift","link in bio","purchase",
                "discount","partnering"] 

rename_col = {idea_change:'idea_change',typicality:'typicality',
              featured_product_typicality:'featured_product_typicality',
              user_similarity:'user_similarity',
              resp_conf:'resp_conf',
              sponsor_by:'sponsor_by',
              follow_likelihood:'follow_likelihood',
              user_perc:'user_perc',
              hour_pw:'hour_pw',
              age:'age',gender:'gender',race:'race'
             }

