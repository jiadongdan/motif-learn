from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def get_ami(lbs_true, lbs_pre):
    return adjusted_mutual_info_score(lbs_true, lbs_pre)

def get_ari(lbs_true, lbs_pre):
    return adjusted_rand_score(lbs_true, lbs_pre)

def get_fmi(lbs_true, lbs_pre):
    return fowlkes_mallows_score(lbs_true, lbs_pre)

def get_sc(X, lbs, metric='cosine'):
    return silhouette_score(X, lbs, metric=metric)

def get_db(X, lbs):
    return davies_bouldin_score(X, lbs)

def get_ch(X, lbs):
    return calinski_harabasz_score(X, lbs)

def get_auc():
    pass