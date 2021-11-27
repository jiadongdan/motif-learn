from sklearn.metrics import jaccard_score
from sklearn.metrics.cluster import pair_confusion_matrix



def cluster_confusion_matrix(lbs, lbs_, flat=False):
    C = pair_confusion_matrix(lbs, lbs_)
    if flat:
        TN = C[0, 0]
        FN = C[1, 0]
        TP = C[1, 1]
        FP = C[0, 1]
        return (TN, FN, TP, FP)
    else:
        return C


def get_jaccard_score(lbs, lbs_):
    TN, FN, TP, FP = cluster_confusion_matrix(lbs, lbs_, flat=True)
    return TP/(TP+FP+FN)