
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
from scipy.spatial import distance
def JSD(P, Q):

    #print(np.min(P.cpu().numpy()), np.max(P.cpu().numpy()), np.min(Q.cpu().numpy()),np.max(Q.cpu().numpy()))
    pa=P.cpu().numpy().reshape(-1)
    qa=Q.cpu().numpy().reshape(-1)
    qa+=(0.-np.min(qa))
    pa+=(0.-np.min(pa))
    jsd= distance.jensenshannon(pa/np.sum(pa), qa/np.sum(qa) )**2

    return jsd