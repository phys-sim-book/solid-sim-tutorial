# ANCHOR: broad_phase
from copy import deepcopy
import numpy as np
import math

import distance.PointEdgeDistance as PE

# check whether the bounding box of the trajectory of the point and the edge overlap
def bbox_overlap(p, e0, e1, dp, de0, de1, toc_upperbound):
    max_p = np.maximum(p, p + toc_upperbound * dp) # point trajectory bbox top-right
    min_p = np.minimum(p, p + toc_upperbound * dp) # point trajectory bbox bottom-left
    max_e = np.maximum(np.maximum(e0, e0 + toc_upperbound * de0), np.maximum(e1, e1 + toc_upperbound * de1)) # edge trajectory bbox top-right
    min_e = np.minimum(np.minimum(e0, e0 + toc_upperbound * de0), np.minimum(e1, e1 + toc_upperbound * de1)) # edge trajectory bbox bottom-left
    if np.any(np.greater(min_p, max_e)) or np.any(np.greater(min_e, max_p)):
        return False
    else:
        return True
# ANCHOR_END: broad_phase

# ANCHOR: accd
# compute the first "time" of contact, or toc,
# return the computed toc only if it is smaller than the previously computed toc_upperbound
def narrow_phase_CCD(_p, _e0, _e1, _dp, _de0, _de1, toc_upperbound):
    p = deepcopy(_p)
    e0 = deepcopy(_e0)
    e1 = deepcopy(_e1)
    dp = deepcopy(_dp)
    de0 = deepcopy(_de0)
    de1 = deepcopy(_de1)

    # use relative displacement for faster convergence
    mov = (dp + de0 + de1) / 3 
    de0 -= mov
    de1 -= mov
    dp -= mov
    maxDispMag = np.linalg.norm(dp) + math.sqrt(max(np.dot(de0, de0), np.dot(de1, de1)))
    if maxDispMag == 0:
        return toc_upperbound

    eta = 0.1 # calculate the toc that first brings the distance to 0.1x the current distance
    dist2_cur = PE.val(p, e0, e1)
    dist_cur = math.sqrt(dist2_cur)
    gap = eta * dist_cur
    # iteratively move the point and edge towards each other and
    # grow the toc estimate without numerical errors
    toc = 0
    while True:
        tocLowerBound = (1 - eta) * dist_cur / maxDispMag

        p += tocLowerBound * dp
        e0 += tocLowerBound * de0
        e1 += tocLowerBound * de1
        dist2_cur = PE.val(p, e0, e1)
        dist_cur = math.sqrt(dist2_cur)
        if toc != 0 and dist_cur < gap:
            break

        toc += tocLowerBound
        if toc > toc_upperbound:
            return toc_upperbound

    return toc
# ANCHOR_END: accd