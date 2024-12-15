import numpy as np
import ERP
import scipy.stats as sta

# STSs should be in the same length, and g is the gap in ERP.
def cal_simi_matrix(sts, g):
    sts_num = len(sts)
    gap = np.array(g, dtype=np.float64)
    dist = np.zeros([sts_num, sts_num])
    for i, data_i in enumerate(sts):
        tsi = np.array(data_i, dtype=np.float64)
        for j, data_j in enumerate(sts):
            tsj = np.array(data_j, dtype=np.float64)
            if i < j:
                dist[i, j] = ERP.erp_dist(tsi, tsj, gap)
                print('ERP between STS-%s and STS-%s has been calculated!' % (i, j))
                dist[j, i] = dist[i, j]
    max_dist = np.max(dist)
    min_dist = np.min(dist)
    simi = 1 - (dist - min_dist)/(max_dist - min_dist) 
    return simi

def global_measure(simi, weight):
    N = len(simi)*(len(simi)-1)
    numerator = np.sum(np.multiply(weight, simi))
    denominator = np.sum(simi)
    s = numerator / denominator
    global_s = s
    w = np.sum(weight)
    es = w/N
    b = N*np.sum(np.multiply(simi, simi))/(np.sum(simi)**2)
    var = w/N*(N-w)/N*(b-1)/(N-1)
    z = (s - es) / np.sqrt(var)
    global_z_score = z
    global_p_value = sta.norm.sf(np.abs(z))
    return global_s, global_z_score, global_p_value

def local_measure(simi, weight):
    num = len(simi)
    local_s = np.zeros([num, 1], dtype=np.float64)
    local_p = np.zeros([num, 1], dtype=np.float64)
    for index in range(num):
        wi = np.sum(weight[index, :])
        s1i = np.dot(weight[index, :], weight[:, index])
        xi = (np.sum(simi[index, :]) - 1 ) / (num - 1)
        s2i = (np.dot(simi[index, :], simi[:, index]) - 1)/(num - 1) - xi**2
        local_s[index, 0] = (np.dot(weight[index, :], simi[:, index]) - wi*xi)/(np.sqrt(s2i) * np.sqrt(((num - 1)*s1i - wi**2)/(num - 2)))
        local_p[index, 0] = sta.norm.sf(np.abs(local_s[index, 0]))
    return local_s, local_p
    