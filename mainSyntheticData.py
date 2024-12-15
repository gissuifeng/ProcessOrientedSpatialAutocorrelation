import STS_SA
import numpy as np


def read_sts(file_path):
    sts = np.loadtxt(file_path, delimiter=",")
    return sts

def cal_spatial_weight_matrix():
    weight = np.zeros([21*28, 21*28], dtype=int)
    for i in range(21):
        for j in range(28):
            if 0 <= i - 1 < 21 and 0 <= j < 28:
                weight[28 * i + j, 28 * (i - 1) + j] = 1
            if 0 <= i + 1 < 21 and 0 <= j < 28:
                weight[28 * i + j, 28 * (i + 1) + j] = 1
            if 0 <= i < 21 and 0 <= j - 1 < 28:
                weight[28 * i + j, 28 * i + j - 1] = 1
            if 0 <= i < 21 and 0 <= j + 1 < 28:
                weight[28 * i + j, 28 * i + j + 1] = 1
    return weight

def print_global_result(s, z, p):
    print('Global S is %f' % s)
    print('Z-score is %f' % z)
    print('P-value is %f' % p)
    

def output_local_result(local_s, local_p, output_path):
    num = len(local_s)
    order = np.arange(1, num + 1).reshape(num, 1)
    local_table = np.concatenate((order, local_s, local_p), axis=1)
    np.savetxt(output_path+'\Local Spatial Autocorrelation.csv', local_table, delimiter=',', fmt='%f', header='id, s, p')

if __name__ == '__main__':
    sts = read_sts(r".\dataset\syntheticData\sts_data.csv")
    weight = cal_spatial_weight_matrix()
    simi = STS_SA.cal_simi_matrix(sts, 0) # here g=0 in ERP
    global_s, global_z_score, global_p_value = STS_SA.global_measure(simi, weight)
    print_global_result(global_s, global_z_score, global_p_value)
    local_s, local_p = STS_SA.local_measure(simi, weight)
    output_local_result(local_s, local_p, r".\result\syntheticResult")
    
