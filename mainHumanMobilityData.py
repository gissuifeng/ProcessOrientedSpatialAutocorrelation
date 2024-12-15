import STS_SA
import numpy as np


def read_sts(file_path):
    sts = np.loadtxt(file_path, delimiter=",")
    num = len(sts)
    return sts, num

def read_spatial_weight_matrix(filepath, num):
    table = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.int64)
    length = len(table)
    weight = np.zeros([num, num], dtype=int)
    index = 0
    while index < length:
        i = table[index, 1]
        j = table[index, 2]
        weight[i - 1][j - 1] = 1
        index += 1
    return weight

def print_global_result(s, z, p):
    print('Global S is %f' % s)
    print('Z-score is %f' % z)
    print('P-value is %f' % p)
    

def output_local_result(local_s, local_p, output_path):
    num = len(local_s)
    order = np.arange(1, num + 1).reshape(num, 1)
    local_table = np.concatenate((order, local_s, local_p), axis=1)
    np.savetxt(output_path+'\Local Spatial Autocorrelation pickup.csv', local_table, delimiter=',', fmt='%f', header='id, s, p')
    #np.savetxt(output_path+'\Local Spatial Autocorrelation dropoff.csv', local_table, delimiter=',', fmt='%f', header='id, s, p')

if __name__ == '__main__':
    sts, num = read_sts(r".\dataset\humanMobilityData\sts_pickup.csv")
    #sts, num = read_sts(r".\dataset\humanMobilityData\sts_dropoff.csv")
    weight = read_spatial_weight_matrix(r".\dataset\humanMobilityData\spatial_weight_matrix.csv", num)
    simi = STS_SA.cal_simi_matrix(sts, 0) # here g=0 in ERP
    global_s, global_z_score, global_p_value = STS_SA.global_measure(simi, weight)
    print_global_result(global_s, global_z_score, global_p_value)
    local_s, local_p = STS_SA.local_measure(simi, weight)
    output_local_result(local_s, local_p, r".\result\humanMobilityResult")
    
