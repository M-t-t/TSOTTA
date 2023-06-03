
from scipy import stats
import numpy as np
if __name__ == '__main__':

    ''' CHBMIT AUC values for each method'''
    method1_auc = [0.565, 0.539, 0.524, 0.556, 0.456, 0.685, 0.959, 0.573, 0.678, 0.499, 0.629, 0.652,
                   0.398, 0.952, 0.515, 0.659, 0.453, 0.56, 0.732, 0.634]   #tsotta
    method2_auc = [0.583, 0.243, 0.591, 0.470, 0.497, 0.529, 0.650, 0.550, 0.627, 0.593, 0.625, 0.550,
                   0.430, 0.720, 0.630, 0.651, 0.205, 0.621, 0.654, 0.549]  # 包含20个受试者的AUC值  src
    method3_auc = [0.699, 0.358, 0.544,	0.611,	0.792, 0.248, 0.633, 0.651,	0.411,	0.427,	0.685, 0.442,
                   0.323, 0.602, 0.435,	0.593,	0.160,	0.743,	0.588,	0.354] #tent

    method4_auc = [0.666,	0.45,	0.5,	0.548,	0.445,	0.556,	0.646,	0.297,	0.753,	0.37,	0.586,	0.475,	0.352,
                   0.577,	0.51,	0.584,	0.163,	0.698,	0.529,	0.51]    #t3a
    method5_auc = [0.72, 0.371, 0.552, 0.57, 0.597, 0.496, 0.624, 0.452, 0.546, 0.383, 0.637, 0.221,
                   0.391, 0.581, 0.431, 0.702, 0.195, 0.724, 0.669, 0.604]  # sar
    #-------------------------------------
    # method7_auc = [0.649, 0.554, 0.527, 0.656, 0.771, 0.272, 0.596, 0.697, 0.313, 0.5, 0.598, 0.694,
    #                0.413, 0.628, 0.368, 0.654, 0.188, 0.705, 0.484, 0.181]  # shot
    # method4_auc = [0.665,	0.44,	0.505,	0.538,	0.503,	0.431,	0.58,	0.447,	0.611,	0.438,	0.579,
    #                0.452,	0.404,	0.556,	0.537,	0.565,	0.197,	0.665,	0.534,	0.48] # eata

    ''' Kaggle AUC values for each method'''
    # method1_auc = [0.755, 0.609, 0.644, 0.596]  # tsotta
    # method2_auc = [0.596,	0.575,	0.548,	0.49]  # 包含20个受试者的AUC值  src
    # method3_auc = [0.662,	0.574,	0.601,	0.511]  # tent
    # method4_auc = [0.574, 0.534, 0.496, 0.467]  # sar
    # method5_auc = [0.063,	0.078,	0.017,	0.029]  # t3a

    # -------------------------------------
    # method6_auc = [0.821,	0.706,	0.749,	0.615]  # shot
    # method7_auc = [0.558, 0.526, 0.478, 0.459]  # eata

    # Combine all AUC values into a numpy array
    auc_values = np.array([method1_auc, method2_auc, method3_auc, method4_auc,
                           method5_auc])

    # Perform paired t-test for each pair of methods
    num_methods = auc_values.shape[0]
    p_values = np.zeros((num_methods, num_methods))

    _,p=stats.ttest_rel(auc_values[0], auc_values[4])
    print(p)

    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            t_statistic, p_value = stats.ttest_rel(auc_values[i], auc_values[j])
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    # Print the p-values
    print("P-values:")
    for i in range(num_methods):
        for j in range(i + 1, num_methods):
            print(f"Method {i + 1} vs. Method {j + 1}: p-value = {p_values[i, j]}")
            # print(p_values[i, j] == p_values[j, i])

