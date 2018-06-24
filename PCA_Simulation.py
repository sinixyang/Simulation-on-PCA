# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 11:45:58 2016

@author: sinix
"""

import os
import numpy

def get_cov_matrix(variance1,variance2,correlation):
#define a 2D covariance matrix using three parameters
    covariance = (variance1*variance2)**0.5 * correlation
    return numpy.array([[variance1,covariance],[covariance,variance2]])

def get_ext_cov_matrix(cov_matrix,variance,corelation):
#add a dimention to the prexisted covariance matrix.
#For example the orginal covariance matrix (parameter "cov_matrix" in this function) is
# |a,b|
# |b,c|
#The variance of the additional variable is d (parameter "variance" in this function)
#The corelation coefficient between the additional variable and all the former variables is a same constant f (parameter "corelation" in this function)
#The extend cov_matrix is 
# |a           ,           b, sqrt(a*d)*f|
# |b           ,           c, sqrt(b*d)*f|
# |sqrt(a*d)*f , sqrt(b*d)*f,           d|
#where sqrt() is square root calculation
    cov_matrix_len = cov_matrix.shape[0]
    ext_shape = (cov_matrix_len + 1,cov_matrix_len + 1)
    ext_cov_matrix = numpy.ones(ext_shape,dtype = "float")
    ext_cov_matrix[0:cov_matrix_len,0:cov_matrix_len] = cov_matrix
    ext_cov_matrix[cov_matrix_len,cov_matrix_len] = variance
    for i in range(cov_matrix_len):
        ext_cov_matrix[cov_matrix_len,i] = ext_cov_matrix[i,cov_matrix_len] = (ext_cov_matrix[i,i]*variance)**0.5 * corelation
    return ext_cov_matrix
    
def get_simu_data(cov_matrix,lambda_,nof_group,group_size = 100):
#get a random dataset 
    dimension = len(cov_matrix)
    zero_array = numpy.zeros((dimension),dtype = float)
    group_mean_arrays = numpy.random.multivariate_normal(zero_array,cov_matrix,nof_group)
    group_cov_matrix = cov_matrix * lambda_
    data_array = zero_array.copy()
    for group_mean_array in group_mean_arrays:
        group_data_array = numpy.random.multivariate_normal(group_mean_array,group_cov_matrix,group_size)
        data_array = numpy.vstack((data_array,group_data_array))
    data_array = data_array[1:]
    return data_array

def get_pc1(cov_matrix):
# PCA algorithm based on covariance matrix
    eigen_values,eigen_vectors = numpy.linalg.eig(cov_matrix)
    eigen_rank = numpy.argsort(eigen_values)[::-1]
    eigen_value_sorted = eigen_values[eigen_rank]
    eigen_vector_sorted = eigen_vectors[:,eigen_rank]
    pc1 = eigen_vector_sorted[:,0]
    pc1_weight = eigen_value_sorted[0]/(eigen_values.sum())
    return pc1,pc1_weight

def get_data_pc1(rdarray):
# apply PCA on the data, returns the eigen vector and eigen value of PC1
    rdarray_centered = rdarray - rdarray.mean(axis = 0)
    data_size = len(rdarray)
    cov_matrix = numpy.dot(rdarray_centered.T,rdarray_centered)/data_size
    pc1,pc1_weight = get_pc1(cov_matrix)
    return pc1,pc1_weight 
  
def get_angle(vector1,vector2):
#calculate the angle between PC1S and PC1T
    angle_cos = numpy.dot(vector1,vector2)
    angle = numpy.arccos(angle_cos)*360/numpy.pi
    angle = angle%180
    if angle >= 90:
        angle = 180 - angle
    return angle

class Recorder():
# recorder the result to a file    
    def __init__(self,record_file):
        self.record_file = record_file
    def __call__(self,record_thing):
        self.file = open(self.record_file, 'a')
        self.file.write(record_thing)
        self.file.close()

def run_simulation(dimension_args = 2, #base dimension
                   correlation_args = (0.1,0.9,9), # parameters for alpha, values are a, a + (b-a)/c, a + 2(b-a)/c,...,a + (c -1)(b-a)/c, b
                   nof_group_args = (1,8),#parameters for g: g is 2^(1+a), 2^(1+a +1),.., 2^(1+a + b)
                   dispersion_args = (-7,7,15), #parameters for Lambda: Lambda values are 2^a, 2^(a + (b-a)/c), 2^(a + 2(b-a)/c),...,2^(a + (c -1)(b-a)/c), 2^b 
                   add_dimension_cor_incre_args = (-0.16,0.16,9), #paratmenter for delta, delta values are a, a + (b-a)/c, a + 2(b-a)/c,...,a + (c -1)(b-a)/c, b
                   group_size = 500, #number of data point in each group
                   nof_case = 1000,# times of simulation for every parameter combanation
                   recorder_folder = r"C:\Users\Sinix Studio\Desktop\PCA_simu", #folder to save results of simulation
                   ):
    record_sub_folder = recorder_folder + "\\pca_simu_sub" #folder to save simulation result of a combination of parameters in 2D case
    record_ex_folder = recorder_folder + "\\pca_simu_ex" #folder to save simulation result of a combination of parameters in 3D case
    record_sub_sta_file = recorder_folder + "\\pca_simu_sub.csv" #record imformation of statistic for 2D case
    record_ex_sta_file = recorder_folder + "\\pca_simu_ex.csv" #record imformation of statistic for 3D case, this this the file
    if not os.path.exists(record_sub_folder):
        os.makedirs(record_sub_folder)
    if not os.path.exists(record_ex_folder):
        os.makedirs(record_ex_folder)
    correlation_list = numpy.linspace(*correlation_args)
    nof_group_list = map(lambda x: 2**(x + nof_group_args[0]),range( nof_group_args[1]))
    dispersion_list = numpy.logspace(*dispersion_args,base = 2) #this lambda
    ext_dimension_cor_incre_list = numpy.linspace(*add_dimension_cor_incre_args)
    dimension =  dimension_args
    recorder_ex_sta = Recorder(record_ex_sta_file)
    recorder_sub_sta = Recorder(record_sub_sta_file)
# Fields in the record
#     Dim: base dimension
#     PC1WT3: PC1's weight in theory (3 dimensional case), redundancy result
#     PC1WT2: PC1's weight in theory (2 dimensional case), redundancy result
#     PC1WS3: PC1's weight in simulation (3 dimensional case), redundancy result
#     PC1WS2: PC1's weight in simulation (2 dimensional case), redundancy result
#     PC1WSTD3: standard deviation of PC1's weight in simulation (3 dimensional case), redundancy result
#     PC1WSTD2: standard deviation of PC1's weight in simulation (2 dimensional case), redundancy result
#     Theta3: the average of angle between theoretical PC1 and simulated PC1 (3 dimensional case)
#     Theta2: the average angle between theoretical PC1 and simulated PC1 (2 dimensional case)
#     ThetaSTD3: the standard variance of angle between theoretical PC1 and simulated PC1 (3 dimensional case), redundancy result
#     ThetaSTD2: the standard variance of angle between theoretical PC1 and simulated PC1 (2 dimensional case), redundancy result
    
    recorder_sub_sta_fields = "Dim,Alpha,Lambda,G,PC1WT3,Theta2,ThetaSTD2,PC1WSMD,PC1WSTDMD\n"
    recorder_sub_sta(recorder_sub_sta_fields)
    recorder_ex_sta_fields = "Dim,Alpha,Lambda,G,Delta,PC1WT3,PC1WT2,Theta2,ThetaSTD2,PC1WS2,PC1WSTD2,Theta3,ThetaSTD3,PC1WS3,PC1WSTD3\n"
    recorder_ex_sta(recorder_ex_sta_fields)
    for correlation in correlation_list:
        cov_matrix = get_cov_matrix(1,1,correlation)
        while (dimension - 2) > 0:
            dimension = dimension -1
            cov_matrix = get_ext_cov_matrix(cov_matrix,1,correlation)
        for nof_group in nof_group_list:
            for dispersion in dispersion_list:
                theoretically_sub_pc1,theoretically_sub_pc1_weight = get_pc1(cov_matrix)
                sub_angle_list,simu_data_sub_pc1_weight_list = [],[]
                ext_dimension_cor_list = map(lambda x:correlation + x,ext_dimension_cor_incre_list)
                ext_dimension_cor_bool_list = map(lambda x:(0 < x)*( x < 1),ext_dimension_cor_list)
                if sum(ext_dimension_cor_bool_list) == len(ext_dimension_cor_bool_list):
                    for ext_dimension_cor in ext_dimension_cor_list:
                        ext_dimension_cor_incre = ext_dimension_cor - correlation
                        record_sub_name = "%d-%.2f-%.2f-%d" %(dimension,correlation,dispersion,nof_group) 
                        record_sub_file = record_sub_folder + "\\" + record_sub_name + ".csv"
                        recorder_sub = Recorder(record_sub_file)
                        record_ex_name = "%d-%.2f-%.2f-%d-%.2f" %(dimension,correlation,dispersion,nof_group,ext_dimension_cor_incre)
                        record_ex_file = record_ex_folder + "\\" + record_ex_name + ".csv"
                        recorder_ex = Recorder(record_ex_file)                        
                        ex_sub_angle_list,ex_simu_data_sub_pc1_weight_list,ex_angle_list,ex_simu_data_pc1_weight_list = [],[],[],[]
                        theoretically_sub_pc1,theoretically_sub_pc1_weight = get_pc1(cov_matrix)
                        ext_cov_matrix = get_ext_cov_matrix(cov_matrix,1,ext_dimension_cor)
                        theoretically_pc1,theoretically_pc1_weight = get_pc1(ext_cov_matrix)
                        recorder_ex_head_args = (dimension,correlation,dispersion,nof_group,ext_dimension_cor_incre,theoretically_pc1_weight,theoretically_sub_pc1_weight)
                        recorder_ex_head = "dim:%d,cor:%.4f,disp:%.4f,nofg:%d,edci:%.4f,tpw:%.4f,tspw:%.4f, , , , \n" %recorder_ex_head_args
                        recorder_ex_fields = "id,sub_angle,simu_data_sub_pc1_weight,angle,simu_data_pc1_weight\n"
                        recorder_ex(recorder_ex_head)
                        recorder_ex(recorder_ex_fields)
                        recorder_sub_head_args = (dimension,correlation,dispersion,nof_group,theoretically_sub_pc1_weight)
                        recorder_sub_head = "dim:%d,cor:%.4f,disp:%.4f,nofg:%d,tpw:%.4f, , \n"  %recorder_sub_head_args
                        recorder_sub_fields = "id,sub_angle,simu_data_sub_pc1_weight\n"
                        recorder_sub(recorder_sub_head)
                        recorder_sub(recorder_sub_fields)
                        
                        for i in range(nof_case):
                            simu_data = get_simu_data(ext_cov_matrix,dispersion,nof_group,group_size)
                            simu_data_sub = simu_data[:,:-1]
                            simu_data_sub_pc1,simu_data_sub_pc1_weight = get_data_pc1(simu_data_sub)
                            sub_angle = get_angle(theoretically_sub_pc1,simu_data_sub_pc1)
                            simu_data_pc1,simu_data_pc1_weight = get_data_pc1(simu_data)
                            angle = get_angle(theoretically_pc1,simu_data_pc1)
                            ex_sub_angle_list.append(sub_angle)
                            sub_angle_list.append(sub_angle)
                            ex_angle_list.append(angle)
                            ex_simu_data_sub_pc1_weight_list.append(simu_data_sub_pc1_weight)
                            simu_data_sub_pc1_weight_list.append(simu_data_sub_pc1_weight)
                            ex_simu_data_pc1_weight_list.append(simu_data_pc1_weight)
                            record_content_all = "%d,%.4f,%.4f,%.4f,%.4f\n" %(i,sub_angle,simu_data_sub_pc1_weight,angle,simu_data_pc1_weight)
                            record_content_sub = "%d,%.4f,%.4f\n" %(i,sub_angle,simu_data_sub_pc1_weight)
                            recorder_ex(record_content_all)
                            recorder_sub(record_content_sub)
                        ex_sub_angles = numpy.array(ex_sub_angle_list,dtype = "float")
                        ex_sub_angle_std = ex_sub_angles.std()
                        ex_sub_angle_mean = ex_sub_angles.mean()
                        ex_simu_data_sub_pc1_weights = numpy.array(ex_simu_data_sub_pc1_weight_list,dtype = "float")
                        ex_simu_data_sub_pc1_weight_mean = ex_simu_data_sub_pc1_weights.mean()
                        ex_simu_data_sub_pc1_weight_std = ex_simu_data_sub_pc1_weights.std()
                        ex_angles = numpy.array(ex_angle_list,dtype = "float")
                        ex_angle_mean = ex_angles.mean()
                        ex_angle_std = ex_angles.std()
                        ex_simu_data_pc1_weights = numpy.array(ex_simu_data_pc1_weight_list, dtype = "float")
                        ex_simu_data_pc1_weight_mean = ex_simu_data_pc1_weights.mean()
                        ex_simu_data_pc1_weight_std = ex_simu_data_pc1_weights.std()

                        recorder_ex_sta_line_args = (dimension,correlation,dispersion,nof_group,ext_dimension_cor_incre,theoretically_pc1_weight,theoretically_sub_pc1_weight,ex_sub_angle_mean,
                                                      ex_sub_angle_std,ex_simu_data_sub_pc1_weight_mean,ex_simu_data_sub_pc1_weight_std,ex_angle_mean,ex_angle_std,ex_simu_data_pc1_weight_mean,ex_simu_data_pc1_weight_std)
                        recorder_ex_sta_line = "%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % recorder_ex_sta_line_args

                        recorder_ex_sta(recorder_ex_sta_line)
                        
                else:
                    record_sub_name = "%d-%.2f-%.2f-%d" %(dimension,correlation,dispersion,nof_group) 
                    record_sub_file = record_sub_folder + "\\" + record_sub_name + ".csv"
                    recorder_sub = Recorder(record_sub_file)
                    recorder_sub_head_args = (dimension,correlation,dispersion,nof_group,theoretically_sub_pc1_weight)
                    recorder_sub_head = "dim:%d,cor:%.4f,disp:%.4f,nofg:%d,tpw:%.4f, , \n"  %recorder_sub_head_args
                    recorder_sub_fields = "id,sub_angle,simu_data_sub_pc1_weight\n"
                    recorder_sub(recorder_sub_head)
                    recorder_sub(recorder_sub_fields)
                    for i in range(nof_case * len(ext_dimension_cor_incre_list)):
                        simu_data_sub = get_simu_data(cov_matrix,dispersion,nof_group,group_size)
                        simu_data_sub_pc1,simu_data_sub_pc1_weight = get_data_pc1(simu_data_sub)
                        sub_angle = get_angle(theoretically_sub_pc1,simu_data_sub_pc1)
                        sub_angle_list.append(sub_angle)
                        simu_data_sub_pc1_weight_list.append(simu_data_sub_pc1_weight)  
                        record_content_sub = "%d,%.4f,%.4f\n" %(i,sub_angle,simu_data_sub_pc1_weight)
                        recorder_sub(record_content_sub)
                sub_angles = numpy.array(sub_angle_list,dtype = "float")
                sub_angle_std = sub_angles.std()
                sub_angle_mean = sub_angles.mean()
                simu_data_sub_pc1_weights = numpy.array(simu_data_sub_pc1_weight_list,dtype = "float")
                simu_data_sub_pc1_weight_mean = simu_data_sub_pc1_weights.mean()
                simu_data_sub_pc1_weight_std = simu_data_sub_pc1_weights.std()
                recorder_sub_sta_line_args = (dimension,correlation,dispersion,nof_group,theoretically_sub_pc1_weight,
                                              sub_angle_mean,sub_angle_std,simu_data_sub_pc1_weight_mean,simu_data_sub_pc1_weight_std)
                recorder_sub_sta_line = "%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" %recorder_sub_sta_line_args
                recorder_sub_sta(recorder_sub_sta_line)
            
run_simulation()


