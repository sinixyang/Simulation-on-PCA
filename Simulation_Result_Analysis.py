# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 15:59:30 2016

@author: Sinix Studio
"""

import numpy
import matplotlib
from  matplotlib import pyplot
from scipy import polyval, polyfit

#hyper parameters
matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'})
color_list = ["gray","rosybrown","brown","red","tomato","orangered","chocolate","saddlebrown",
              "darkorange","orange","gold","yellow","olive","lawngreen","darksage","green","lime",
              "lightseagreen","darkcyan","cyan","deepskyblue","steelblue","dodgerblue",
              "royalblue","navy","blue","slateblue","indigo","violet","purple","magenta","deeppink","cimson"
              ]
              

def load_simu_result(file_routine):
    #load the result fille "simu_ex.csv"
    f = open(file_routine,"r")
    nof_item = -1
    for l in f:
        nof_item = nof_item + 1
    
    f.seek(0)
    fields_str = f.readline()
    fields = fields_str[:-1].split(",")
    fdtype = map(lambda x: (x,'S10'),fields)
    simu_result = numpy.zeros(nof_item, dtype = fdtype)
    for i,l in enumerate(f):
        values = tuple(l[:-1].split(","))
        simu_result[i] = values
    return simu_result

def get_unique_values(simu_result,limit = 0):
    #collect result according to combination of parameters, only the first "limit" fields are included
    fields = simu_result.dtype.names
    if limit == 0:
        pass
    else:
        fields = fields[:limit]
    unique_value_dict = {}
    for field in fields:
        unique_value_dict[field] = []
    for item in simu_result:
        for field in fields:
            if item[field] not in unique_value_dict[field]:
                unique_value_dict[field].append(item[field])
    return unique_value_dict
        
def filter_data(simu_result,rule):
    #filter result simu_result according to the rule
    simu_result_len = len(simu_result)
    data_filter = numpy.ones(simu_result_len,dtype = "bool")
    for item in rule:
        field,value = item
        temp_filter = numpy.where(simu_result[field] == value,True,False)
        data_filter = temp_filter*data_filter
    return simu_result[data_filter]
    

def str_to_unicode(str_asc, mathtype_map = [("Alpha",u"α"),("Lambda",u"λ"),("Theta2",u"\\theta_2 "),("Theta3",u"\\theta_3 "),("Delta",u"δ"),("G", u"g")]):
    #convert string to unicode string that including mathtype
    str_uni = unicode(str_asc)
    str_uni = u"$" + str_uni + u"$"
    for comparison in mathtype_map:
        print comparison
        str_uni = str_uni.replace(comparison[0],comparison[1])
    return str_uni
        
def plot_series(simu_result,field_limit,series_field_name,fixed_fields,x_field_name,y_field_name, horizon = "",fitting = False,xscale_log = False, grid = True, yscale_log = False):
    #show series of lines according to the result
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    unique_value_dict = get_unique_values(simu_result,field_limit)
    series_field_values = unique_value_dict[series_field_name]
    color_index_base = int(len(color_list)/len(series_field_values))
    plot_title = u""
    for field_name,field_value in fixed_fields:
        plot_title = plot_title + unicode(field_name) + u"=" + unicode( field_value) + u";"
    plot_title = str_to_unicode(plot_title)
    for i,field_value in enumerate(series_field_values):
        filter_rule = [(series_field_name,field_value)] + list(fixed_fields)
        filtered_data = filter_data(simu_result,filter_rule)
        label = series_field_name + " = " + str(field_value)
        label = str_to_unicode(label)
        #labels.append(label)
        data_x = filtered_data[x_field_name].astype("f")
        data_y = filtered_data[y_field_name].astype("f")
        ax.plot(data_x, data_y, c = color_list[i*color_index_base],marker ="o",label = label,alpha = 1)
        if fitting == True:
            (ar,br)=polyfit(data_x,data_y,1)
            reg_y=polyval([ar,br],data_x)
            ax.plot(data_x, reg_y, c = color_list[i*color_index_base])
        if horizon != "":
            meanvalue = filtered_data[horizon].astype("f").mean()
            hor_y = data_x.copy()
            hor_y.fill(meanvalue)
            ax.plot(data_x, hor_y, c = color_list[i*color_index_base],linestyle = "--")
    if xscale_log:
        ax.set_xscale('log')
    if yscale_log:
        ax.set_yscale("log", basey=2)
        major_ticks = numpy.logspace(1, 6, 6,base = 2) # adjust y major ticks
        ax.set_yticks(major_ticks)                                               
        minor_ticks = numpy.logspace(1, 6, 11,base = 2) # adjust y minor ticks
        ax.set_yticks(minor_ticks,minor=True)
    ax.legend(title = plot_title,loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':10})
    y_field_name =  str_to_unicode(y_field_name)
    x_field_name =  str_to_unicode(x_field_name)
    ax.set_xlabel(x_field_name)
    ax.set_ylabel(y_field_name)
    if grid == True:
        ax.grid(which='both')   
    pyplot.show()

def plot_cdmap(simu_result,field_limit,fix_field ,zx_field_name,zy_field_name,zh_field_name,x_field_name,y_field_name,delta = 0,x_log = False, y_log = False):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    unique_value_dict = get_unique_values(simu_result,field_limit)
    x_values = unique_value_dict[x_field_name]
    y_values = unique_value_dict[y_field_name]
    title = "CASE: "
    for field in fix_field:
        title = title + field[0] + "=" + field[1] + ", " 
    title = title + "Delta = " + str(delta)
    title =  str_to_unicode(title)
    title = "   " + title
    dx,dy,dz = [],[],[]
    for x_value in x_values:
        for y_value in y_values:
            if x_log == True:
                x = numpy.log(float(x_value))
                
            else:
                x = float(x_value)
            if y_log == True:
                y = numpy.log(float(y_value))
                y_label = "$\\log_2(\\lambda)$"
            else:
                y = float(y_value)
            x_field = (x_field_name,x_value)
            y_field = (y_field_name,y_value)
            
            filter_rule =  [x_field,y_field] + fix_field
            filtered_data = filter_data(simu_result,filter_rule)
            data_x = filtered_data[zx_field_name].astype("f")
            data_y = numpy.log2(filtered_data[zy_field_name].astype("f"))
            data_h = numpy.log2(filtered_data[zh_field_name].astype("f").mean())
            (ar,br)=polyfit(data_x,data_y,1)
            data_z=polyval([ar,br],delta)    
            data_z = 2**data_z - 2**data_h
            dx.append(x)
            dy.append(y)
            dz.append(data_z)
    dx = numpy.array(dx)
    dy = numpy.array(dy)
    dz = numpy.array(dz)
    mask_plus = dz >=0
    mask_neg = ~mask_plus
    dz = (numpy.abs(dz) + 0.1)*15
    scatter_plus = ax.scatter(dx[mask_plus],dy[mask_plus],marker='o',s = dz[mask_plus],c = "r",linewidth = 0)
    scatter_neg = ax.scatter(dx[mask_neg],dy[mask_neg],marker='o',s = dz[mask_neg],c = "b",linewidth = 0)
    ax.set_xlabel(str_to_unicode(x_field_name),size = "large" )
    ax.set_ylabel(y_label,size = "large")
    if mask_neg.sum() == 0:
        ax.legend([scatter_plus, (scatter_plus)], ["$\\theta_\\Delta (Plus) $" + title], scatterpoints=1,ncol=1,fontsize=12)
    elif mask_plus.sum() == 0:
        ax.legend([scatter_neg, (scatter_neg)], ["$\\theta_\\Delta (Negative)$" + title], scatterpoints=1,ncol=1,fontsize=12)
    else:
        ax.legend([scatter_neg, (scatter_neg,scatter_plus)], ["$\\theta_\\Delta (Negative) $","$\\theta_\\Delta (Plus)$" + title],
                   scatterpoints=1,ncol=3,fontsize=12, handletextpad  = 0, columnspacing = 0)
    pyplot.show()
                
#Parameter range
#Alpha = ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
#Lambda = ['0.0156','0.0312','0.0625','0.125','0.25','0.5','1','2','4','8','16','32','64']
#G = ['2', '4', '8', '16', '32', '64', '128', '256']
#Delta = ['-0.175','-0.1312','-0.0875','-0.0438','0','0.0438','0.0875','0.1312','0.175']

##PLOT FIGTURE 3
file_routine = r"D:\Cloud\OneDrive - Five Stars Activation Group\PaperToWrite\PCA simulation\PCA_simu\analysis\pca_simu_sub_pub.csv"
simu_result = load_simu_result(file_routine)
plot_series(simu_result,5,"Lambda",[("G","256")],"Alpha","Theta2",horizon = "", grid = True, fitting = False,yscale_log = True)

##PLOT FIGTURE 4
file_routine = r"D:\Cloud\OneDrive - Five Stars Activation Group\PaperToWrite\PCA simulation\PCA_simu\analysis\pca_simu_ex_pub.csv"
simu_result = load_simu_result(file_routine)
plot_series(simu_result,5,"Alpha",[("G","32"),("Lambda","1")],"Delta","Theta3",horizon = "Theta2", grid = False, fitting = False, yscale_log = True)

##PLOT FIGTURE 5
file_routine = r"D:\Cloud\OneDrive - Five Stars Activation Group\PaperToWrite\PCA simulation\PCA_simu\analysis\pca_simu_ex_pub.csv"
simu_result = load_simu_result(file_routine)
plot_cdmap(simu_result = simu_result, field_limit =5, fix_field = [("G","4")], zx_field_name = "Delta" , zy_field_name = "Theta3", zh_field_name = "Theta2",
           x_field_name = "Alpha", y_field_name = "Lambda", delta = -0.08, x_log = False, y_log = True)
