import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import datetime


def velocity_calc(x1,x2,delta_t):
    ''' calculate velocity using the displacements and time'''
    vel = (x2-x1) / delta_t
    return vel

def acceleration_calc(v1,v2,delta_t):
    ''' calculate acceleration using the displacements and time'''
    acc = (v2-v1) / delta_t
    return acc

def average(X,df_time,df_label):
    ''' Calculate average distance travelled during a time interval '''
    avg=[]
    for i,j in enumerate(X) :
        temp = []
        if list(df_time["UTC time"])[i] in list(df_label["UTC time"]):
            temp = X[:i+1]
            avg.append(np.mean(temp))
    return avg



def modify_data(df_time,df_labels):
    ''' Modifies the data in the times and labels data, adds a date column consisting of the dates, adds velocity and accelration data in the dataset too'''
    
    df_labels["UTC time"] = [datetime.datetime.strptime(i, "%Y-%m-%dT%H:%M:%S.%f") for i in list(df_labels["UTC time"])]
    df_time["UTC time"] = [datetime.datetime.strptime(i, "%Y-%m-%dT%H:%M:%S.%f") for i in list(df_time["UTC time"])]
            
    df_labels["x_avg"] = average(df_time.x,df_time,df_labels)
    df_labels["y_avg"] = average(df_time.y,df_time,df_labels)
    df_labels["z_avg"] = average(df_time.z,df_time,df_labels)
    
    delta_ts = [(df_labels["UTC time"][i]-df_labels["UTC time"][i-1]).total_seconds() for i in range(1,len(df_labels["UTC time"]))]
    delta_ts.insert(0,0)
    
    velocity_x = [velocity_calc( df_labels.x_avg[i-1], df_labels.x_avg[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    velocity_x.insert(0,0)
    
    velocity_y = [velocity_calc( df_labels.y_avg[i-1], df_labels.y_avg[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    velocity_y.insert(0,0)
    
    velocity_z = [velocity_calc(df_labels.z_avg[i-1], df_labels.z_avg[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    velocity_z.insert(0,0)
    
    accelartion_x = [acceleration_calc(velocity_x[i-1], velocity_x[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    accelartion_x.insert(0,0)
    
    accelartion_y = [acceleration_calc(velocity_y[i-1], velocity_y[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    accelartion_y.insert(0,0)
    
    accelartion_z = [acceleration_calc(velocity_z[i-1], velocity_z[i], delta_ts[i]) for i in range(1,len(df_labels.x_avg))]
    accelartion_z.insert(0,0)
    
    df_labels["acceleration_x"] = accelartion_x
    df_labels["acceleration_y"] = accelartion_y
    df_labels["acceleration_z"] = accelartion_z
