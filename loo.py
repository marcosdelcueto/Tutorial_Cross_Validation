#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
######################################################################################################
def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    x1,x2,f=generate_data(-10,10,1.0,0.2)
    # Prepare X and y for KRR
    X,y = prepare_data_to_KRR(x1,x2,f)
    hyperparams = (0.01,1.5)
    KRR_function(hyperparams,X,y)
######################################################################################################
def generate_data(xmin,xmax,Delta,noise):
    # Calculate f=sin(x1)+cos(x2)
    x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    x2 = np.arange(xmin,xmax+Delta,Delta)   # generate x2 values from xmin to xmax
    x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points
    f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid
    # Add random noise to f
    random.seed(2020)                       # set random seed for reproducibility
    for i in range(len(f)):
        for j in range(len(f[0])):
            f[i][j] = f[i][j] + random.uniform(-noise,noise)  # add random noise to f(x1,x2)
    return x1,x2,f
######################################################################################################
def prepare_data_to_KRR(x1,x2,f):
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    y=f.flatten()
    X=np.array(X)
    y=np.array(y)
    return X,y
######################################################################################################
def KRR_function(hyperparams,X,y):
    # Assign hyper-parameters
    alpha_value,gamma_value = hyperparams
    # Split data into test and train: random state fixed for reproducibility
    loo = LeaveOneOut()
    y_pred_total = []
    y_test_total = []
    # kf-fold cross-validation loop
    for train_index, test_index in loo.split(X):
        #print('NEW FOLD')
        #print('train_index:', train_index)
        #print('test_index:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print('y_test')
        #print(sorted(y_test))
        # Scale X_train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Fit KRR with (X_train_scaled, y_train), and predict X_test_scaled
        KRR = KernelRidge(kernel='rbf',alpha=alpha_value,gamma=gamma_value)
        y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)
    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    diff_list = [np.abs(a_i - b_i) for a_i, b_i in zip(y_pred_total, y_test_total)]
    print('mean: %.4f' % np.mean(diff_list))
    print('var: %.4f' % np.var(diff_list))
    print('stdev: %.4f' % statistics.stdev(diff_list))
    #print('y_test_total')
    #print(y_test_total)
    #print('y_pred_total')
    #print(y_pred_total)
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    r_pearson,_=pearsonr(y_test_total,y_pred_total)
    print('alpha: %.2f . gamma: %.2f . rmse: %.4f . r: %.4f' %(alpha_value,gamma_value,rmse,r_pearson)) # Uncomment to print intermediate results
    plot_scatter(y_test_total,y_pred_total)
    return rmse
######################################################################################################
def plot_scatter(x,y):
    x = np.array(x)
    y = np.array(y)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    r, _ = pearsonr(x, y)
    rmse = np.sqrt(mean_squared_error(x,y))
    ma = np.max([x.max(), y.max()]) + 0.1
    mi = np.min([x.min(), y.min()]) - 0.1
    ax = plt.subplot(gs[0])
    ax.scatter(x, y, color="C0")
    #ax.tick_params(axis='both', which='major', direction='in', labelsize=22, pad=10, length=5)
    ax.set_xlabel(r"Actual $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_ylabel(r"Predicted $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    ax.set_aspect('equal')
    ax.plot(np.arange(mi, ma + 0.1, 0.1), np.arange(mi, ma + 0.1, 0.1), color="k", ls="--")
    ax.annotate(u'$RMSE$ = %.4f' % rmse, xy=(0.15,0.85), xycoords='axes fraction', size=12)
    file_name="prediction_loo.png"
    plt.savefig(file_name,dpi=600,bbox_inches='tight')
######################################################################################################
main()
