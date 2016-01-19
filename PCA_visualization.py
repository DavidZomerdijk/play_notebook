# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:14:51 2016

@author: davidzomerdijk
"""
import pickle
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

def kPCA_visualization2d(X, y):
   
    kpca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=10, n_components=2)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    class_1 = []
    class_0 = []
     
    for i in range(0, len(y)):
        
        if y[i] == 1:
            class_1.append( X_kpca[i] )
        else:
            class_0.append( X_kpca[i]  )
    
    class_0_x = []
    class_0_y = []
    class_1_x = []
    class_1_y = []
    for x in class_0:
        class_0_x.append( x[0] )
        class_0_y.append( x[1] )
        
    for x in class_1:
        class_1_x.append( x[0] )
        class_1_y.append( x[1] )
        

    # Plot
    #print principle component

    plt.title("kPCA kernel = linear")
    plt.plot( class_0_x, class_0_y, "ro")
    plt.plot( class_1_x, class_1_y, "go")
    plt.title("Projection by PCA")
    plt.xlabel("1st principal component")
    plt.ylabel("2nd component")
    

    
    plt.show()
    
def kPCA_visualization1d(X, y):
   
    kpca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=10, n_components=2)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    class_1 = []
    class_0 = []

    for i in range(0, len(y)):
        
        if y[i] == 1:
            class_1.append(  list( X_kpca[i] )[0] )
        else:
            class_0.append(  list( X_kpca[i] )[0] )
    print "check"
    print class_1[:10]
    import numpy
    from matplotlib import pyplot
    

    pyplot.hist(class_1, 50, alpha=0.5, label='class 1' )  
    pyplot.hist(class_0, 50, alpha=0.5, label='class 0')

    pyplot.legend(loc='upper right')
    pyplot.show()


    




    
    
    
