# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:03:40 2016

@author: davidzomerdijk
"""


#gives accuracy of the trained model
def accuracy(y_test, y_predict):
    good_predict = 0
    for i in range(0, len(y_test)):
        if( int( float( str( y_test[i]))) == int( float( str( y_predict[i] )))):
            good_predict += 1
        
    return (1.0 * good_predict)/len(y_test) 
    
#plot ROC curve