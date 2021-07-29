#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:15:43 2021

@author: theresasawi
"""


def setParams(key):
    
    
    ##EXAMPLE##
    ##must change for each example
    if key == 'Parkfield_Repeaters':
        
        pathProj = f'/Users/theresasawi/Documents/12_Projects/Parkfield_Repeaters/'
        
        pathCat = pathProj + 'data_raw/catall.txt'
        
        pathWF = pathProj + 'data_raw/BAV.waves/'
        
        network = 'NC'
        
        station = 'BAV'
        
        channel = 'EHZ'
        
        channel_ID = 0 # the index number for an obspy stream object
        
        filetype = '.mseed'
    
        cat_columns =  ['event_ID','lat','long','depthOrMag','year','month','day','hour','minute','second','magnitudeOrDepth']
    



    
    return pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns



def setSgramParams(key):

    """
    set for each project
    """
    print( ' key : ', key)
    if key == "Parkfield_Repeaters":
        fmin            = 8
        fmax            = 40
        winLen_Sec      = .2#seconds
        fracOverlap     = 1/4
        nfft            = 2**10 #padding
        
        
        
    return fmin, fmax, winLen_Sec, fracOverlap, nfft

    