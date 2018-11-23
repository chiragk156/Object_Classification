#!/usr/bin/env python

 
import sys
import cv2


def select_regions(image, method='f'):
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);
 
    # read image
    im = cv2.imread(image)
    # resize image
    newHeight = 400
    newWidth = int(im.shape[1]*400/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (method):
        ss.switchToSelectiveSearchFast(sigma = 0.5)
 
    # Switch to high recall but slow Selective Search method
    elif (method == 'q'):
        ss.switchToSelectiveSearchQuality(sigma = 0.5)
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)
 
    # run selective search segmentation on input image
    rects = ss.process()
    return rects
