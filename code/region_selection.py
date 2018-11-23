import sys
import cv2

#im = image // method='f'/'q'
def select_regions(im, method='f'):
    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    # resize image
    # newHeight = 400
    # newWidth = int(im.shape[1]*400/im.shape[0])
    # im = cv2.resize(im, (newWidth, newHeight))
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (method == 'f'):
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
