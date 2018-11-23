from region_selection import  *
from svm_classification import trainSVM
from vgg_output import get_vgg_output



def get_overlap_area_ratio(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    area_rect1 = w1*h1
    area_rect2 = w2*h2
    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    if (dx>=0) and (dy>=0):
        return (dx*dy)/(area_rect1+area_rect2-dx*dy)
    else:
        '''
            -1 implies that there is no common area.
        '''
        return -1


def maximum_region_selector(svm_classifier,im,rect_list,area_threshold = 0.3,score_threshold = 1):
    '''
        region info is a dict with key as region rect and value is the class predicted by the 
    '''
    
    vgg_output = get_vgg_output(im,rect_list)

    region_outputs = svm_classifier.predict(vgg_output)
    region_outputs = region_outputs.astype(int)
    print(region_outputs,region_outputs.shape)

    score_regions = svm_classifier.decision_function(vgg_output)
    print(score_regions,score_regions.shape)
    '''
        storing indices of rect_list in final_regions
    '''
    final_regions = []

    for i in range(len(rect_list)):

        check = True
        for j in final_regions:
            if(score_regions[i][region_outputs[i]] < score_threshold):
                break
            if(region_outputs[j] == region_outputs[i]):
                overlap_area_ratio = get_overlap_area_ratio(rect_list[i],rect_list[j])
                '''
                    first check if score is greater than the image considered
                    if overlap_area = -1 implies no common area.
                    else check if the overlap area is greater than 
                '''
                if(overlap_area_ratio > area_threshold):
                    print(region_outputs[i])
                    if(score_regions[i][region_outputs[i]] < score_regions[j][region_outputs[j]]):
                        '''
                            remove j index from the final list and add i.
                        '''
                        check = False
                    else:
                        final_regions.remove(j)
        if(check == True):
            final_regions.append(i)

    for i in final_regions:
        x,y,w,h = rect_list[i]
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('output',im)
    cv2.waitKey(0)

    result = []
    for i in final_regions:
        result.append((rect_list[i],region_outputs[i],score_regions[i][region_outputs[i]]))
    return result                        
            
if __name__ == '__main__':
    pass




