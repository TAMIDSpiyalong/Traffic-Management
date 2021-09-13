import numpy as np

def iou (boxes1, boxes2):
    '''
    boxes1: m x 4 numpy array
    boxes2: n x 4 numpy array
    '''
    boxes1 = np.array(boxes1, dtype='float32')
    boxes2 = np.array(boxes2, dtype='float32')
    
    m = boxes1.shape[0] # number of boxes1
    n = boxes2.shape[0] # number of boxes2
    
    boxes1_area = (boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])
    boxes1_area = boxes1_area.repeat(n).reshape((m,n)) # converts to mxn matrix
    
    boxes2_area = (boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])
    boxes2_area = np.tile(boxes2_area, (1,m)).reshape((m,n)) # converts to mxn matrix
    
    boxes1 = np.tile(boxes1, (1,n)).reshape((m,n,4))
    boxes2 = np.tile(boxes2, (m,1)).reshape((m,n,4))
    
    top = np.maximum(boxes1[:,:,:2],boxes2[:,:,:2])
    bot = np.minimum(boxes1[:,:,2:],boxes2[:,:,2:])
    
    diff = bot - top
    diff[diff<0] = 0
    intersection_area = diff[:,:,0] * diff[:,:,1]
    union_area = boxes1_area + boxes2_area - intersection_area
    
    # avoid division by zero
    idx = np.logical_or(boxes1_area==0, boxes2_area==0)
    union_area[idx] = 1
    
    return intersection_area/union_area

def is_TP (ious, iou_threshold=0.5):
    '''
    INPUT:
        m x n numpy array.
        - IoU between m detected boxes and n groud truth boxes
        - m detected boxes are sorted in descending order of confidence
    OUTPUT:
        m x 1 boolean array 
        - indicates if corresponding detected box is true positve
    '''
    m, n = ious.shape
    
    result = np.zeros(m,dtype=bool) # to store the result
    
    for i in range(m):
        idx = np.argmax( ious[i,:] ) # index of the max iou
        if ious[i,idx] >= iou_threshold:
            result[i] = True
            ious[:,idx] = -1 # turn off the ground truth box that is already detected
            
    return result




def evaluate (groundtruths, detections, included_class_names):
    '''
    groundtruths['image_name']: 
        shape = (m, 1+4) 
        [class_id, x0, y0, x1, y1]
                  
    detections['image_name']  : 
        shape=(n,1+4+1) 
        [class_id, x0, y0, x1, y1, confidence]
    '''
    
    auc       = {c: 0 for c in included_class_names}
    precision = {c:[] for c in included_class_names}
    recall    = {c:[] for c in included_class_names}
    real_precision = {c:0 for c in included_class_names}
    real_recall    = {c:0 for c in included_class_names}
    
    for c in included_class_names:
        detections_tps = np.array([])
        detections_confs = []
        num_gt = 0
        num_dt = 0
        for i in groundtruths:
            if groundtruths[i]==0:continue
            bx_gt = np.array(groundtruths[i])
            bx_gt = bx_gt[ bx_gt[:,0] == c,: ]
            num_gt += len(bx_gt)
            bx_dt = np.array(detections[i])
            if bx_dt.shape[0] == 0: continue
            bx_dt = bx_dt[ bx_dt[:,0]==c,: ] 
            num_dt = num_dt + len(bx_dt)
            if bx_dt.shape[0] == 0: continue
            if bx_gt.shape[0] != 0:
                
                ious = iou(bx_dt[:,1:5] , bx_gt[:,1:5] )
                tps  = is_TP(ious)
            else:
                tps = np.zeros(len(bx_dt))
            
            confs = bx_dt[:,-1]
            detections_tps = np.append(detections_tps,tps)
            detections_confs =  np.append(detections_confs,confs)

        # sort
        idc = np.argsort(detections_confs)[::-1]
        detections_tps = detections_tps[idc]
        
        num_tp = 0
        for i, tp in enumerate(detections_tps):
            if tp: num_tp += 1
            recall[c].append( num_tp/num_gt+0.000000000001 )
            precision[c].append( num_tp/(i+1) )
        if num_gt==0 or len(bx_dt)==0:
            continue
        else:
            real_precision[c] = num_tp/(num_dt)
            real_recall[c] = num_tp/(num_gt)
        
        for i in range(len(precision[c])):
            precision[c][i] = max(precision[c][i:])
        for i in range(1,len(precision[c])):
            auc[c] += precision[c][i] * ( recall[c][i]-recall[c][i-1] )
            
    for c in included_class_names:
#         recall[c].append(recall[c][-1])
        recall[c].append(1.0)
#         precision[c].append(0.0)
        precision[c].append(0.0)
    
#     real_auc ={}
#     for each in auc:
#         if auc[each]>0.01:
#             real_auc.update({each:auc[each]})
    real_auc=auc
    m_a_p = sum(real_auc.values())/len(real_auc)
    return m_a_p, real_auc, precision, recall,real_precision,real_recall