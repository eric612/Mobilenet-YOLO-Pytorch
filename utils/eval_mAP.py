import torch
from utils.iou import *
import torch.multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from itertools import product
import time

def eval_single_image_recall(this_true_labels,this_det_labels,true_box,true_difficultie,det_box,det_score):
    #print(true_boxes[num].shape)
    n_easy_object = 0
    #this_true_labels = (true_label == c)
    #this_det_labels = (det_label == c)
    #print(this_true_labels)
    true_class_boxes = true_box[this_true_labels]

    true_class_difficulties = true_difficultie[this_true_labels]
    n_easy_object += (1 - true_class_difficulties).sum()  # ignore difficult objects
    
    
    det_class_boxes = det_box[this_det_labels]  # (n_class_detections, 4)
    det_class_scores = det_score[this_det_labels]  # (n_class_detections)  
    n_class_detections = det_class_boxes.size(0)

    true_positive = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
    false_positive = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
    if n_class_detections == 0:
        #sharedlist.append([true_positive,false_positive,n_easy_object,det_class_scores])
        return (true_positive,false_positive,n_easy_object,det_class_scores)
        #print(true_positive,false_positive,n_easy_object)
        #return true_positive,false_positive,n_easy_object,det_class_scores
    true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects) 
    for d in range(n_class_detections):
        this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
        object_boxes = true_class_boxes

        object_difficulties = true_class_difficulties
        if object_boxes.size(0) == 0:
            false_positive[d] = 1
            continue
        # Find maximum overlap of this detection with objects in this image of this class
        overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
        max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
        
        
        # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
        # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
        original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[ind]
        # We need 'original_ind' to update 'true_class_boxes_detected'
        
        # If the maximum overlap is greater than the threshold of 0.5, it's a match
        if max_overlap.item() > 0.5:
            # If the object it matched with is 'difficult', ignore it
            if object_difficulties[ind] == 0:
            # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positive[d] = 1
                    true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positive[d] = 1
        # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
        else:
            false_positive[d] = 1                
    #sharedlist.append([true_positive,false_positive,n_easy_object,det_class_scores])
    return (true_positive,false_positive,n_easy_object,det_class_scores)
    #print(true_positive,false_positive,n_easy_object)
    #return true_positive,false_positive,n_easy_object,det_class_scores
    
def eval_class_ap(c,num_of_imgs,true_labels,det_labels,true_boxes,true_difficulties,det_boxes,det_scores):
    n_easy_class_objects = 0
    true_positives = torch.zeros(0, dtype=torch.float).to(device)  # (n_class_detections)
    false_positives = torch.zeros(0, dtype=torch.float).to(device)  # (n_class_detections) 
    det_class_scores_all = torch.zeros(0, dtype=torch.float).to(device)  # (n_class_detections) 
    #ctx = mp.get_context('spawn')
    #pool = ctx.Pool(processes=4)
    #class_labels = [c] * num_of_imgs
    #manager = ctx.Manager()
    #sharedlist= manager.list() 
    '''
    data = list()
    for class_label,true_label,det_label,true_boxe,true_difficultie,det_boxe,det_score in zip(class_labels,true_labels,det_labels,true_boxes,true_difficulties,det_boxes,det_scores):
        data.append([c,class_label,true_label,det_label,true_boxe,true_difficultie,det_boxe,det_score])
    results = pool.map(eval_single_image_recall,data)
    pool.close()
    pool.join()
    for result in results:
        true_positives = torch.cat((true_positives,result[0]),0)
        false_positives = torch.cat((false_positives,result[1]),0)
        n_easy_class_objects += result[2]
        det_class_scores_all = torch.cat((det_class_scores_all,result[3]),0)   
    '''
    
    for num in range(num_of_imgs):
        #print(true_boxes[num].shape)
        #eval_single_image_recall(sharedlist,c,true_labels[num],det_labels[num],true_boxes[num],true_difficulties[num],det_boxes[num],det_scores[num])
        true_positive,false_positive,n_easy_object,det_class_scores = eval_single_image_recall((true_labels[num] == c) ,(det_labels[num] == c) ,true_boxes[num],true_difficulties[num],det_boxes[num],det_scores[num])
        true_positives = torch.cat((true_positives,true_positive),0)
        false_positives = torch.cat((false_positives,false_positive),0)
        n_easy_class_objects += n_easy_object
        det_class_scores_all = torch.cat((det_class_scores_all,det_class_scores),0)
    '''
    for idx,(true_positive,false_positive,n_easy_object,det_class_scores) in enumerate(sharedlist):
        true_positives = torch.cat((true_positives,true_positive),0)
        false_positives = torch.cat((false_positives,false_positive),0)
        n_easy_class_objects += n_easy_object
        det_class_scores_all = torch.cat((det_class_scores_all,det_class_scores),0)        
    '''
    # Compute cumulative precision and recall at each detection in the order of decreasing scores
    #print(true_positives.shape)
    det_class_scores_all, sort_ind = torch.sort(det_class_scores_all, dim=0, descending=True)  # (n_class_detections)

    true_positives = true_positives[sort_ind]  # (n_class_detections)
    false_positives = false_positives[sort_ind]  # (n_class_detections, 4)    
    n_sum_true_positive = torch.sum(true_positives)
    n_sum_false_positive = torch.sum(false_positives)
    cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
    cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
    cumul_precision = cumul_true_positives / (
            cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
    cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

    # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
    recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
    precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = cumul_recall >= t
        if recalls_above_t.any():
            precisions[i] = cumul_precision[recalls_above_t].max()
        else:
            precisions[i] = 0.    
           
    return precisions.mean().item(),n_sum_true_positive,n_sum_false_positive
    
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties,classes_name):
    start_time  = time.process_time()
    n_classes = len(classes_name) 
    #print(n_classes)
    classes_map = {k: v  for v, k in enumerate(classes_name)}
    #classes_map['background'] = 0
    od_classes_map = {v: k for k, v in classes_map.items()}  # Inverse mapping
    
    """
    Calculate the Mean Average Precision (mAP) of detected objects.
    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation
    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    #print(len(det_boxes),len(det_labels),len(det_scores),len(true_boxes),len(true_labels),len(true_difficulties))
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    num_of_imgs = len(det_boxes)
    # print(len(det_boxes), len(det_labels), len(det_scores), len(true_boxes), len(true_labels), len(true_difficulties))

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    class_true_positive = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    class_false_positive = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    
    for c in range(1, n_classes):
        precision,n_sum_true_positive,n_sum_false_positive = eval_class_ap(c,num_of_imgs,true_labels,det_labels,true_boxes,true_difficulties,det_boxes,det_scores)

        average_precisions[c - 1] = precision
        class_true_positive[c - 1] = n_sum_true_positive
        class_false_positive[c - 1] = n_sum_false_positive
        
        #n_easy_class_objects = int(n_easy_class_objects)
    # Calculate Mean Average Precision (mAP)
    
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {od_classes_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    class_true_positive = {od_classes_map[c + 1]: v for c, v in enumerate(class_true_positive.tolist())}
    class_false_positive = {od_classes_map[c + 1]: v for c, v in enumerate(class_false_positive.tolist())}
    print("The time used to execute this is given below")

    end_time  = time.process_time()

    print(end_time - start_time )
    return average_precisions, mean_average_precision, class_true_positive, class_false_positive