def wh_to_x2y2(bbox):
    bbox[...,0] = bbox[...,0] - bbox[...,2]/2
    bbox[...,1] = bbox[...,1] - bbox[...,3]/2
    bbox[...,2] = bbox[...,2] + bbox[...,0]
    bbox[...,3] = bbox[...,3] + bbox[...,1]