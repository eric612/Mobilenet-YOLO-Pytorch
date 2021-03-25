# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
import numpy as np
import torch
import random
import torchvision.transforms.functional as FT
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    # #box iou
    # output = intersection/ areas_set_2

    return intersection / union  # (n1, n2)

class Image_Augmentation():

    def expand_od(self,image, boxes, filler):
        """
        Perform a zooming out operation by placing the image in a larger canvas of filler material.

        Helps to learn to detect smaller objects.

        :param image: image, a tensor of dimensions (3, original_h, original_w)
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :param filler: RBG values of the filler material, a list like [R, G, B]
        :return: expanded image, updated bounding box coordinates
        """
        # Calculate dimensions of proposed expanded (zoomed-out) image
        original_h = image.size(1)
        original_w = image.size(2)
        max_scale = 4
        scale = random.uniform(1, max_scale)
        new_h = int(scale * original_h)
        new_w = int(scale * original_w)

        # Create such an image with the filler
        filler = torch.FloatTensor(filler)  # (3)
        new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
        # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
        # because all expanded values will share the same memory, so changing one pixel will change all

        # Place the original image at random coordinates in this new image (origin at top-left of image)
        left = random.randint(0, new_w - original_w)
        right = left + original_w
        top = random.randint(0, new_h - original_h)
        bottom = top + original_h
        new_image[:, top:bottom, left:right] = image

        # Adjust bounding boxes' coordinates accordingly
        new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)  # (n_objects, 4), n_objects is the no. of objects in this image

        return new_image, new_boxes

    def random_crop_od(self,image, boxes, labels, difficulties):
        """
        Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

        Note that some objects may be cut out entirely.

        Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

        :param image: image, a tensor of dimensions (3, original_h, original_w)
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :param labels: labels of objects, a tensor of dimensions (n_objects)
        :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
        :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
        """
        original_h = image.size(1)
        original_w = image.size(2)
        # Keep choosing a minimum overlap until a successful crop is made
        while True:
            # Randomly draw the value for minimum overlap
            min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping
            #min_overlap = min(abs(random.gauss(0, 0.6)),.9)
            #min_overlap = random.choice([min_overlap,min_overlap/2, None])
            #print(min_overlap)
            #min_overlap = random.choice([0.,0.,.1,.1,.3,.5,.7, None])  # 'None' refers to no cropping
            # If not cropping
            if min_overlap is None:
                return image, boxes, labels, difficulties

            # Try up to 50 times for this choice of minimum overlap
            # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
            max_trials = 50
            for _ in range(max_trials):
                # Crop dimensions must be in [0.3, 1] of original dimensions
                # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
                min_scale = 0.3
                scale_h = random.uniform(min_scale, 1)
                scale_w = random.uniform(min_scale, 1)
                new_h = int(scale_h * original_h)
                new_w = int(scale_w * original_w)

                # Aspect ratio has to be in [0.5, 2]
                aspect_ratio = new_h / new_w
                if not 0.5 < aspect_ratio < 2:
                    continue

                # Crop coordinates (origin at top-left of image)
                left = random.randint(0, original_w - new_w)
                right = left + new_w
                top = random.randint(0, original_h - new_h)
                bottom = top + new_h
                crop = torch.FloatTensor([left, top, right, bottom])  # (4)

                # Calculate Jaccard overlap between the crop and the bounding boxes
                overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                               boxes)  # (1, n_objects), n_objects is the no. of objects in this image
                overlap = overlap.squeeze(0)  # (n_objects)

                # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
                if overlap.max().item() < min_overlap:
                    continue

                # Crop image
                new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

                # Find centers of original bounding boxes
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

                # Find bounding boxes whose centers are in the crop
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

                # If not a single bounding box has its center in the crop, try again
                if not centers_in_crop.any():
                    continue

                # Discard bounding boxes that don't meet this criterion

                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
                new_boxes[:, :2] -= crop[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
                new_boxes[:, 2:] -= crop[:2]

                return new_image, new_boxes, new_labels, new_difficulties

    def flip_od(self,image, boxes):
        """
        Flip image horizontally.

        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :return: flipped image, updated bounding box coordinates
        """
        # Flip image
        new_image = FT.hflip(image)

        # Flip boxes
        new_boxes = boxes
        new_boxes[:, 0] = image.width - boxes[:, 0] - 1
        new_boxes[:, 2] = image.width - boxes[:, 2] - 1
        new_boxes = new_boxes[:, [2, 1, 0, 3]]

        return new_image, new_boxes

    def resize_od(self,image, boxes, dims=(300, 300), return_percent_coords=True):
        """
        Resize image. For the SSD300, resize to (300, 300).

        Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
        you may choose to retain them.

        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
        """
        # Resize image
        # new_image = FT.resize(image, dims)
        new_image = FT.to_tensor(image)
        new_image = torch.nn.functional.interpolate(new_image.unsqueeze(0), size=dims[0], mode='bilinear',align_corners=False).squeeze(0)
        new_image = FT.to_pil_image(new_image)


        # Resize bounding boxes
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        new_boxes = boxes / old_dims  # percent coordinates

        if not return_percent_coords:
            new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
            new_boxes = new_boxes * new_dims

        return new_image, new_boxes

    def photometric_distort(self,image):
        """
        Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

        :param image: image, a PIL Image
        :return: distorted image
        """
        new_image = image

        distortions = [FT.adjust_brightness,
                       FT.adjust_contrast,
                       FT.adjust_saturation,
                       FT.adjust_hue,
                       FT.adjust_gamma]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ is 'adjust_hue':
                    # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                    adjust_factor = random.uniform(0.5, 1.5)

                # Apply this distortion
                new_image = d(new_image, adjust_factor)

        return new_image

    def transform_od(self,image, boxes, labels, difficulties, dims, mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225],phase = 'train'):
        """
        Apply the transformations above.

        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :param labels: labels of objects, a tensor of dimensions (n_objects)
        :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
        :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
        :param dims: (H, W)
        :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
        """
        assert phase in {'train', 'test'}

        # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
        # see: https://pytorch.org/docs/stable/torchvision/models.html
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        new_image = image
        new_boxes = boxes
        new_labels = labels
        new_difficulties = difficulties
        # Skip the following operations if validation/evaluation
        if phase == 'train':
            # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
            new_image = self.photometric_distort(new_image)

            # Convert PIL image to Torch tensor
            new_image = FT.to_tensor(new_image)
            
            # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
            # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
            if random.random() < 0.5:
                new_image, new_boxes = self.expand_od(new_image, boxes, filler=mean)

            # Randomly crop image (zoom in)
            new_image, new_boxes, new_labels, new_difficulties = self.random_crop_od(new_image, new_boxes, new_labels,
                                                                                new_difficulties)

            # Convert Torch tensor to PIL image
            new_image = FT.to_pil_image(new_image)

            # Flip image with a 50% chance
            if random.random() < 0.5:
                new_image, new_boxes = self.flip_od(new_image, new_boxes)

        # Resize image to (H, W) - this also converts absolute boundary coordinates to their fractional form
        #new_image, new_boxes = self.resize_od(new_image, new_boxes, dims=dims)
        '''
        draw = ImageDraw.Draw(new_image)     
        for i in range(len(new_boxes)):
            # Boxes
            box_location = (new_boxes[i]*dims[0]).tolist()
            #print(box_location)
            draw.rectangle(xy=box_location,outline='#e6194b')
            #draw.rectangle([0,0,1,1])
            #draw.rectangle(xy=[l + 1. for l in box_location])  # a second rectangle at an offset of 1 pixel to increase line thickness
        del draw 
        cv2.imwrite('frame.jpg', cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR))
        '''
             
        # Convert PIL image to Torch tensor
        #new_image = FT.to_tensor(new_image)


        # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
        #new_image = FT.normalize(new_image, mean=mean, std=std)

        return new_image, new_boxes, new_labels, new_difficulties