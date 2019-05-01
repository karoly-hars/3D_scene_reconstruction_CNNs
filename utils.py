import math
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colormap import rgb2hex

HEIGHT = 320
WIDTH = 320

CLASSES = {0: "invalid", 1: "flat", 2: "constructions", 3: "street furnitures",
           4: "vegetation", 5: "sky", 6: "humans", 7: "vehicles"}
CLASS_COLORS = {0: [0, 0, 0], 1: [1, 1, 0], 2: [1, 0.5, 0], 3: [0, 0, 1],
                4: [0, 1, 0], 5: [0, 1, 1], 6: [1, 0.4, 1], 7: [1, 0, 0]}


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def scale_and_crop_img(img):
    img = img[..., ::-1]
    # resizing
    scale = max(WIDTH/img.shape[1], HEIGHT/img.shape[0])
    img = cv2.resize(img, (math.ceil(img.shape[1]*scale), math.ceil(img.shape[0]*scale)))
                                        
    # center crop to input size
    y_crop = img.shape[0] - HEIGHT
    x_crop = img.shape[1] - WIDTH
    img = img[math.floor(y_crop/2):img.shape[0]-math.ceil(y_crop/2),
              math.floor(x_crop/2):img.shape[1]-math.ceil(x_crop/2)]
    return img
    
    
def transform_img(img):   
    img = data_transform(img)
    img = img[None, :, :, :]
    return img


def load_img(img_path):
    img = cv2.imread(img_path)
    img = scale_and_crop_img(img)
    img = transform_img(img)
    return img
    

def correct_img(img):
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (std * img + mean)
    img = np.clip(img, 0, 1)
    return img
    

def show_img_preds(img, depth_pred, seg_pred, uc_th=0.0, apply_depth_mask=False):  
    plt.figure(0, figsize=(8, 6))
    
    # plot input img
    plt.subplot(2, 3, 1)
    plt.title("RGB")
    img = correct_img(img)
    plt.imshow(img)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])
    
    # plot depth image
    plt.subplot(2, 3, 2)
    plt.title("depth estimation")
    depth_pred = depth_pred[0, :, :]
    plt.imshow(depth_pred)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])
    
    # plot segmentation
    plt.subplot(2, 3, 3)
    plt.title("segmentation")
    seg_labels = np.argmax(seg_pred, 0)+1
    mask = np.zeros(shape=(seg_labels.shape[0], seg_labels.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_labels, np.asarray(key))
        mask[:, :, 0] += class_mask*CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask*CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask*CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    # plot masked depth image
    plt.subplot(2, 3, 5)
    plt.title("masked de")
    if apply_depth_mask:
        # mask high gradient regions ~ these are usually not as accurate
        grad = np.asarray(np.gradient(depth_pred))
        grad = np.abs(grad[0, :, :]) + np.abs(grad[1, :, :])
        grad_mask = grad < 0.9
        
        depth_mask = depth_pred < 50.0  # mask everything that is farther than 50m
        depth_pred = depth_pred * depth_mask * grad_mask
        
    plt.imshow(depth_pred)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    # plot masked seg
    plt.subplot(2, 3, 6)
    plt.title("masked seg")
    # mask out pixels where the certainty of the class prediction is lower than the uc_threshold
    uc = np.max(seg_pred, 0)
    uc_mask = uc > uc_th
    seg_labels = np.argmax(seg_pred, 0)+1
    seg_labels *= uc_mask
    mask = np.zeros(shape=(seg_labels.shape[0], seg_labels.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_labels, np.asarray(key))
        mask[:, :, 0] += class_mask*CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask*CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask*CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])
   
    plt.draw()

  
def show_point_cloud(img, depth_pred, seg_pred, f_len, uc_th=0.0, apply_depth_mask=False):
    
    img = correct_img(img)
    depth_pred = np.transpose(depth_pred, (1, 2, 0))
    depth_pred = depth_pred[:, :, 0]

    if apply_depth_mask:
        # mask high gradient regions ~ these are usually not as accurate
        grad = np.asarray(np.gradient(depth_pred))
        grad = np.abs(grad[0, :, :]) + np.abs(grad[1, :, :])
        grad_mask = grad < 0.95
        
        depth_mask = (depth_pred < 50.0)*(depth_pred > 5.0)  # mask everything that is farther than 50m
        depth_pred = depth_pred * depth_mask * grad_mask

    # mask out pixels where the certainty of the class prediction is lower than the uc_threshold
    uc = np.max(seg_pred, 0)
    uc_mask = uc > uc_th
    seg_pred = np.argmax(seg_pred, 0)+1
    seg_pred *= uc_mask
    mask = np.zeros(shape=(seg_pred.shape[0], seg_pred.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_pred, np.asarray(key))
        mask[:, :, 0] += class_mask*CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask*CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask*CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    mask = (img*0.7)+(mask*0.3)
    
    # generate 3D points
    x = []
    y = []
    z = []
    colors = []
    idx = 0
    for i in range(depth_pred.shape[0]):
        for j in range(depth_pred.shape[1]):
            idx += 1
            # if the distance is too large or small, skip
            if depth_pred[i, j] > 50.0 or depth_pred[i, j] < 5.0:
                continue
            # if the pixel is classified as sky or if its uncertain, skip
            if seg_pred[i, j] == 5 or seg_pred[i, j] == 0:
                continue
            # only show every 2nd pixel
            if idx % 2 == 1:
                continue
            
            z.append(depth_pred[i, j])
            y.append(i*depth_pred[i, j]/f_len)
            x.append((-160)+j*depth_pred[i, j]/f_len)
            
            # color based on mask (0.7*pixel color + 0.3*label color)
            r, g, b = int(mask[i, j][0]*255), int(mask[i, j][1]*255), int(mask[i, j][2]*255)
            colors.append(rgb2hex(r, g, b))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=colors, marker=",", s=5)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.view_init(elev=-37., azim=-117.)

    plt.draw()
    plt.show()
