import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
from colormap import rgb2hex


HEIGHT = 320
WIDTH = 320


CLASSES = {
    0: 'invalid',
    1: 'flat',
    2: 'construction',
    3: 'street furniture',
    4: 'vegetation',
    5: 'sky',
    6: 'human',
    7: 'vehicle'
}


CLASS_COLORS = {
    0: [0, 0, 0],
    1: [1, 1, 0],
    2: [1, 0.5, 0],
    3: [0, 0, 1],
    4: [0, 1, 0],
    5: [0, 1, 1],
    6: [1, 0.4, 1],
    7: [1, 0, 0]
}


def scale_image(img, scale=None):
    """Resize/scale an image. If a scale is not provided, scale it closer to HEIGHT x WIDTH."""
    # if scale is None, scale to the longer size
    if scale is None:
        scale = max(WIDTH / img.shape[1], HEIGHT / img.shape[0])

    new_size = (math.ceil(img.shape[1] * scale), math.ceil(img.shape[0] * scale))
    image = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
    return image


def center_crop(img):
    """Center crop and image to HEIGHT x WIDTH."""
    corner = ((img.shape[0] - HEIGHT) // 2, (img.shape[1] - WIDTH) // 2)
    img = img[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
    return img


def img_transform(img):
    """Normalize and image."""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = data_transform(img)
    return img


def create_plots(img, depth_pred, seg_pred, focal_len, uncertainty_threshold, apply_depth_mask):
    """Visualize the network output by creating a 2D plot and a 3D segmented pointcloud."""
    # create 2D visualization
    draw_img_preds(img, depth_pred, seg_pred, uncertainty_threshold, apply_depth_mask)
    # visualize the points in 3D
    draw_point_cloud(img, depth_pred, seg_pred, focal_len, uncertainty_threshold, apply_depth_mask)
    plt.show()


def draw_img_preds(img, depth_pred, seg_pred, uncertainty_threshold=0.0, apply_depth_mask=False):
    """Display the RGB image, and the corresponding depth and segmentation images next to each other."""
    plt.figure(0, figsize=(8, 6))

    # plot input img
    plt.subplot(2, 3, 1)
    plt.title('RGB')
    plt.imshow(img)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    # plot depth image
    plt.subplot(2, 3, 2)
    plt.title('depth estimation')
    depth_pred = depth_pred[0, :, :]
    plt.imshow(depth_pred)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    # plot segmentation
    plt.subplot(2, 3, 3)
    plt.title('segmentation')
    seg_labels = np.argmax(seg_pred, 0) + 1
    mask = np.zeros(shape=(seg_labels.shape[0], seg_labels.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_labels, np.asarray(key))
        mask[:, :, 0] += class_mask * CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask * CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask * CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    # plot masked depth image
    plt.subplot(2, 3, 5)
    plt.title('masked de')
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
    plt.title('masked seg')
    # mask out pixels where the certainty of the class prediction is lower than the uncertainty threshold
    uc = np.max(seg_pred, 0)
    uc_mask = uc > uncertainty_threshold
    seg_labels = np.argmax(seg_pred, 0) + 1
    seg_labels *= uc_mask
    mask = np.zeros(shape=(seg_labels.shape[0], seg_labels.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_labels, np.asarray(key))
        mask[:, :, 0] += class_mask * CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask * CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask * CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.3)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticks([])

    plt.draw()


def draw_point_cloud(img, depth_pred, seg_pred, f_len, uncertainty_threshold=0.0, apply_depth_mask=False):
    """Create a segmented 3D pointcloud from the RGB image, the corresponding depth estimation and the segmentation."""
    depth_pred = np.transpose(depth_pred, (1, 2, 0))
    depth_pred = depth_pred[:, :, 0]

    if apply_depth_mask:
        # mask high gradient regions ~ these are usually not as accurate
        grad = np.asarray(np.gradient(depth_pred))
        grad = np.abs(grad[0, :, :]) + np.abs(grad[1, :, :])
        grad_mask = grad < 0.95

        # focus on the immediate surroundings: mask everything that is farther than 50m
        depth_mask = (depth_pred < 50.0) * (depth_pred > 5.0)
        depth_pred = depth_pred * depth_mask * grad_mask

    # mask out pixels where the certainty of the class prediction is lower than the uncertainty threshold
    uc = np.max(seg_pred, 0)
    uc_mask = uc > uncertainty_threshold
    seg_pred = np.argmax(seg_pred, 0) + 1
    seg_pred *= uc_mask
    mask = np.zeros(shape=(seg_pred.shape[0], seg_pred.shape[1], 3))
    for key in CLASSES:
        class_mask = np.isin(seg_pred, np.asarray(key))
        mask[:, :, 0] += class_mask * CLASS_COLORS[key][0]
        mask[:, :, 1] += class_mask * CLASS_COLORS[key][1]
        mask[:, :, 2] += class_mask * CLASS_COLORS[key][2]
    mask = np.clip(mask, 0, 1)
    mask = (img / 255.0 * 0.7) + (mask * 0.3)

    # generate 3D points
    x = []
    y = []
    z = []
    colors = []
    idx = 0
    for i in range(depth_pred.shape[0]):
        for j in range(depth_pred.shape[1]):
            idx += 1
            # focus on the immediate surroundings: mask everything that is farther than 50m
            # also, mask out things that are too close, this might be noise
            if depth_pred[i, j] > 50.0 or depth_pred[i, j] < 2.0:
                continue
            # if the pixel is classified as sky or if it is invalid, skip
            if seg_pred[i, j] == 5 or seg_pred[i, j] == 0:
                continue
            # only show every 2nd pixel, this is more than enough for visualization
            if idx % 2 == 1:
                continue

            z.append(depth_pred[i, j])
            y.append(i * depth_pred[i, j] / f_len)
            x.append((-160) + j * depth_pred[i, j] / f_len)

            # color based on mask (0.7 * pixel color + 0.3 * label color)
            r, g, b = int(mask[i, j][0] * 255), int(mask[i, j][1] * 255), int(mask[i, j][2] * 255)
            colors.append(rgb2hex(r, g, b))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, marker=',', s=5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=-37., azim=-117.)

    plt.draw()
