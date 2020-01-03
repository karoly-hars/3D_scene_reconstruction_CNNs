import argparse
import cv2
import torch
import torch.nn.functional as F
from network import ResnetUnetHybrid
import image_utils


def predict_img(img_path, focal_len):
    """Given an image create a 3D model of the environment, based depth estimation and semantic segmentation."""
    # switch to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('using GPU:', use_gpu)

    # load models
    print('Loading models...')
    model_de = ResnetUnetHybrid.load_pretrained(output_type='depth', use_gpu=use_gpu)
    model_seg = ResnetUnetHybrid.load_pretrained(output_type='seg', use_gpu=use_gpu)
    model_de.eval()
    model_seg.eval()

    # load image
    img = cv2.imread(img_path)[..., ::-1]
    img = image_utils.scale_image(img)
    img = image_utils.center_crop(img)
    inp = image_utils.img_transform(img)
    inp = inp[None, :, :, :]
    if use_gpu:
        inp = inp.cuda()

    print('Plotting...')
    output_de = model_de(inp)
    output_seg = model_seg(inp)
    
    # up-sample outputs
    output_de = F.interpolate(output_de, size=(320, 320), mode='bilinear', align_corners=True)
    output_seg = F.interpolate(output_seg, size=(320, 320), mode='bilinear', align_corners=True)

    # use softmax on the segmentation output
    output_seg = F.softmax(output_seg, dim=1)

    # plot the results
    output_de = output_de.cpu()[0].data.numpy()
    output_seg = output_seg.cpu()[0].data.numpy()
    image_utils.create_plots(img, output_de, output_seg, focal_len, uncertainty_threshold=0.9, apply_depth_mask=True)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, required=True, help='Path to input image.')
    parser.add_argument('-f', '--focal_len', type=float, required=False, default=2264.0,
                        help='The focal length of the camera. '
                             'Default: 2264 (this value should work for the example images).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    predict_img(args.img_path, args.focal_len)
