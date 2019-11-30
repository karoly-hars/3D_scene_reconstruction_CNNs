import torch
from utils import load_img, show_img_preds, show_point_cloud
import argparse
import net
import torch.nn.functional as F


def predict_img(img_path, focal_len):    
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print("\nusing GPU:", use_gpu)    

    # loading models
    print("\nLoading models...")
    model_de = net.get_model(output="depth", use_gpu=use_gpu)
    if use_gpu:
        model_de = model_de.cuda()

    model_seg = net.get_model(output="seg", use_gpu=use_gpu)
    if use_gpu:
        model_seg = model_seg.cuda()
            
    # setting models to evaluation mode
    model_de.eval()
    model_seg.eval()
    print("Done.")
      
    # reading image
    img = torch.Tensor(load_img(img_path))
    
    # running model on the image
    if use_gpu:
        img = img.cuda()
        
    print("Plotting...")
    output_de = model_de(img)
    output_seg = model_seg(img)
    
    # bilinear upsampling
    output_de = F.interpolate(output_de, size=(320, 320), mode="bilinear", align_corners=True)
    output_seg = F.interpolate(output_seg, size=(320, 320), mode="bilinear", align_corners=True)

    # softmax for semantic segmentation
    output_seg = F.softmax(output_seg, dim=1)

    # plotting the results
    output_de = output_de.cpu()[0].data.numpy()
    output_seg = output_seg.cpu()[0].data.numpy()
    img = img.cpu()[0].data.numpy()  
    show_img_preds(img, output_de, output_seg, uc_th=0.9, apply_depth_mask=True)

    # visualize the points in 3D
    show_point_cloud(img, output_de, output_seg, focal_len, uc_th=0.9, apply_depth_mask=True)
    print("Done")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str,  help="path to the RGB image input")
    parser.add_argument("focal_len", type=float,  help="focal length of the camera")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    predict_img(args.img_path, args.focal_len)
