import torch
from torch.autograd import Variable
from utils import load_img, show_img_preds, show_point_cloud
import argparse
import net_DE
import net_seg
import torch.nn.functional as F



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str,  help='path to the RGB image input')
    parser.add_argument('focal_len', type=float,  help='focal length of the camera')
    args = parser.parse_args()
    
    # switching to GPU if possible
    use_gpu = torch.cuda.is_available()
    print('\nusing GPU:', use_gpu)    
    
    
    # loading models
    print('\nLoading models...')
    model_DE = net_DE.hyb_net(use_gpu=use_gpu)
    if use_gpu:
        model_DE = model_DE.cuda()

    model_seg = net_seg.hyb_net(use_gpu=use_gpu)
    if use_gpu:
        model_seg = model_seg.cuda()
            
    # setting models to evalutation mode
    model_DE.eval()
    model_seg.eval()
    print('Done.')
      
    # reading image
    img = load_img(args.img_path)
    
    # running model on the image
    if use_gpu:
        img = Variable(img.cuda())
    else:
        img = Variable(img)
    print('Plotting...')
    output_de = model_DE(img)
    output_seg = model_seg(img)
    
    # bilinear upsampling   
    resizer = torch.nn.Upsample(size=(320,320), mode='bilinear', align_corners=True)
    output_seg = resizer(output_seg)
    output_seg = F.softmax(output_seg, dim=1)
    output_de = resizer(output_de)
    
        
    # ploting the results
    output_de = output_de.cpu()[0].data.numpy()
    output_seg= output_seg.cpu()[0].data.numpy()
    img = img.cpu()[0].data.numpy()  
    show_img_preds(img, output_de, output_seg, uc_th=0.9, apply_depth_mask=True)
    
    
    # visualize the point cloud
    show_point_cloud(img, output_de, output_seg, args.focal_len, uc_th=0.9, apply_depth_mask=True)
    print('Done')
    

if __name__ == "__main__":
    main()


