

3D Scene Reconstruction From Single Monocular Images Using Deep Learning
============================================
My aim is to build a sematincally labeled 3D point cloud from monocular driving images. I predict depth and semantic information separately with two CNNs. Then, the depth predictions are used to build the point cloud and the points are labeled/colored based on the segmentation.


### CNNs
The networks' architecture is similar to the one described in the following paper:
Károly Harsányi, Attila Kiss, András Majdik, Tamás Szirányi: A Hybrid CNN Approach for Single Image Depth Estimation: A Case Study. IWCIM - 6th International Workshop on Computational Intelligence for Multimedia Understanding, 2018.
The approach we described in our paper was [implemented using PyTorch](https://github.com/karoly-hars/DE_resnet_unet_hyb).

The model for depth estimation has the exact same structure as the one in the paper. It is trained on the [Cityscapes dataset](https://www.cityscapes-dataset.com/dataset-overview/). I used all the available images from the 'train' and 'trainextra' subsets for training. 

The model for segmentation also has a similar structure, but the last layer has 7 output channels, one for each of the classes in the segmentation task. It was trained on the [BDD100K](http://bair.berkeley.edu/blog/2018/05/30/bdd/) dataset. It categorizes pixels into the following classes: flat surface, building, street furniture, vegetation, sky, human, and vehicle.

Both networks were trained on images with a size of 320x320 pixels. Their output size is 160x160, but after inference, it is upsampled to 320x320 pixels to match the size input.

Since the datasets contain driving images, using these models to create predictions in other environments will most likely lead to subpar results.

### Building the 3D Point Cloud
The depth estimation has a higher error rate around the edges of the objects, which can cause discrepancies in the point cloud. To prevent this, depth pixels with high gradients are masked out. Furthermore, I used the segmented image to detect the edges of the objects by examining the uncertainties in the semantic predictions. If the model shows a high uncertainty for a pixel (i.e. the max value for the pixel after the final softmax layer is lower than a threshold), the corresponding voxel is left out from the point cloud. Also, I left out the voxels from the point cloud which were predicted to be farther away from the sensor than 50 meters plus every voxel that was classified as 'sky'.

Based on the focal length of our camera and the depth estimation we can transform image coordinates into 3D coordinates. The points of the point cloud are then colored based on their predicted semantic label.

### Requirements
The code was tested with:
- python 3.5 and 3.6
- pytorch 1.3.1
- opencv-python 3.4.3
- matplotlib 2.2.3
- numpy 1.15.4
- colormap 1.0.2
- easydev 0.9.37

I used an NVIDIA card with ~8 GB memory for testing. Since the two networks will not necessarily fit into the memory of a single GPU it might be more practical to modify the code and run the depth and semantic predictions sequentially instead of parallelly.

### How to
```
python3 predict_img.py -i <path_to_image> -f <focal_lenght>
```
### Examples
The example images were sampled from the validation set of the Cityscapes dataset. The images were cropped to hide the hood of the 'ego vehicle'.

Munster98          |  Munster116
:-------------------------:|:-------------------------:
![Screenshot](docs/munster_98.png) | ![Screenshot](docs/munster_116.png)

Lindau38          |  Munster43
:-------------------------:|:-------------------------:
![Screenshot](docs/lindau_38.png) | ![Screenshot](docs/munster_43.png)

 
### Remarks
- the point clouds capture a very narrow region because a 320x320 image only represents a narrow slice of the environment. 
- it seems that the depth estimation is often inaccurate around street furniture (poles, traffic signs, etc).

