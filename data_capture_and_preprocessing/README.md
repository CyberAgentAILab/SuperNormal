
This is a step-by-step guide to preprocess the raw images captured by an iPhone for the MVPS task.
You can download our raw images using the following command *(~6 GB per object)*.

```
gdown 'https://drive.google.com/file/d/1BcCuZR0C-snmCNf8iGhkFgkQ6arfcQ-L/view?usp=sharing' --fuzzy
unzip flower_girl.zip
rm flower_girl.zip
```

## File structure
You should have the following file structure under each object's folder:
```
 - RAW
 - mask
 - cameras.xml
```

The `RAW` folder contains all the DNG images captured by an iPhone. 
The `mask` folder contains the foreground masks for each view.
The `cameras.xml` contains the calibrated camera parameters using [Metashape](https://oakcorp.net/agisoft/download/). 

## Step-by-step data pre-processing
First we convert the DNG images to PNG file format.
```
# pip install rawpy
python iPhone_mvps_data_preprocessing.py --data_dir <path/to/obj_folder>
```
Now the file structure looks like this
```
    - RAW
    - mvps_png_full
    - sfm_png_full
    - mask
    - cameras.xml
```
The `mvps_png_full` folder contains the pre-processed images for photometric stereo, and the `sfm_png_full` folder contains the images for camera calibration using Structure from Motion.
In each view, we first take an image in ambient light and then additionally illuminate the object with an active light source.
So the first image in each view is collected in `sfm_png_full`.

### Mask preparation
Now we prepare the foreground masks for each view.
We used SAM to interactively segment the foreground objects.
Please install SAM according to the [official instructions](https://github.com/facebookresearch/segment-anything).
After installation, run the following command to segment the foreground objects for all views:

```
python sam_mvps.py --data_dir <path/to/obj_folder/mvps_png_full> --checkpoint <path/to/sam_vit_h_4b8939.pth>
```
This will pop up a window where you can interactively segment the foreground objects.
Select points on the object to segment the foreground object, and press `Esc` to check the intermediate results.
Continue to select points until you are satisfied with the segmentation results, and press `Enter` to save the mask.
The process will be repeated for all views.

The same mask will be saved in two places: `obj_folder/mask` and the corresponding folder containing the image from the same viewpoint. 
The latter will be used for normal map estimation.

### Camera calibration
In [MetaShape](https://oakcorp.net/agisoft/download/), import the images in the `sfm_png_full` folder and run the camera calibration process.
```
[Workflow] -> [Add Folder] -> select `sfm_png_full` -> select single cameras -> [Workflow] -> [Align Photos]
```

After camera calibration, export the camera parameters to `cameras.xml`.
```
[File] -> [Export] -> [Export Cameras]
```

The resulting `cameras.xml` file is what we have put in the object folder.


### Normal map estimation
Install [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023) and run the following command to generate the normal maps for each view:
```
python <path/to/sdm_unips/main.py> --session_name YOUR_SESSION_NAME --test_dir <path/to/obj_folder/mvps_png_full> --checkpoint <path/to/sdm_unips_checkpoint_dir> --scalable --target normal
```
Tips: Prepare the mask for each view to improve the normal estimation results. This should be done when you have completed the previous mask segmentation step.

The original SDM-UniPS code outputs normal maps in the PNG format. You can instead get EXR format by replacing [this line](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/blob/96e68f353173c2ae85bfe609e4728a19a2f8c92e/sdm_unips/modules/builder/builder.py#L162) with the following one:
```
pyexr.write(f'{testdata.data.data_workspace}/normal.exr', nout)
```
Remember to install the [pyexr](https://github.com/tvogels/pyexr) package and import it in the file.
After normal estimation, we collect the normal maps in the same folder.
Since SDM-UniPS estimates normal maps in camera space, we also convert them to the world space using the camera parameters from the previous step.

```
python gather_and_convert_normal_map.py --data_dir <path/to/obj_folder> --sdm_unips_result_dir <path/to/YOUR_SESSION_NAME/results>
```
The file structure is now as follows:
```
    - RAW
    - mvps_png_full
    - sfm_png_full
    - mask
    - normal_camera_space_sdmunips
    - normal_world_space_sdmunips
    - cameras.xml
    - results # if your SDM-UniPS output is in this folder
```

### Convert camera parameters to NeuS format
The last step is to convert the camera parameters to the NeuS format.
```
python metashape2neus.py --xml_path <path/to/obj_folder/cameras.xml>
```
This will create a `cameras_sphere.npz` file in the same folder as `cameras.xml`.
We also provide the converter to NeuS2 format. Check `metashape_to_neus2_json_and_images.py` for more details.

## Tips for capturing your own data
We used the iPhone's built-in camera app to take the images. Here are some tips for successful reconstruction:
- Use a tripod to stabilize the camera.
- Use a remote shutter release to avoid camera shake.
- Keep the same focus point in each view. On iPhone, you can press and hold the screen to lock the focus point.
- Use a white/black background to simplify the segmentation process.
- Use a turntable to capture the object from different angles. 
- Place the object on a textured surface to help the Structure from Motion process.
- Place the object in the center of the image.
- We used a [video light](https://www.ulanzi.com/collections/lighting/products/mini-led-video-light-ulanzi-vl49-1672) to illuminate the object from different angles in each view. Other light sources like a ring light/flashlight may also work.
- In each view, vary the light source's position sufficiently around the camera. We used 12 different light positions in our setup. 
- Reduce the exposure if the captured images are overexposed.

The above capture process can be done with off-the-shelf equipment, but it is tedious. 
It would be more convenient if you could build a custom rig to automate the capture process, such as [this example](https://youtu.be/zyEw-1QUlkU?si=8RvYC23emoP8TXrU).