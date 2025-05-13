# How To Run ControlNet On SynRS3D
1. clone the repository 
2. Download dependencies required to run seg_image_datasets.py by running the command below. Make sure to swap the pytorch version based on your cuda version. 
    - pip install --upgrade \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
  controlnet-aux opencv-python pillow numpy diffusers tqdm
3. download the [synrs3d dataset](https://zenodo.org/records/13905264)
4. We include a reference image but you can change that by altering the path of real_path in seg_image_datasets.py. 
5. Finally make sure to change the paths of where you stored the synthetic datasets and where you would like to store the outputs in seg_image_datasets.py
6. You should be all set to run seg_image_datasets.py on the SynRS3D dataset
