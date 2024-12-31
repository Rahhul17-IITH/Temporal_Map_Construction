Heads up!

- Download the Nuscenes v1.0 mini dataset and place it in the dataset directory.
- Then download the Maps Expansion Pack v1.3 US and place it in /data/nuScenes/maps/
- Rename the respective files by dropping from the -Resnet50 or -Resnet18 suffixes before running the respective scripts in the notebook for a particular architecture, i.e, either Resnet50 or Resnet 18.
- The files without the -Resnet50 or -Resnet18 suffixes have the implementations for ONNX.
- The outputs gifs are in the op_gifs directory.
- The scripts and the respective outputs for the experiments are in the " scripts_to_run.ipynb " notebook.
- The 30 output .pt model files and the hdmapnet_final.onnx files would be created in the "runs" directory while running the training script.
 
