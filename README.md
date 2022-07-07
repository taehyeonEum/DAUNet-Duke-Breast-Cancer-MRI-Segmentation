Model : **DAUNet**(Dense ASPP Unet)

Use[ Duke-Breast-Cancer-MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) Dataset for Semantic Segmentation for customed mask

Best Score : 0.623(Dice Score)


if you just want to train model :

    python main.py --device='cuda:0' --log_param=True --visualization=none

if you want to visualize result to 2d images :

    python main.py --device='cuda:0' --log_param=True --visualization=2d

if you want to visualize result to 3d images :

    python main.py --device='cuda:0' --log_param=True --visualization=3d

if you want to visualize both :

    python main.py --device='cuda:0' --log_param=True --visualization=both

--device: select device // --log_param: choose to save results or not // --visualization: visualization methods


If you want to **look for default setting**, you can fine in **/required_classes/config.py**

You can change model's Configuration in it


Additionaly, you can find *experiment log* in

    output/duke_segmantation_log.xlsx

    output/total_result/__model_name__.txt
