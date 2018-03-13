# MobileNet-V2-Tensorlayer
## Tensorlayer for MobileNet-V2
 `HighLayer.py:` based on tensorlayer.ike tl.layers. Conv2dLayer + tl.layers.BatchNormLayer = hl.conv2d,etc.<br>
 `DiyLayer.py:`in bottleneck moudle,we will need add op.<br>
 `MobileNetV2.py:`the structure of MobileNet-V2,and I trained it on CK+ dataset.<br>
 
 ## training on CK+:<br>
 ### test accuracy:<br>
 ![train_accuracy](https://github.com/DasudaRunner/MobileNet-V2-tensorlayer/blob/master/png/accuracy.png)<br>
 ### test loss:<br>
 ![test loss](https://github.com/DasudaRunner/MobileNet-V2-tensorlayer/blob/master/png/testloss.png)
