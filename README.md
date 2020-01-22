# semantic_hazard_cloud
Semantically identifiy hazard and create a 3D color coded cloud

The following package is needed:
- image-segmentation-keras 

In image-segmentation-keras package, change the following line

https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/predict.py#L80 

from 
```
$ return pr 

```
to 

```
$ return seg_img

```
