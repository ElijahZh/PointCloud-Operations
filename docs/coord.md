<!-- ## Coordinate -->

Classes for point coordinates

## NormalizeCoord
::: augmentation_class.NormalizeCoord
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]   # <--- don't render __init__, etc.
**Normalize PC into Unit Sphere Space**
<p align="center">
  <img src="../img/NormalizeCoord1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/NormalizeCoord2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## PositiveShift
::: augmentation_class.PositiveShift
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Positive Shift PC**
<p align="center">
  <img src="../img/PositiveShift1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/PositiveShift2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## CenterShift
::: augmentation_class.CenterShift
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Center Shift PC**
<p align="center">
  <img src="../img/CenterShift1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/CenterShift2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## RandomShift
::: augmentation_class.RandomShift
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Shift PC**
<p align="center">
  <img src="../img/RandomShift1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomShift2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

### RandomRotate
::: augmentation_class.RandomRotate
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Rotate PC**
<p align="center">
  <img src="../img/RandomRotate1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomRotate2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## RandomScale
::: augmentation_class.RandomScale
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Scale PC**
<p align="center">
  <img src="../img/RandomScale1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomScale2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## RandomTranslate
::: augmentation_class.RandomTranslate
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Translate PC**
<p align="center">
  <img src="../img/RandomTranslate1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomTranslate2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## RandomJitter
::: augmentation_class.RandomJitter
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Jitter PC**
<p align="center">
  <img src="../img/RandomJitter1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomJitter2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## RandomFlip
::: augmentation_class.RandomFlip
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Flip PC**
<p align="center">
  <img src="../img/RandomFlip1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomFlip2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## RandomDropout
::: augmentation_class.RandomDropout
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Random Dropout PC**
<p align="center">
  <img src="../img/RandomDropout1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/RandomDropout2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## ShufflePoint
::: augmentation_class.ShufflePoint
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]


## PointClip
::: augmentation_class.PointClip
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Clip PC**
<p align="center">
  <img src="../img/PointClip1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/PointClip2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## ClipGaussianJitter
::: augmentation_class.ClipGaussianJitter
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Clip Gaussian Jitter on PC**
<p align="center">
  <img src="../img/ClipGaussianJitter1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/ClipGaussianJitter2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## ElasticDistortion
::: augmentation_class.ElasticDistortion
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: False
        members: [__call__]
**Elastic Distortion on PC**
<p align="center">
  <img src="../img/ElasticDistortion1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/ElasticDistortion2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>