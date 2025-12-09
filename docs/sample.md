<!-- ## Sampling -->

Classes for point cloud sampling

## Sampling
::: augmentation_class.Sampling
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: True
        members: [__call__]
**FPS PC with 4096 points**
<p align="center">
  <img src="../img/Sampling1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/Sampling2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## SamplingDynamic
::: augmentation_class.SamplingDynamic
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: True
        members: [__call__]
**FPS PC with surface area adjusted points**
<p align="center">
  <img src="../img/SamplingDynamic1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/SamplingDynamic2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

## SphereCrop
::: augmentation_class.SphereCrop
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: True
        members: [__call__]
**Sphere Crop PC**
<p align="center">
  <img src="../img/SphereCrop1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/SphereCrop2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>


## GridSample
::: augmentation_class.GridSample
    handler: python
    options:
        show_root_heading: False
        show_root_toc_entry: False
        show_source: True
        members: [__call__]
**Grid Sample on PC**

Original PC vs Grid Sample (random select, fixed grid size)
<p align="center">
  <img src="../img/GridSample1.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/GridSample2.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>

Grid Sample (random select, fixed grid numbers) vs Grid Sample (mean, fixed grid size)
<p align="center">
  <img src="../img/GridSample3.png" alt="Before" style="width:48%; max-width:48%; height:auto; margin-right:1%;">
  <img src="../img/GridSample4.png" alt="After" style="width:48%; max-width:48%; height:auto; margin-left:1%;">
</p>