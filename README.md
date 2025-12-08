# Point Cloud Operation Methods

This repository contains a collection of Python utilities for **point cloud operations**, including:

- Point sampling (mesh-based and point-based)
- Coordinate-related augmentations
- Normal-related operations
- Color-related augmentations
- Add noise into existing point cloud
- NumPy → PyTorch tensor conversion
- Simple visualization utilities using Open3D’s interactive 3D viewer

The goal is to provide a focused toolkit with **clean docstrings and full documentation**, so it’s easy to review and reuse the methods in your own projects.


This project is majorly based on the functions and design ideas from Pointcept, with:

- A subset of their point cloud operation utilities

- Some extra methods specific to this project

- Added visualization and documentation for easier review

If you’re interested in a more complete and large-scale point cloud framework, please check out the original [Pointcept](https://github.com/Pointcept/Pointcept) repository.


## Installation

Before using the code in this repository, please install the following Python packages:

```bash
pip install numpy torch scipy scikit-learn open3d trimesh pymeshlab matplotlib addict
```


## Documentation

A full set of browsable docs (generated with MkDocs + mkdocstrings) is available here:

👉 Full documentation: https://ElijahZh.github.io/PointCloud-Operations/

The docs include:

- API reference for each function/class

- Explanations of arguments and return values

- Visual examples of augmentations and sampling methods


## Visualization

Visualization is provided through Open3D GUI, using its interactive 3D viewer:

- You can load point clouds or meshes and interact with them (rotate, zoom, inspect).

- For comparison of different augmentations or sampling strategies, we recommend:
    - Running the operations locally

    - Viewing the results in the Open3D GUI viewer

    - Taking screenshots of the rendered point clouds to include in docs or reports

Some example result images are included in the documentation to illustrate how each transform affects the point cloud.


## License

This project is licensed under the [MIT License](./LICENSE).