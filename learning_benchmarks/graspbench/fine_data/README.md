python>=3.8
pip install open3d


The data is stored in `npz` file

- pts_4096: 100 files
- pts_2048: 100 files
- pts_1024: 100 files
- pts_512: 100 files



datapoint format:
- pos: n x 3, e.g., 4096 x 3, the xyz position of the point cloud
- normals: n x3, e.g., 4096 x 3, the surface normal for each point
- idx: 1, the index related to the grasp pose.
- quat: 1x4, the querternion associated with the idx
- rotation: 3x3, the rotation matrix
- depth: 1, the grasping depth value

Task:
1. the model can take the `pos: nx3` and output `normals: nx3` which is equivariant, rho_1 vector.
2. the model can take the `pos: nx3` and output `distribution: nx1` where the idx is the argmax and it is invariant; as well as outputting `rotation: 3x3` for the index and it is equivariant.

utils function:
there is a `data_description.py` used to checking and visualization the data.
