from point_cloud_object import PointCloudObject
import numpy as np

point_cloud_data = np.load("sample_point_clouds/coke_can_2_high_res.npy")

# Create the Object by assigning it an (N, 3) numpy array
object = PointCloudObject(point_cloud_data)


object.compute_surface_normals()
object.plot_surface_normals()

object.compute_principle_curvatures()
object.plot_principle_curvature_disributions()