import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class PointCloudObject:
    def __init__(self, object):
        self.xyz_points = object
        self.nearest_neigbours_indices = None

        # Object pose
        self.center_x = 0
        self.center_y = 0
        self.center_z = 0

        # Surface Features
        self.surface_normals_vector = []
        self.surface_normals_angles = []
        self.priciple_curvatures = []

    def compute_surface_normals(self, neighbours=6):

        min_z = 0 #np.min(self.xyz_points[:,2])
        max_z = np.max(self.xyz_points[:,2])
        min_x = np.min(self.xyz_points[:,0])
        min_y = np.min(self.xyz_points[:,1])
        max_x = np.max(self.xyz_points[:,0])
        max_y = np.max(self.xyz_points[:,1])
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        z_diff = max_z - min_z

        self.center_x = min_x + x_diff/2
        self.center_y = min_y + y_diff/2
        self.center_z = min_z + z_diff/2

        # First calculate the undirected normals by fitting a plane trough the Eigenvectors of the Covariance Matrix
        undirected_normals = []
        nn = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree').fit(self.xyz_points)
        _, self.nearest_neigbours_indices = nn.kneighbors(self.xyz_points)
        for index_group in self.nearest_neigbours_indices:
            COV = np.cov(np.array(self.xyz_points)[index_group].T)
            w, v = np.linalg.eig(COV)

            min_ev_index = np.argmin(w)
            undirected_normals.append([*list(self.xyz_points[index_group[0]]),*list(v[min_ev_index]/10000)])

        # Calculate the directed normals using an inside outside distinction
        for un in undirected_normals:
            pointcloudpoint_to_center = np.linalg.norm(np.array(un[:3]) - np.array(self.center()))
            vector_tip_to_center = np.linalg.norm((np.array(un[:3]) + np.array(un[3:])) - np.array(self.center()))
            if vector_tip_to_center < pointcloudpoint_to_center:
                self.surface_normals_vector.append([*un[:3],*list((-10000) * np.array(un[3:]))])
            else:
                self.surface_normals_vector.append([*un[:3],*list(10000 * np.array(un[3:]))])

        # Calculate the angles (not neccessarily needed if you only need vectors)
        for vec in self.surface_normals_vector:
            zenith = np.arctan2(vec[2], vec[0])
            azimuth = None
            if vec[0] >= 0 and vec[1] >= 0:
                azimuth = np.arctan2(vec[1], vec[0])
            elif vec[0] < 0 and vec[1] >= 0:
                azimuth = np.arctan2(vec[1], vec[0])
            elif vec[0] <= 0 and vec[1] < 0:
                azimuth = np.arctan2(-vec[1], -vec[0]) + np.pi
            elif vec[0] >= 0 and vec[1] < 0:
                azimuth = np.arctan2(-vec[1], vec[0]) + (3*np.pi)/2
            self.surface_normals_angles.append(np.array([azimuth, zenith]))


    def compute_principle_curvatures(self):
        for i, p in enumerate(self.xyz_points):
            # Estimate normal curvatures k_n
            k_n = []
            theta_n = []
            M_n = []
            first_index = True
            for index_neigbour in self.nearest_neigbours_indices[i]:
                if first_index == True:
                    first_index = False
                    continue
                q_index = index_neigbour 
                q = self.xyz_points[q_index] - p
                M_i = self.surface_normals_vector[q_index][3:]
            
                n_xy = (q[0]*M_i[0] + q[1]*M_i[1]) / np.sqrt(q[0]**2 + q[1]**2)

                k_n.append(n_xy / (np.sqrt(n_xy**2 + M_i[2]**2)*np.sqrt(q[0]**2 + q[1]**2)))

                # Least Square Fitting for principle curvatures
                N = self.surface_normals_vector[i][3:]
                psi = np.arccos(N[2])
                phi = np.arctan2(N[1], N[0])
                X = np.array([-np.sin(phi), np.cos(phi), 0])
                Y = np.array([np.cos(psi)*np.cos(phi), np.cos(psi)*np.sin(phi), -np.sin(psi)])

                # Project pq onto Plane X-Y
                proj_pq_on_N = np.dot(p*q, N)*np.array(N)
                proj_pq_on_XY = p*q - proj_pq_on_N
                theta = np.arccos(np.dot(X, proj_pq_on_XY/np.linalg.norm(proj_pq_on_XY))) 
                theta_n.append(theta)
                M_n.append(np.array([np.cos(theta)**2, 2*np.sin(theta)*np.cos(theta), np.sin(theta)**2]))

            M = np.array(M_n, dtype=np.float)
            k_n = np.array(k_n, dtype=np.float)
            A, B, C = np.linalg.lstsq(M, k_n, rcond=-1)[0]
            weingarten_matrix = np.matrix([[A, B], [B, C]])
            
            principal_cuvatures = np.linalg.eigvalsh(weingarten_matrix)
            self.priciple_curvatures.append(principal_cuvatures)

    def center(self):
        return [self.center_x, self.center_y, self.center_z]

    def plot_surface_normals(self):
        soa = np.array(self.surface_normals_vector)
        soa.T[3:] = soa.T[3:]/100
        X, Y, Z, U, V, W = zip(*soa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        plt.show()

    def plot_principle_curvature_disributions(self):
        pc = np.array(self.priciple_curvatures).T[0]
        pc2 = np.array(self.priciple_curvatures).T[1]
        
        # Some Outlier Removal Code from StackOverflow to remove the bad curvature estimation on the edges 
        filtered = pc[~is_outlier(pc)]
        filtered2 = pc2[~is_outlier(pc2)]

        plt.hist(filtered, bins = 20)
        plt.title("Min Principle Curvature distribution")
        plt.show()

        plt.hist(filtered2, bins = 20)
        plt.title("Max Principle Curvature distribution")
        plt.show()



def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh