import numpy as np
import teaserpp_python
from Config import Config
import gtsam as gt
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2,
                   NonlinearFactorGraph, NonlinearISAM, Pose3,
                   PriorFactorPoint3, PriorFactorPose3, Rot3,
                   PinholeCameraCal3_S2, Values, Point3) # symbol_shorthand_X, symbol_shorthand_L)
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt

# import g2o
# class PoseOptimizer(g2o.SparseOptimizer):
#     def __init__(self, ):
#         super().__init__()
#         solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
#         solver = g2o.OptimizationAlgorithmLevenberg(solver)
#         super().set_algorithm(solver)
#         self.edge_list = []
#         self.edge_outlier = np.array([], dtype=bool)
#         self.v_se3 = g2o.VertexSE3Expmap()
#         self.v_se3.set_id(0)   # internal id
#         self.v_se3.set_fixed(False)
#         super().add_vertex(self.v_se3)
#         self.pose = []
#         self.inv_lvl_sigma2 = np.zeros((8,), dtype=np.float)
#         for idx in np.arange(8):
#             self.inv_lvl_sigma2[idx] = 1./1.2**(2*idx-2)
#
#     def optimize(self, max_iterations=10):
#         self.edge_outlier = np.full(len(self.edge_list), False)
#         for iteration in range(4):
#             # self.v_se3.set_estimate(self.pose)
#             super().initialize_optimization(0)
#             super().optimize(max_iterations)
#             print("ITER", self.vertex(0).estimate().to_vector())
#             print("Initial Correspondence: ", np.count_nonzero(1-self.edge_outlier))
#             n_bad = 0
#             for idx in range(len(self.edge_list)):
#                 e = self.edge_list[idx]
#                 e.compute_error()
#                 chi2 = e.chi2()
#                 # print("Iter ", iteration, "Chi: " ,chi2)
#                 if chi2 > 7.815:
#                     self.edge_outlier[idx] = True
#                     e.set_level(1)
#                     n_bad += 1
#                 else:
#                     self.edge_outlier[idx] = False
#                     e.set_level(0)
#                 if iteration == 2:
#                     e.set_robust_kernel(None)
#
#             print("NUM BADS: ", n_bad, ":", len(self.edge_list))
#         return self.edge_outlier
#
#     def add_pose(self, pose, fixed=False):
#         self.v_se3.set_estimate(pose)
#         self.pose = pose
#
#     def add_point(self, world_pos,
#                   measurement_cam,
#                   octave,
#                   robust_kernel=g2o.RobustKernelHuber(np.sqrt(7.815))):   # ??% CI
#
#         edge = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
#         edge.set_vertex(0, self.vertex(0))
#
#         fx = Config().fx
#         fy = Config().fy
#         cx = Config().cx
#         cy = Config().cy
#         bf = Config().bf
#
#         edge.fx = fx
#         edge.fy = fy
#         edge.cx = cx
#         edge.cy = cy
#         edge.bf = bf
#         edge.Xw = world_pos
#
#         edge.set_measurement(measurement_cam)   # projection
#         information = self.inv_lvl_sigma2[octave]*np.identity(3)
#         edge.set_information(information)
#
#         if robust_kernel is not None:
#             edge.set_robust_kernel(robust_kernel)
#
#         super().add_edge(edge)
#
#         self.edge_list.append(edge)
#
#     def get_pose(self):
#         return self.vertex(0).estimate()


class PoseOptimizerTeaser:
    def __init__(self):
        self.NOISE_BOUND = 0.1  # 0.05
        self.solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        self.solver_params.cbar2 = 0.6  # 1
        self.solver_params.noise_bound = self.NOISE_BOUND
        self.solver_params.estimate_scaling = False
        self.solver_params.rotation_estimation_algorithm = \
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        self.solver_params.rotation_gnc_factor = 1.4
        self.solver_params.rotation_max_iterations = 200
        self.solver_params.rotation_cost_threshold = 1e-12
        self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)

    def optimize(self, src, dst):
        # start = time.time()
        self.solver.solve(src, dst)
        # end = time.time()

        solution = self.solver.getSolution()

        trans = np.hstack((solution.rotation, np.expand_dims(solution.translation, axis=1)))
        trans = np.concatenate((trans, np.expand_dims(np.array([0, 0, 0, 1]), axis=1).T), axis=0)

        return trans


class PoseOptimizerGTSAM:
    def __init__(self):
        fx = Config().fx
        fy = Config().fy
        cx = Config().cx
        cy = Config().cy
        bf = Config().bf
        # Create realistic calibration and measurement noise model
        # format: fx fy skew cx cy baseline
        baseline = bf/fx
        self.K_stereo = gt.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, baseline)
        self.K_mono = gt.Cal3_S2(fx, fy, 0.0, cx, cy)

        self.deltaMono = np.sqrt(5.991)
        self.deltaStereo = np.sqrt(7.815)

        self.depth_threshold = bf/fx * 60

        # Create graph container and add factors to it
        self.graph = gt.NonlinearFactorGraph()

        # Create initial estimate for camera poses and landmarks
        self.initialEstimate = gt.Values()

        # add a constraint on the starting pose
        # first_pose = gt.Pose3()
        # self.graph.add(gt.NonlinearEqualityPose3(X(1), first_pose))

        self.inv_lvl_sigma2 = np.zeros((8,), dtype=np.float)
        for idx in np.arange(8):
            self.inv_lvl_sigma2[idx] = 1. / 1.2 ** (2 * idx - 2)

        # point counter for landmarks and octave container
        self.counter = 1
        self.octave = []

        self.is_stereo = []

    def add_pose(self, R, t):
        # Add measurements
        # pose 1
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(520, 480, 440), stereo_model, x1, l1, K))
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(120, 80, 440), stereo_model, x1, l2, K))
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(320, 280, 140), stereo_model, x1, l3, K))

        # pose 2
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(570, 520, 490), stereo_model, x2, l1, K))
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(70, 20, 490), stereo_model, x2, l2, K))
        # graph.add(gt.GenericStereoFactor3D(gt.StereoPoint2(320, 270, 115), stereo_model, x2, l3, K))
        # self.initialEstimate.insert(X(1), gt.Rot3(pose[0]), gt.Point3(pose[1]))
        t = t.reshape((3, 1))
        self.initialEstimate.insert(X(1), gt.Pose3(np.concatenate((R, t), axis=1)))

    def add_point(self, pointsInitial, measurements, octave):

        if pointsInitial[-1] > self.depth_threshold:
            information = self.inv_lvl_sigma2[octave] * np.identity(2)
            stereo_model = gt.noiseModel_Diagonal.Information(information)
            huber = gt.noiseModel_mEstimator_Huber.Create(self.deltaMono)
            robust_model = gt.noiseModel_Robust(huber, stereo_model)
            factor = gt.GenericProjectionFactorCal3_S2(gt.Point2(measurements[0], measurements[2]), robust_model,
                                              X(1), L(self.counter), self.K_mono)
            self.is_stereo.append(False)
        else:
            information = self.inv_lvl_sigma2[octave] * np.identity(3)
            stereo_model = gt.noiseModel_Diagonal.Information(information)
            huber = gt.noiseModel_mEstimator_Huber.Create(self.deltaStereo)
            robust_model = gt.noiseModel_Robust(huber, stereo_model)
            factor = gt.GenericStereoFactor3D(gt.StereoPoint2(*tuple(measurements)), robust_model,
                                              X(1), L(self.counter), self.K_stereo)
            self.is_stereo.append(True)

        self.graph.add(gt.NonlinearEqualityPoint3(L(self.counter), gt.Point3(pointsInitial)))
        self.initialEstimate.insert(L(self.counter), gt.Point3(pointsInitial))
        self.graph.add(factor)
        self.octave.append(octave)
        self.counter += 1

    def optimize(self, flag_verbose=False):
        # optimize
        edge_outlier = np.full(self.counter-1, False)
        error_th_stereo = [7.815, 7.815, 5, 5]
        error_th_mono = [5.991, 5.991, 3.5, 3.5]
        # error_th_stereo = [7.815, 7.815, 7.815, 7.815]
        # error_th_mono = [5.991, 5.991, 5.991, 5.991]
        for iteration in range(4):
            if flag_verbose:
                errors = []
            optimizer = gt.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
            result = optimizer.optimize()
            n_bad = 0
            if flag_verbose:
                print(f"Number of Factors: {self.graph.nrFactors()-self.graph.size()//2, self.graph.size()//2}")
            error_s = error_th_stereo[iteration]
            error_m = error_th_mono[iteration]
            for idx in range(1, self.graph.size(), 2):
                try:
                    if self.is_stereo[idx]:
                        factor = gt.dynamic_cast_GenericStereoFactor3D_NonlinearFactor(self.graph.at(idx))
                    else:
                        factor = gt.dynamic_cast_GenericProjectionFactorCal3_S2_NonlinearFactor(self.graph.at(idx))
                except:
                    if flag_verbose:
                        errors.append(0)
                    continue

                error = factor.error(result)
                # print(error)

                if flag_verbose:
                    errors.append(error)

                # if error > 7.815:
                if (self.is_stereo[idx] and error > error_s) or (not self.is_stereo[idx] and error > error_m):
                    edge_outlier[idx//2] = True
                    self.graph.remove(idx)
                    n_bad += 1
                else:
                    edge_outlier[idx//2] = False
                if iteration == 2:
                    if self.is_stereo[idx]:
                        information = self.inv_lvl_sigma2[self.octave[idx//2]] * np.identity(3)
                        stereo_model = gt.noiseModel_Diagonal.Information(information)
                        new_factor = gt.GenericStereoFactor3D(factor.measured(), stereo_model, X(1),
                                                              L(idx//2+1), self.K_stereo)
                    else:
                        information = self.inv_lvl_sigma2[self.octave[idx // 2]] * np.identity(2)
                        stereo_model = gt.noiseModel_Diagonal.Information(information)
                        new_factor = gt.GenericProjectionFactorCal3_S2(factor.measured(), stereo_model,
                                                                       X(1),
                                                                       L(idx // 2 + 1), self.K_mono)
                    self.graph.replace(idx, new_factor)

            if flag_verbose:
                fig, ax = plt.subplots()
                ax.bar(np.arange(0, len(errors)).tolist(), errors)
                plt.show()

                print("NUM BADS: ", n_bad)

        pose = result.atPose3(X(1))

        # marginals = gt.Marginals(self.graph, result)
        # cov = marginals.marginalCovariance(gt.X(1))

        return pose, edge_outlier  # self.graph, result


class PoseGraphOptimizerGTSAM:
    def __init__(self):
        # Create graph container and add factors to it
        self.graph = gt.NonlinearFactorGraph()

        # Create initial estimate for camera poses and landmarks
        self.initialEstimate = gt.Values()

        sigmas = np.array([5*np.pi/180, 5*np.pi/180, 5*np.pi/180, 0.05, 0.05, 0.05])
        self.covariance = gt.noiseModel.Diagonal.Sigmas(sigmas)
        self.graph.add(gt.NonlinearEqualityPose3(X(0), gt.Pose3(np.eye(4))))

        self.result = None
        self.marginals = None

    def add_node(self, kf):
        self.initialEstimate.insert(X(kf.kfID), gt.Pose3(kf.pose_matrix()))
        for kf_n, rel_pose, _ in kf.neighbors:
            if kf_n.kfID > kf.kfID:
                continue
            self.graph.add(gt.BetweenFactorPose3(X(kf.kfID), X(kf_n.kfID),
                                                 gt.Pose3(rel_pose), self.covariance))

    def add_node_optimize(self, kf):
        self.add_node(kf)
        result, marginals = self.optimize()
        return result, marginals

    def optimize(self):
        optimizer = gt.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
        result = optimizer.optimize()
        marginals = gt.Marginals(self.graph, result)
        return result, marginals


class PoseOptimizerRANSAC:
    def __init__(self):
        self.n_iteration = 100

    @classmethod
    def procrustes(cls, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.

            d, Z, [tform] = procrustes(X, Y)

        Inputs:
        ------------
        X, Y
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.

        scaling
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        """
        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection is not 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2

            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform