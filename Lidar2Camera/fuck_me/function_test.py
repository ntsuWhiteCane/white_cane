import numpy as np
A_w = np.array([1, 0, 0])
B_w = np.array([0, 1, 0])
D_w = np.array([0, 0, 1])
C_w = np.array([1, 1, 1])
# # C_lidar = newton_raphson.newton_raphson(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# # print(C_w)
A_lidar = np.array([-1, 1, 0])
B_lidar = np.array([0, 1, 1])
D_lidar = np.array([-1, 0, 1])
C_lidar = np.array([0, 0, 0])

world_points = np.stack((A_w, B_w, C_w, D_w))
lidar_points = np.stack((A_lidar, B_lidar, C_lidar, D_lidar))

world_points_mass_center = (A_w + B_w + C_w + D_w)/4
lidar_points_mass_center = (A_lidar + B_lidar + C_lidar + D_lidar)/4

world_points_new = world_points - world_points_mass_center
lidar_points_new = lidar_points - lidar_points_mass_center

H = np.dot(lidar_points_new.T, world_points_new)
# print(H)
U, sigma, VT = np.linalg.svd(H)

R = np.dot(VT.T, U.T)
print()
# print(R)

T = world_points_mass_center.T - np.dot(R, lidar_points_mass_center.T)

print("world: ", world_points)
print("lidar: ", lidar_points)
print(f"R_1: \n{R}")
print(f"T_1: {T}")
print(np.linalg.det(R))
a = np.linalg.norm(world_points, axis=-1) 
print(np.linalg.norm(world_points, axis=-1))
print(np.mean(a))