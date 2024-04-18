import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
from pose_interp import interpolate_poses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

path_to_scan_states = "/home/charles/Documents/zhongnan/fastlio-color/test-offline-color/test03/visual_odom_in_lidar_ts.txt"
path_to_visual_odom_timestamp = "/home/charles/Documents/zhongnan/fastlio-color/test-offline-color/test03/visual_odom.txt"
path_to_output = "/home/charles/Documents/zhongnan/fastlio-color/test-offline-color/test03/vo_interpolated_odom.txt"

# T_lidar_to_cam = np.array()
Tlc = [
        -0.9999487,
        -0.0060765,
        -0.0081089,
        0.07317228523672217,
        0.0080976,
        0.0018884,
        -0.9999654,
        -0.048289129209371784,
         0.0060916,
        -0.9999797,
        -0.0018391 ,
        -0.06748855890332943,
        0.0,
        0.0,
        0.0,
        1.0
    ]

Tci = [
        0.999916812873781,
        0.0005159530022363122,
        0.012888022534107549,
        0.14019804643418302,
        0.012897647385573308,
        -0.02959566554827303,
        -0.9994787377791847,
        0.03680700838981143,
	-0.00013425445093047644,
	0.9995618191854141, 
	-0.029599858149797875, 
	-0.01400772611768411,
        0.0,
        0.0,
        0.0,
        1.0
    ]

# origin tci
# Tci = [
#         0.999916812873781,
#         0.0005159530022363122,
#         0.012888022534107549,
#         0.14019804643418302,
#         0.012897647385573308,
#         -0.02959566554827303,
#         -0.9994787377791847,
#         0.03680700838981143,
# 	-0.00013425445093047644,
# 	0.9995618191854141, 
# 	-0.029599858149797875, 
# 	-0.09400772611768411,
#         0.0,
#         0.0,
#         0.0,
#         1.0
#     ]



def read_scan_states(file_path):
    """读取scan_states文件并提取时间戳、位置和旋转数据."""
    scan_states = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 8:
                    logging.error("Invalid line with insufficient data: %s", line)
                    continue
                timestamp = float(parts[0])
                position = np.array(parts[1:4], dtype=float)
                rotation = np.array(parts[4:8], dtype=float)  # quaternion [w, x, y, z]
                rotation = np.array([rotation[1],rotation[2],rotation[3],rotation[0]]) #[x, y, z, w]
                scan_states.append((timestamp, position, rotation))
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    return scan_states

# def interpolate_scan_states(scan_states, timestamps):
#     """在SE3李群上插值scan_states以匹配visual_odom的时间戳."""
#     interpolated_states = []
#     se3_interpolator = SE3Interpolator()  # 假设初始化一个SE3插值器
#     for ts in timestamps:
#         try:
#             interpolated_state = se3_interpolator.interpolate(scan_states, ts)
#             interpolated_states.append(interpolated_state)
#         except ValueError as e:
#             logging.error("Interpolation error at timestamp %s: %s", ts, e)
#             continue
#     return interpolated_states


def interpolate_scan_states(scan_states, timestamps):
    """使用SE3插值scan_states以匹配visual_odom的时间戳."""
    pose_timestamps = [int(state[0] * 1e6) for state in scan_states] 

    # print("pose_timestamps is: ",pose_timestamps)
    abs_poses = [np.concatenate((state[1], state[2])) for state in scan_states]  # 合并位置和旋转为一个数组

    origin_timestamp = int(pose_timestamps[0])
    
    # np.savetxt(kf_pose_fname.replace(".txt", "_interp.txt"), np.array(interpolated_pose), fmt="%1.6f")

    interpolated_poses = interpolate_poses(pose_timestamps, abs_poses, timestamps, origin_timestamp)
    
    # 将插值结果拆分回位置和旋转格式
    interpolated_states = []
    for pose in interpolated_poses:
        timestamp, position, quaternion = pose[0], pose[1:4], pose[4:8]
        interpolated_states.append((timestamp, position, quaternion))
    return interpolated_states

def save_camera_odometry(camera_states, timestamps, file_path):
    """将相机odometry数据保存到新的txt文件中."""
    try:
        with open(file_path, 'w') as file:
            # print(camera_states)
            # for ts, (position, rotation) in zip(timestamps, camera_states):
            #     file.write(f"{ts} {position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}\n")

            for ts, position, rotation in camera_states:
                # file.write(f"{ts:.6f} {position[0]} {position[1]} {position[2]} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}\n") 
                file.write(f"{ts:.6f} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f} {rotation[0]:.8f} {rotation[1]:.8f} {rotation[2]:.8f} {rotation[3]:.8f}\n") 

    except IOError as e:
        logging.error("Failed to write to file %s: %s", file_path, e)

def read_visual_odom_timestamps(file_path):
    """读取视觉里程计时间戳文件."""
    # timestamps = []
    # try:
    #     with open(file_path, 'r') as file:
    #         timestamps = [float(line.strip()) for line in file]
    # except FileNotFoundError:
    #     logging.error("File not found: %s", file_path)
    #     raise

    timestamps = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 8:
                    logging.error("Invalid line with insufficient data: %s", line)
                    continue
                timestamp = float(parts[0])
                # print("timestamp is ", timestamp)

                # position = np.array(parts[1:4], dtype=float)
                # rotation = np.array(parts[4:8], dtype=float)  # quaternion [w, x, y, z]
                # rotation = np.array([rotation[1],rotation[2],rotation[3],rotation[0]]) #[x, y, z, w]
                timestamps.append((timestamp))
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    # timestamps = [int(ts[0] * 1e6) for ts in timestamps] 
    timestamps_np = np.array(timestamps)
    timestamps_np = np.squeeze(timestamps_np[:]) * 1e6
    timestamps_list = timestamps_np.astype(np.int64).tolist()
    # print(timestamps_list)
    return timestamps_list

    # timestamps = np.genfromtxt(file_path)
    
    # timestamps = np.squeeze(timestamps[:]) * 1e6
    # timestamps = timestamps.astype(np.int64).tolist()
    # # print(f"{timestamps:.6f}")
    # return timestamps

def transform_to_camera_frame(interpolated_states, T_cam_imu):
    """使用imu到相机的外参转换imu odometry到相机odometry."""
    camera_states = []
    for timestamp, position, quaternion in interpolated_states:


        # SO3 odom_R_lidar = state_point.rot * state_point.offset_R_L_I;
        # V3D odom_t_lidar = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        
        # 对于旋转，我们需要将四元数转换为旋转矩阵，然后应用变换，最后转换回四元数
        T_imu_cam = np.linalg.inv(T_cam_imu)
        rot_matrix = R.from_quat(quaternion).as_matrix()  # 假设quaternion是[x, y, z, w]格式
        cam_rot_matrix = np.dot(rot_matrix, T_imu_cam[:3, :3], )  # 应用旋转部分的变换
        cam_quaternion = R.from_matrix(cam_rot_matrix).as_quat()  # 转换回四元数
        

       # 创建SE3变换矩阵（这里假设T_lidar_to_cam是一个4x4的numpy数组）
        pos_homogeneous = np.hstack((position, [1]))  # 将位置转换为齐次坐标
        # cam_pos_homogeneous = np.dot(T_cam_imu, pos_homogeneous)  # 应用变换
        cam_pos_homogeneous = position + np.dot(cam_rot_matrix, T_imu_cam[:3, 3])

        # 四元数可能需要根据库的要求调整顺序
        camera_states.append((timestamp, cam_pos_homogeneous[:3], cam_quaternion))
    return camera_states

if __name__=="__main__":

    scan_states = read_scan_states(path_to_scan_states)
    visual_timestamps = read_visual_odom_timestamps(path_to_visual_odom_timestamp)
    interpolated_states = interpolate_scan_states(scan_states, visual_timestamps)
    # 假设T_lidar_to_cam已经定义
    # T_cam_to_lid = np.array(Tlc).reshape(4,4)
    # T_lidar_to_cam = np.linalg.inv(T_cam_to_lid)

    # T_cam_imu = np.array(Tci).reshape(4,4)
    # camera_states = transform_to_camera_frame(interpolated_states, T_cam_imu)
    # camera_states = interpolated_states
    save_camera_odometry(interpolated_states, visual_timestamps, path_to_output)
