import torch
# from lietorch.groups import SO3, SE3
from BA.BA_utils import *
from BA.BA_filtering import BA
from BA.utils import *
import numpy as np
import matplotlib.pyplot as plt
import json
# import pandas
import ipdb

def od_pipe(data, orbit_lat_long):

    ### Specify hyperparameters
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-3
    num_iters = 10

    ### Read data      # Need Paulo's help here
    # read csv file with pandas with ',' as delimiter
    # data = np.genfromtxt('data/one_pass_output.csv', delimiter=',')[1:]# np.array(pandas.read_csv("data/one_pass_output.csv" , sep=','))
    # orbit_lat_long = np.genfromtxt('data/lonlat.csv', delimiter=',') #np.array(pandas.read_csv("data/lonlat.csv", sep=','))
    # altitudes = np.genfromtxt('data/altitudes.csv')#np.array(pandas.read_csv("data/altitudes.csv"))[:,0]
    # intrinsics = np.genfromtxt("data/intrinsics.csv", delimiter=',') #  might have to specify manually

    # data = np.genfromtxt('data1/pixels.csv', delimiter=',')# np.array(pandas.read_csv("data/one_pass_output.csv" , sep=','))
    # orbit_lat_long = np.genfromtxt('data1/lonlat.csv', delimiter=',') #np.array(pandas.read_csv("data/lonlat.csv", sep=','))
    altitudes = orbit_lat_long[:,-1]/1000 #np.genfromtxt('data1/altitudes.csv')#np.array(pandas.read_csv("data/altitudes.csv"))[:,0]
    orbit_lat_long = orbit_lat_long[:,:-1]
    intrinsics = np.genfromtxt("data1/intrinsics.csv", delimiter=',')[0] #  might have to specify manually
    ii = data[:,0]
    # intrinsics = np.array(intrinsics)

    ### Convert data to right coordinates - ECEF to ECI    # Need Paulo's help here
    gt_pos_eci = convert_latlong_to_cartesian(orbit_lat_long[:,1], orbit_lat_long[:,0], altitudes)
    gt_vel_eci = compute_velocity_from_pos(gt_pos_eci, dt)
    gt_quat_eci = convert_pos_to_quaternion(gt_pos_eci)
    poses_gt_eci = np.concatenate([gt_pos_eci, gt_quat_eci], axis=1)
    landmarks_xyz = convert_latlong_to_cartesian(data[:, 4], data[:,3])#data["y(latitude)"], data["x(longitude)"]) 
    landmarks_uv = np.stack([data[:,1], data[:,2]] , axis=1) #data["x"], data["y"]
    gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    gt_acceleration = torch.tensor(gt_acceleration)
    gt_pos_eci = torch.tensor(gt_pos_eci)
    gt_vel_eci = torch.tensor(gt_vel_eci)
    poses_gt_eci = torch.tensor(poses_gt_eci)
    gt_quat_eci = torch.tensor(gt_quat_eci)
    landmarks_xyz = torch.tensor(landmarks_xyz)
    landmarks_uv = torch.tensor(landmarks_uv)
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).repeat(len(gt_pos_eci), 1)

    # gt_acceleration = compute_acceleration_from_omega(gt_omega, dt)

    ### Obtain acceleration from orbital dynamics and angular velocity from IMU
    # dyn_params = get_all_r_sun_moon_PN()
    x = torch.cat([gt_pos_eci, gt_vel_eci], dim=1)
    # t, params = dyn_params[-1], dyn_params[:-1]
    # gt_acceleration = RK4_orbit_dynamics_avg(x, h) #RK4_avg(x, t, h, params)
    # gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    gt_omega = compute_omega_from_quat(gt_quat_eci, dt)
    imu_meas = torch.cat((gt_omega, gt_acceleration), dim=-1)   # for now, assume that the IMU gives us the accurate angular velocity and acceleration
    # ipdb.set_trace()
    landmark_uv_proj = landmark_project(poses_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), ii, jacobian=False)
    # print("landmark_uv_proj", landmark_uv_proj[0,:10])
    # print("landmark_uv_proj", landmarks_uv[:10])
    print("mean landmark difference : ", (landmark_uv_proj[0,:] - landmarks_uv).mean(dim=0))
    # ipdb.set_trace()
    
    
    ### Initial guess for poses, velocities
    T = len(gt_pos_eci)
    # ipdb.set_trace()
    # offset = torch.tensor([1, 1, 1, 0, 0, 0, 0])[None, None].repeat(1, T, 1)*100
    position_offset = torch.randn((T, 3))*0 
    orientation_offset = torch.ones([T, 3])*0.2
    orientation_offset[:, :2] = 0
    position = poses_gt_eci.float()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.float()[:, 3:]) + orientation_offset)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).float() + offset# torch.zeros(1, T, 7)
    velocities = gt_vel_eci.unsqueeze(0).float() # torch.zeros(1, T, 3)
    imu_meas = imu_meas.unsqueeze(0)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)

    for i in range(num_iters):
        poses, velocities = BA(poses, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, intrinsics, Sigma, V, poses_gt_eci)

# def od_pipe(data, orbit_lat_long):
def process_ground_truths(orbit, landmarks_dict, intrinsics, dt, time_idx):

    ### Convert data to right coordinates - ECEF to ECI    # Need Paulo's help here
    gt_pos_eci = orbit[time_idx,:3]#convert_latlong_to_cartesian(orbit_lat_long[:,1], orbit_lat_long[:,0], altitudes)
    # gt_pos_eci = eci_to_ecef(gt_pos_eci, time_idx)
    gt_vel_eci = compute_velocity_from_pos(orbit[:,:3], dt) # orbit[:,3:6]
    # gt_vel_eci = compute_velocity_from_pos(gt_pos_eci, dt)
    # gt_quat_eci_full = np.concatenate([orbit[:,7:10], orbit[:, 6:7]], axis=-1)#convert_pos_to_quaternion(gt_pos_eci)
    zc, yc, xc = convert_quaternion_to_xyz_orientation(orbit[:,6:10], np.arange(len(orbit)))
    gt_quat_eci_full = convert_xyz_orientation_to_quat(xc, yc, zc, np.arange(len(orbit)))
    gt_quat_eci = gt_quat_eci_full[time_idx, :]
    poses_gt_eci = np.concatenate([gt_pos_eci, gt_quat_eci], axis=1)
    # ipdb.set_trace()
    # poses_gt_eci = np.concatenate([orbit[:,:3], orbit[:,6:10]], axis=1)
    landmarks_xyz = convert_latlong_to_cartesian(landmarks_dict["lonlat"][:, 1], landmarks_dict["lonlat"][:,0], landmarks_dict["frame"])#data["y(latitude)"], data["x(longitude)"]) 
    landmarks_uv = landmarks_dict["uv"] #data["x"], data["y"] or landmarks_dict["uv"]
    gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    gt_acceleration = torch.tensor(gt_acceleration)
    gt_pos_eci = torch.tensor(gt_pos_eci)
    gt_vel_eci = torch.tensor(gt_vel_eci)
    poses_gt_eci = torch.tensor(poses_gt_eci)
    gt_quat_eci = torch.tensor(gt_quat_eci)
    gt_quat_eci_full = torch.tensor(gt_quat_eci_full)
    landmarks_xyz = torch.tensor(landmarks_xyz)
    landmarks_uv = torch.tensor(landmarks_uv)
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).repeat(len(gt_pos_eci), 1)
    return gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration


if __name__ == "__main__":

    ### Specify hyperparameters
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-3
    num_iters = 100
    torch.set_printoptions(precision=4, sci_mode=False)

    ### Read data      # Need Paulo's help here
    # read csv file with pandas with ',' as delimiter
    # data = np.genfromtxt('landmarks/one_pass_output.csv', delimiter=',')[1:]# np.array(pandas.read_csv("data/one_pass_output.csv" , sep=','))
    sample_dets = False
    if not sample_dets:
        landmarks = np.load("landmarks/all_dets.npy", allow_pickle=True)
    else:
        landmarks = np.load("landmarks/sample_dets.npy", allow_pickle=True)
    # landmarks = [(0,0), (34.0413025185241,120.29134064842114)], [(0,12), (34.25080363316998,120.57863828344972)], [(0,24), (34.44452826130661,120.84781470157446)], [(0,36), (34.62475766637252,121.10134837146262)], [(0,48), (34.79329020541273,121.34120370042804)], [(0,60), (34.951572085217755,121.56896879772643)], [(0,72), (35.100786090229796,121.78594937883759)], [(0,84), (35.24191356244681,121.9932343986008)], [(0,96), (35.3757787925144,122.19174324405365)], [(0,108), (35.50308179829647,122.38226070500868)], [(0,120), (35.6244225900973,122.56546280213388)], [(0,132), (35.740319967518985,122.74193701444)], [(0,144), (35.85122575724692,122.91219746751261)], [(0,156), (35.95753627108441,123.07669735990757)], [(0,168), (36.05960113255078,123.23583840383684)], [(0,180), (36.15773059092413,123.38997878012647)], [(0,192), (36.25220129823302,123.53943941719763)], [(0,204), (36.34326112524709,123.68450917439682)], [(0,216), (36.431133080304164,123.82544917388462)], [(0,228), (36.51601861927481,123.96249635686894)], [(0,240), (36.598100367032636,124.0958665235991)], [(0,252), (36.67754446843046,124.22575686465295)], [(0,264), (36.75450250378583,124.35234808540766)], [(0,276), (36.82911324489226,124.47580637785794)], [(0,288), (36.9015039992668,124.59628484061923)], [(0,300), (36.97179193934181,124.71392502582216)], [(0,312), (37.04008512010451,124.82885800309118)], [(0,324), (37.106483422548045,124.94120547039817)], [(0,336), (37.17107939375068,125.0510806455637)], [(0,348), (37.23395891677719,125.15858903176915)], [(0,360), (37.295201903219834,125.26382922457466)], [(0,372), (37.35488277331454,125.3668933930521)]
    # landmarks = np.array(landmarks)
    landmarks_dict = {
            "frame": [],
            "uv" : [],
            "lonlat" : [],
            "confidence" : [],
        }
    time_idx = []
    ii = []
    filler_idx = 1
    for i in range(len(landmarks)-2):
        num_points = 0
        while  filler_idx*1000 < landmarks[i,0]:
            time_idx.append(filler_idx*1000)
            filler_idx += 1
        for j in range(len(landmarks[i,1])):
            # if landmarks[i,1][j][3] < 0.5:
            #     continue
            landmarks_dict["frame"].append(landmarks[i,0])
            landmarks_dict["uv"].append(landmarks[i,1][j][:2])
            if not sample_dets:
                landmarks_dict["lonlat"].append(landmarks[i,1][j][2])#[2:4])#
                landmarks_dict["confidence"].append(landmarks[i,1][j][3])#[4])#
            else:
                landmarks_dict["lonlat"].append(landmarks[i,1][j][2:4])
                landmarks_dict["confidence"].append(landmarks[i,1][j][4])
            ii.append(len(time_idx))
            num_points += 1
        if num_points > 0:
            time_idx.append(landmarks[i,0])


    # ipdb.set_trace()
    ii = np.array(ii)
    time_idx = np.array(time_idx)# - 1
    landmarks_dict["frame"] = np.array(landmarks_dict["frame"])
    landmarks_dict["uv"] = np.array(landmarks_dict["uv"])
    landmarks_dict["lonlat"] = np.array(landmarks_dict["lonlat"])
    landmarks_dict["confidence"] = np.array(landmarks_dict["confidence"])

    with open('landmarks/orbit_eci_quat.txt', 'r') as infile:
        orbit = json.load(infile)
    orbit = np.array(orbit)
    # orbit_lat_long = np.genfromtxt('data/lonlat.csv', delimiter=',') #np.array(pandas.read_csv("data/lonlat.csv", sep=','))
    # altitudes = np.genfromtxt('data/altitudes.csv')#np.array(pandas.read_csv("data/altitudes.csv"))[:,0]
    # intrinsics = np.genfromtxt("data/intrinsics.csv", delimiter=',') #  might have to specify manually

    # data = np.genfromtxt('data1/pixels.csv', delimiter=',')# np.array(pandas.read_csv("data/one_pass_output.csv" , sep=','))
    # orbit_lat_long = np.genfromtxt('data1/lonlat.csv', delimiter=',') #np.array(pandas.read_csv("data/lonlat.csv", sep=','))
    # altitudes = orbit_lat_long[:,-1]/1000 #np.genfromtxt('data1/altitudes.csv')#np.array(pandas.read_csv("data/altitudes.csv"))[:,0]
    # orbit_lat_long = orbit_lat_long[:,:-1]
    intrinsics = np.genfromtxt("landmarks/intrinsics.csv", delimiter=',')[0] #  might have to specify manually
    # ii = data[:,0]
    # intrinsics = np.array(intrinsics)

    gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration = process_ground_truths(orbit, landmarks_dict, intrinsics, dt, time_idx)

    # gt_acceleration = compute_acceleration_from_omega(gt_omega, dt)

    ### Obtain acceleration from orbital dynamics and angular velocity from IMU
    # dyn_params = get_all_r_sun_moon_PN()
    # x = torch.cat([gt_pos_eci, gt_vel_eci], dim=1)
    # t, params = dyn_params[-1], dyn_params[:-1]
    # gt_acceleration = RK4_orbit_dynamics_avg(x, h) #RK4_avg(x, t, h, params)
    # gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    # landmark_xyz1 = torch.tensor(np.array([[-2.9365e+06,  3.7009e+06,  4.2706e+06]]))/1e3
    # landmark_uv_proj = landmark_project(poses_gt_eci[:1].unsqueeze(0), landmarks_xyz[:1].unsqueeze(0), intrinsics[:1].unsqueeze(0), ii[:1], jacobian=False)
    landmark_uv_proj = landmark_project(poses_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), ii, jacobian=False)
    # print("landmark_uv_proj", landmark_uv_proj[0,:10])
    # print("landmark_uv_proj", landmarks_uv[:10])
    mask = ((landmark_uv_proj[:, :, 0] > 0)*(landmark_uv_proj[:, :, 1] > 0)*(landmark_uv_proj[:, :, 0] < 2600)*(landmark_uv_proj[:, :, 1] < 2000))[0].float().unsqueeze(-1)
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)*mask).abs().mean(dim=0))
    ipdb.set_trace()
    landmarks_uv = landmark_uv_proj[0, :]
    
    
    ### Initial guess for poses, velocities
    # ipdb.set_trace()
    # offset = torch.tensor([1, 1, 1, 0, 0, 0, 0])[None, None].repeat(1, T, 1)*100

    T = len(gt_pos_eci)
    N  = max(time_idx[1:] - time_idx[:-1])
    gt_omega = compute_omega_from_quat(gt_quat_eci_full, dt)#
    velocities = torch.zeros((1, T, N, 3))
    omegas = torch.zeros((1, T, N, 3))
    accelerations = torch.zeros((1, T, N, 3))
    # ipdb.set_trace()
    # velocities[:, :, 0, :] = gt_vel_eci.unsqueeze(0).float()
    # omegas[:, :, 0, :] = gt_omega.unsqueeze(0).float()
    # accelerations[:, :, 0, :] = gt_acceleration.unsqueeze(0).float()
    for i in range(1, T):
        velocities[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_vel_eci[time_idx[i-1]:time_idx[i], :].unsqueeze(0).float()
        omegas[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_omega[time_idx[i-1]:time_idx[i], :].unsqueeze(0).float()
        accelerations[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_acceleration[time_idx[i-1]:time_idx[i], :].unsqueeze(0).float()
    imu_meas = torch.cat((omegas, accelerations), dim=-1)   # for now, assume that the IMU gives us the accurate angular velocity and acceleration
    position_offset = torch.randn((T, 3))*100
    # position_offset[0, :] = 0
    orientation_offset = torch.randn([T, 3])*0.2
    # orientation_offset[0, :] = 0
    position = poses_gt_eci.float()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.float()[:, 3:]) + orientation_offset)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).float() + offset# torch.zeros(1, T, 7)
    # velocities = gt_vel_eci.unsqueeze(0).float() # torch.zeros(1, T, 3)
    # imu_meas = imu_meas.unsqueeze(0)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)

    for i in range(num_iters):
        poses, velocities = BA(poses, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, time_idx, intrinsics, Sigma, V, poses_gt_eci)


