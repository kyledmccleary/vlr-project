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
    position = poses_gt_eci.double()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.double()[:, 3:]) + orientation_offset)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).double() + offset# torch.zeros(1, T, 7)
    velocities = gt_vel_eci.unsqueeze(0).double() # torch.zeros(1, T, 3)
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
    # zc, yc, xc = orbit[:, 3:6], orbit[:, 6:9], orbit[:, 9:12]
    gt_quat_eci_full = convert_xyz_orientation_to_quat(xc, yc, zc, np.arange(len(orbit)))
    # ipdb.set_trace()
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
    landmarks_uv = torch.tensor(landmarks_uv).double()
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).repeat(len(gt_pos_eci), 1)
    return gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration

def read_data(sample_dets=False):
    if not sample_dets:
        landmarks = np.load("landmarks/all_dets.npy", allow_pickle=True)
        # landmarks = np.load("landmarks/dets.npy", allow_pickle=True)
    else:
        landmarks = np.load("landmarks/sample_dets.npy", allow_pickle=True)
    landmarks_dict = {
            "frame": [],
            "uv" : [],
            "lonlat" : [],
            "confidence" : [],
        }
    # times = np.load("landmarks/times.npy", allow_pickle=True)
    ipdb.set_trace()
    time_idx = []
    ii = []
    filler_idx = 1
    for i in range(len(landmarks)):#10, 25):#
        # if i > 13 and i <21:
        #     continue
        if i <9 or i > 30:
            continue
        num_points = 0
        # while  filler_idx*1000 < times[landmarks[i,0]]:
        # while  filler_idx*1000 < landmarks[i,0]:
        #     time_idx.append(filler_idx*1000)
        #     filler_idx += 1
        for j in range(len(landmarks[i,1])):
            # if landmarks[i,1][j][3] < 0.5:
            #     continue
            # landmarks_dict["frame"].append(times[landmarks[i,0]])
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
            # time_idx.append(times[landmarks[i,0]])
            time_idx.append(landmarks[i,0])
            # print(i, time_idx)
    # ipdb.set_trace()
    ii = np.array(ii)
    time_idx = np.array(time_idx)# - 1
    landmarks_dict["frame"] = np.array(landmarks_dict["frame"])
    landmarks_dict["uv"] = np.array(landmarks_dict["uv"])
    landmarks_dict["lonlat"] = np.array(landmarks_dict["lonlat"])
    landmarks_dict["confidence"] = np.array(landmarks_dict["confidence"])

    # with open('landmarks/orbit_eci_quat2.txt', 'r') as infile:
    with open('landmarks/orbit_eci_quat.txt', 'r') as infile:
        orbit = json.load(infile)
    orbit = np.array(orbit)

    intrinsics = np.genfromtxt("landmarks/intrinsics.csv", delimiter=',')[0] #  might have to specify manually
    return orbit, landmarks_dict, intrinsics, time_idx, ii

def read_detections(sample_dets=False):
    landmarks = np.load("landmarks/detections.npy", allow_pickle=True)
    landmarks_dict = {}
    # ipdb.set_trace()
    landmarks_dict["frame"] = landmarks[:,0]
    landmarks_dict["uv"] = landmarks[:,1:3]
    landmarks_dict["lonlat"] = landmarks[:,3:5]
    landmarks_dict["confidence"] = landmarks[:,5]
    time_idx = np.unique(landmarks[:,0]).astype(np.int64)
    ii = []
    for i, tidx in enumerate(time_idx):
        num_points = (landmarks[:,0]==tidx).sum()
        ii = ii + [i]*num_points
    ii = np.array(ii)
    with open('landmarks/seq.txt', 'r') as infile:
        orbit = json.load(infile)
    orbit = np.array(orbit)
    # ipdb.set_trace()
    orbit[:,0], orbit[:,1], orbit[:,2] = ecef_to_eci(orbit[:,0], orbit[:,1], orbit[:,2], times = np.arange(orbit.shape[0]))

    intrinsics = np.genfromtxt("landmarks/intrinsics.csv", delimiter=',')[0] #  might have to specify manually
    return orbit, landmarks_dict, intrinsics, time_idx, ii

def remove_elems(mask, gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx):
    ii_old = ii[mask][:-5]
    import copy
    ii_new = copy.deepcopy(ii[mask][:-5])
    mask_poses = np.unique(ii_old)
    for i in range(ii_old.max()):
        if (i==ii_old).sum() > 2:
            mask_poses.append(i)
        else:
            mask1 = (ii_old != i)
            mask = mask*mask1
            
    for i in range(ii_old.max()):
        if i not in ii_old:
            mask1 = ii_old > i
            ii_new[mask1] = ii_new[mask1] - 1
            # ipdb.set_trace()
    # ipdb.set_trace()
    mask_poses = np.array(mask_poses)
    gt_pos_eci = gt_pos_eci[mask_poses]
    # gt_vel_eci = gt_vel_eci[mask_poses]
    poses_gt_eci = poses_gt_eci[mask_poses]
    gt_quat_eci = gt_quat_eci[mask_poses]
    # gt_quat_eci_full = gt_quat_eci_full[mask_poses]
    # gt_acceleration = gt_acceleration[mask_poses]
    time_idx = time_idx[mask_poses]
    return gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii_new, time_idx
        
    

if __name__ == "__main__":

    ### Specify hyperparameters
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-3
    num_iters = 100
    torch.set_printoptions(precision=4, sci_mode=False)

    ### Read data 
    sample_dets = False
    orbit, landmarks_dict, intrinsics, time_idx, ii = read_data(sample_dets)
    # orbit, landmarks_dict, intrinsics, time_idx, ii = read_detections(sample_dets)
    gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration = process_ground_truths(orbit, landmarks_dict, intrinsics, dt, time_idx)

    ### Obtain acceleration from orbital dynamics and angular velocity from IMU
    # dyn_params = get_all_r_sun_moon_PN()
    # x = torch.cat([gt_pos_eci, gt_vel_eci], dim=1)
    # t, params = dyn_params[-1], dyn_params[:-1]
    # gt_acceleration = RK4_orbit_dynamics_avg(x, h) #RK4_avg(x, t, h, params)
    # gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    landmark_uv_proj = landmark_project(poses_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), ii, jacobian=False)
    mask = ((landmark_uv_proj[:, :, 0] > 0)*(landmark_uv_proj[:, :, 1] > 0)*(landmark_uv_proj[:, :, 0] < 2600)*(landmark_uv_proj[:, :, 1] < 2000)*((landmark_uv_proj - landmarks_uv[None]).norm(dim=-1)<1000) )[0]
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)*mask.double().unsqueeze(-1)).abs().mean(dim=0))
    print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[:,None], landmark_uv_proj[0]], dim=-1)[:20])
    # mask[17] = False
    # gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx, mask = remove_elems(mask, gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx)
    # ipdb.set_trace()
    ii = ii[mask]#[:-5]
    landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask], landmarks_uv[mask], landmark_uv_proj[:, mask]
    # landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask][:-5], landmarks_uv[mask][:-5], landmark_uv_proj[:, mask][:,:-5]#, ii[mask][:-5]
    confidences = torch.tensor(landmarks_dict["confidence"])[mask].double()#[:-5]
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)).abs().mean(dim=0))
    # print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[mask][:,None], landmark_uv_proj[0]], dim=-1)[:100])
    ipdb.set_trace()
    noise_level = 0.0#5
    landmarks_uv += (landmark_uv_proj[0, :] - landmarks_uv)*(1-noise_level)
        
    ### Initial guess for poses, velocities
    # offset = torch.tensor([1, 1, 1, 0, 0, 0, 0])[None, None].repeat(1, T, 1)*100

    T = len(gt_pos_eci)
    N  = max(time_idx[1:] - time_idx[:-1])
    gt_omega = compute_omega_from_quat(gt_quat_eci_full, dt)#
    velocities = torch.zeros((1, T, N, 3))
    omegas = torch.zeros((1, T, N, 3))
    accelerations = torch.zeros((1, T, N, 3))
    # ipdb.set_trace()
    # velocities[:, :, 0, :] = gt_vel_eci.unsqueeze(0).double()
    # omegas[:, :, 0, :] = gt_omega.unsqueeze(0).double()
    # accelerations[:, :, 0, :] = gt_acceleration.unsqueeze(0).double()
    for i in range(1, T):
        velocities[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_vel_eci[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        omegas[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_omega[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        accelerations[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_acceleration[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
    imu_meas = torch.cat((omegas, accelerations), dim=-1)   # for now, assume that the IMU gives us the accurate angular velocity and acceleration
    position_offset = torch.randn((T, 3))*0#*100
    # position_offset[0, :] = 0
    orientation_offset = torch.randn([T, 3])*0.2
    orientation_offset[0, :] = 0
    position = poses_gt_eci.double()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.double()[:, 3:]) + orientation_offset)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).double() + offset# torch.zeros(1, T, 7)
    # velocities = gt_vel_eci.unsqueeze(0).double() # torch.zeros(1, T, 3)
    # imu_meas = imu_meas.unsqueeze(0)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)
    lamda_init = 1e-4

    for i in range(num_iters):
        poses, velocities, lamda_init = BA(i, poses, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci)
        if i%5==0:
            ipdb.set_trace()


