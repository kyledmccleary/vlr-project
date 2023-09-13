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
from trajgen_pipe import *

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

    gt_pos_eci = orbit[time_idx,:3]
    #convert_latlong_to_cartesian(orbit_lat_long[:,1], orbit_lat_long[:,0], altitudes)
    gt_vel_eci = compute_velocity_from_pos(orbit[:,:3], dt) 
    # zc, yc, xc = convert_quaternion_to_xyz_orientation(orbit[:,6:10], np.arange(len(orbit)))
    zc, yc, xc = orbit[:, 3:6], orbit[:, 6:9], orbit[:, 9:12]
    gt_quat_eci_full = convert_xyz_orientation_to_quat(xc, yc, zc, np.arange(len(orbit)))
    gt_quat_eci = gt_quat_eci_full[time_idx, :]
    poses_gt_eci = np.concatenate([gt_pos_eci, gt_quat_eci], axis=1)
    # poses_gt_eci = np.concatenate([orbit[:,:3], orbit[:,6:10]], axis=1)
    landmarks_xyz = convert_latlong_to_cartesian(landmarks_dict["lonlat"][:, 1], landmarks_dict["lonlat"][:,0], landmarks_dict["frame"])#data["y(latitude)"], data["x(longitude)"]) 
    landmarks_uv = landmarks_dict["uv"]
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
    time_idx = []
    ii = []
    filler_idx = 1
    for i in range(len(landmarks)):#10, 25):#
        # if i > 13 and i <21:
        #     continue
        # if (i>4 and i <10):
        #     continue
        # if (not i == 9) and (not i == 30):
        #     continue
        if i <10:# or i>24:
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
                landmarks_dict["lonlat"].append(landmarks[i,1][j][2])
                landmarks_dict["confidence"].append(landmarks[i,1][j][3])
            else:
                landmarks_dict["lonlat"].append(landmarks[i,1][j][2:4])
                landmarks_dict["confidence"].append(landmarks[i,1][j][4])
            ii.append(len(time_idx))
            num_points += 1
        if num_points > 0:
            # time_idx.append(times[landmarks[i,0]])
            time_idx.append(landmarks[i,0])
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
    # landmarks = np.load("landmarks/rand_detections_0.npy", allow_pickle=True)
    landmarks = np.load("landmarks/detections.npy", allow_pickle=True)
    landmarks[:,0]+=1
    landmarks_dict = {}
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
    with open('landmarks/orbit_3hr_noskip.txt', 'r') as infile:
        orbit = json.load(infile)
    orbit = np.array(orbit)
    orbit[:,0], orbit[:,1], orbit[:,2] = ecef_to_eci(orbit[:,0]/1000, orbit[:,1]/1000, orbit[:,2]/1000, times = np.arange(orbit.shape[0]))
    # orbit = np.load("landmarks/rand_seq_0.npy")
    #print(orbit[0,:], "\n", orbit2[0,:])
    #print(orbit.shape, orbit2.shape, landmarks.shape, time_idx.shape)
    #print(time_idx.shape)

    intrinsics = np.genfromtxt("landmarks/intrinsics.csv", delimiter=',')[0] #  might have to specify manually
    return orbit, landmarks_dict, intrinsics, time_idx, ii

def remove_elems(mask, gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx):
    ii_old = ii[mask][:-5]
    import copy
    ii_new = copy.deepcopy(ii[mask][:-5])
    mask_poses = np.unique(ii_old)
    # for i in range(ii_old.max()):
    #     if (i==ii_old).sum() > 2:
    #         mask_poses.append(i)
    #     else:
    #         mask1 = (ii_old != i)
    #         mask = mask*mask1
            
    for i in range(ii_old.max()):
        if i not in ii_old:
            mask1 = ii_old > i
            ii_new[mask1] = ii_new[mask1] - 1
    mask_poses = np.array(mask_poses)
    gt_pos_eci = gt_pos_eci[mask_poses]
    poses_gt_eci = poses_gt_eci[mask_poses]
    gt_quat_eci = gt_quat_eci[mask_poses]
    time_idx = time_idx[mask_poses]
    return gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii_new, time_idx, mask
        
def add_proxy_landmarks(landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, intrinsics, poses_gt_eci, confidences):
    idx = np.unique(ii)
    N = 8
    for i in idx:
        mask = (ii == i)
        if mask.sum() > 4:
            continue
        noise = torch.randn((N, 2))*20
        offset = torch.randn((N, 3))*50
        landmarks_xyz = torch.cat([landmarks_xyz, landmarks_xyz[mask][:1] + offset], dim=0)
        ii = np.concatenate([ii]+ [ii[mask][:1]]*N, axis=0)
        proj = landmark_project(poses_gt_eci.unsqueeze(0), (landmarks_xyz[-N:])[None], intrinsics.unsqueeze(0), ii[-N:], jacobian=False)
        landmarks_uv = torch.cat([landmarks_uv,proj[0]+noise], dim=0)
        landmark_uv_proj = torch.cat([landmark_uv_proj, proj], dim=1)
        confidences = torch.cat([confidences, torch.ones(N)*0.75], dim=0)
    return landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, confidences

def seeding(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)    

if __name__ == "__main__":
    ### Specify hyperparameters
    seeding(0)
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-3
    num_iters = 100
    torch.set_printoptions(precision=4, sci_mode=False)

    ### Read data 
    sample_dets = False
    # orbit, landmarks_dict, intrinsics, time_idx, ii = read_data(sample_dets)
    orbit, landmarks_dict, intrinsics, time_idx, ii = read_detections(sample_dets)
    gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration = process_ground_truths(orbit, landmarks_dict, intrinsics, dt, time_idx)

    ### Obtain acceleration from orbital dynamics and angular velocity from IMU
    states_gt_eci = torch.cat([poses_gt_eci, gt_vel_eci[time_idx]], dim=-1)
    landmark_uv_proj = landmark_project(states_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), ii, jacobian=False)
    ipdb.set_trace()
    mask = ((landmark_uv_proj[:, :, 0] > 0)*(landmark_uv_proj[:, :, 1] > 0)*(landmark_uv_proj[:, :, 0] < 2600)*(landmark_uv_proj[:, :, 1] < 2000)*((landmark_uv_proj - landmarks_uv[None]).norm(dim=-1)<1000)*(torch.tensor(landmarks_dict["confidence"])>0.5) )[0]
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)*mask.double().unsqueeze(-1)).abs().mean(dim=0))
    print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[:,None], landmark_uv_proj[0], landmarks_uv], dim=-1)[mask][:20])
    ii = ii[mask]#[:-5]
    landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask], landmarks_uv[mask], landmark_uv_proj[:, mask]
    # landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask][:-5], landmarks_uv[mask][:-5], landmark_uv_proj[:, mask][:,:-5]#, ii[mask][:-5]
    confidences = torch.tensor(landmarks_dict["confidence"])[mask].double()#[:-5]
    # landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, confidences = add_proxy_landmarks(landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, intrinsics, poses_gt_eci, confidences)
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)).abs().mean(dim=0))
    ipdb.set_trace()
    noise_level = 0
    landmarks_uv += (landmark_uv_proj[0, :] - landmarks_uv)*(1-noise_level)
        
    ### Initial guess for poses, velocities
    T = len(gt_pos_eci)
    N  = max(time_idx[1:] - time_idx[:-1])
    gt_omega = compute_omega_from_quat(gt_quat_eci_full, dt)
    # velocities = torch.zeros((1, T, N, 3))
    velocities = gt_vel_eci[time_idx].unsqueeze(0).double()
    ipdb.set_trace()
    omegas = torch.zeros((1, T, N, 3))
    accelerations = torch.zeros((1, T, N, 3))
    for i in range(1, T):
        # velocities[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_vel_eci[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        omegas[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_omega[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        accelerations[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_acceleration[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
    imu_meas = torch.cat((omegas, accelerations), dim=-1)   # for now, assume that the IMU gives us the accurate angular velocity and acceleration (we don't use the accelerations, we just use the dynamics)
    position_offset = torch.randn((T, 3))*100
    orientation_offset = torch.randn([T, 3])*0.2
    velocity_offset = torch.randn([T, 3])*velocities.abs().mean()*0.1
    position = poses_gt_eci.double()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.double()[:, 3:]) + orientation_offset)
    vels = velocities.double() + velocity_offset.unsqueeze(0)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    states = torch.cat([poses, vels], dim=-1)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)
    lamda_init = 1e-4
    for i in range(num_iters):
        states, velocities, lamda_init = BA(i, states, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci)

def attitude_debugging():
    ### Specify hyperparameters
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-3
    num_iters = 100
    torch.set_printoptions(precision=4, sci_mode=False)
    seeding(0)

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
    mask = ((landmark_uv_proj[:, :, 0] > 0)*(landmark_uv_proj[:, :, 1] > 0)*(landmark_uv_proj[:, :, 0] < 2600)*(landmark_uv_proj[:, :, 1] < 2000)*((landmark_uv_proj - landmarks_uv[None]).norm(dim=-1)<1000) )[0]*0 + 1
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)*mask.double().unsqueeze(-1)).abs().mean(dim=0))
    print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[:,None], landmark_uv_proj[0]], dim=-1)[:20])
    ii = ii[mask]#[:-5]
    landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask], landmarks_uv[mask], landmark_uv_proj[:, mask]
    confidences = torch.tensor(landmarks_dict["confidence"])[mask].double()#[:-5]
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)).abs().mean(dim=0))
    ipdb.set_trace()
    noise_level = 0.0#5
    landmarks_uv += (landmark_uv_proj[0, :] - landmarks_uv)*(1-noise_level)
        
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
    orientation =quaternion_exp(quaternion_log(poses_gt_eci.double()[:, 3:]) + orientation_offset)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).double() + offset# torch.zeros(1, T, 7)
    # velocities = gt_vel_eci.unsqueeze(0).double() # torch.zeros(1, T, 3)
    # imu_meas = imu_meas.unsqueeze(0)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)
    lamda_init = 1e-4

    for i in range(num_iters):
        # poses, velocities, lamda_init = BA(i, poses, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci)
        landmarks = landmarks_uv
        iter = i
        poses = poses.double()
        # ipdb.set_trace()
        v = velocities.double()
        imu_meas = imu_meas.double()
        landmarks = landmarks.double()
        landmarks_xyz = landmarks_xyz.double()
        intrinsics = intrinsics.double()
        # time_idx = time_idx.double()
        quat_coeff = 100 #+ min(iter*10, 900)
        dim = 3

        bsz = poses.shape[0]
        r_pred, Hq, qgrad = predict_attitude(poses, velocities, imu_meas, time_idx, quat_coeff, jacobian=True) 
        
        n = poses.shape[1]    
        JTr = - qgrad.reshape(bsz, n, dim).reshape(bsz, -1)
        # ipdb.set_trace()
        lamda = lamda_init
        init_residual = torch.cat([r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).abs().mean()
        # Hq6 = torch.cat([torch.cat([torch.zeros(n*dim, n*dim).reshape(1, n, dim, n, dim), Hq.reshape(bsz, n, dim, n, dim)*0], dim=-1),
        #                  torch.cat([Hq.new_zeros(bsz, n, dim, n, 3), Hq.reshape(bsz, n, dim, n, dim)], dim=-1)], dim=2).reshape(bsz, n*dim*2, n*dim*2)
        # grad6 = -torch.cat([torch.zeros(bsz, n, dim), qgrad], dim=-1).reshape(bsz, -1)
        # JTwJ6 =  torch.eye(n*dim*2)[None]*lamda + Hq6
        # dpose6 = torch.linalg.solve(JTwJ6, grad6).reshape(bsz, n, dim*2) 
        # JTwJ =  torch.eye(n*dim)[None]*lamda + Hq
        # dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)
        # ipdb.set_trace()
        while True:

            JTwJ =  torch.eye(n*dim)[None]*lamda + Hq#*100#.01 #+ torch.eye(n*dim)[None]*1e-5+#.1 # 
            # JTwJ.view(bsz, n, dim, n, dim)[:, :, 3:, :, 3:] += Hq.view(bsz, n, 3, n, 3)
            try:
                # ipdb.set_trace()
                dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)
            
                # ipdb.set_trace()
                position = poses[:,:,:3] #+ dpose[:,:,:3]
                rotation = quaternion_multiply(poses[:,:,3:], quaternion_exp(dpose))#[:,:,3:]))
                rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
                rotation[:, 0] = poses[:, 0, 3:].clone()
                poses_new = torch.cat([position, rotation], 2)
                # landmark_est = landmark_project(poses_new, landmarks_xyz, intrinsics, ii, jacobian=False)
                r_pred1 = predict_attitude(poses_new, velocities, imu_meas, time_idx, quat_coeff, jacobian=False) 
                r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)#*0.01
                residual = r_pred1
                print("lamda: ", lamda, r_pred1.abs().mean())
            except:
                lamda = lamda*10
                if lamda > 1e4:
                    print("lamda too large")
                    ipdb.set_trace()
                
                continue

            lamda = lamda*10
            if (residual.abs().mean()) < init_residual:
                break
            if lamda > 1e4:
                print("lamda too large")
                break
            
        lamda_init = max(min(1e-1, lamda*0.01), 10)

        ## backtracking line search
        alpha = 1
        init_residual = (r_pred.reshape(-1)*np.sqrt(Sigma)).norm()
        # init_residual = (r_obs.abs()).sum() + (r_pred.abs()).sum()*np.sqrt(Sigma)
        print("alpha: ", alpha, r_pred.abs().mean()* np.sqrt(Sigma))
        print("final quat: ", (poses_new[0,:, 3:]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((poses_new[0,:, 3:] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
        # print("final: ", (poses_new2[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
        # print("final: ", (poses_new3[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
        print("init quat: ", (poses[0,:, 3:] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((poses[0,:, 3:] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
        print("final pos: ", (poses_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0))
        print("init pos: ", (poses[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
        print("r_pred :", r_pred[0,:].abs().mean(dim=0))
        # ipdb.set_trace()
        poses = poses_new
        # , velocities, lamda_init

        # print((poses_gt_eci[:,3:]*poses[0,:,3:]).sum(dim=-1))
        predq = propagate_rotation_dynamics(poses[:, :, 3:], imu_meas[..., :3].double(), time_idx, 1, False)[0]
        print((predq[0,:-1,:]*poses[0,1:,3:]).sum(dim=-1))
        if i%5==0:

            ipdb.set_trace()



# if __name__ == "__main__":
def dynamics_debugging():
    ### Specify hyperparameters
    seeding(0)
    h = 1 # Frequency = 1 Hz
    dt = 1/h
    V = 1e-3
    Sigma = 1e-4
    num_iters = 100
    torch.set_printoptions(precision=4, sci_mode=False)

    ### Read data 
    sample_dets = False
    # orbit, landmarks_dict, intrinsics, time_idx, ii = read_data(sample_dets)
    orbit, landmarks_dict, intrinsics, time_idx, ii = read_detections(sample_dets)
    gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration = process_ground_truths(orbit, landmarks_dict, intrinsics, dt, time_idx)

    ### Obtain acceleration from orbital dynamics and angular velocity from IMU
    # dyn_params = get_all_r_sun_moon_PN()
    # x = torch.cat([gt_pos_eci, gt_vel_eci], dim=1)
    # t, params = dyn_params[-1], dyn_params[:-1]
    # gt_acceleration = RK4_orbit_dynamics_avg(x, h) #RK4_avg(x, t, h, params)
    # gt_acceleration = compute_velocity_from_pos(gt_vel_eci, dt)
    states_gt_eci = torch.cat([poses_gt_eci, gt_vel_eci[time_idx]], dim=-1)
    landmark_uv_proj = landmark_project(states_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), ii, jacobian=False)
    mask = ((landmark_uv_proj[:, :, 0] > 0)*(landmark_uv_proj[:, :, 1] > 0)*(landmark_uv_proj[:, :, 0] < 2600)*(landmark_uv_proj[:, :, 1] < 2000)*((landmark_uv_proj - landmarks_uv[None]).norm(dim=-1)<1000)*(torch.tensor(landmarks_dict["confidence"])>0.8) )[0]
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)*mask.double().unsqueeze(-1)).abs().mean(dim=0))
    print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[:,None], landmark_uv_proj[0]], dim=-1)[mask][:20])
    # ipdb.set_trace()
    # mask[17] = False
    # gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx, mask = remove_elems(mask, gt_pos_eci, gt_vel_eci, poses_gt_eci, gt_quat_eci, gt_quat_eci_full, landmarks_xyz, landmarks_uv, intrinsics, gt_acceleration, ii, time_idx)
    # ipdb.set_trace()
    ii = ii[mask]#[:-5]
    landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask], landmarks_uv[mask], landmark_uv_proj[:, mask]
    # landmarks_xyz, landmarks_uv, landmark_uv_proj = landmarks_xyz[mask][:-5], landmarks_uv[mask][:-5], landmark_uv_proj[:, mask][:,:-5]#, ii[mask][:-5]
    confidences = torch.tensor(landmarks_dict["confidence"])[mask].double()#[:-5]
    # landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, confidences = add_proxy_landmarks(landmarks_xyz, landmarks_uv, landmark_uv_proj, ii, intrinsics, poses_gt_eci, confidences)
    print("mean landmark difference : ", ((landmark_uv_proj[0,:] - landmarks_uv)).abs().mean(dim=0))
    # print(torch.cat([(landmark_uv_proj[0,:] - landmarks_uv), torch.tensor(landmarks_dict["confidence"])[mask][:,None], landmark_uv_proj[0]], dim=-1)[:100])
    ipdb.set_trace()
    noise_level = 0.0
    landmarks_uv += (landmark_uv_proj[0, :] - landmarks_uv)*(1-noise_level)
        
    ### Initial guess for poses, velocities
    # offset = torch.tensor([1, 1, 1, 0, 0, 0, 0])[None, None].repeat(1, T, 1)*100

    T = len(gt_pos_eci)
    N  = max(time_idx[1:] - time_idx[:-1])
    gt_omega = compute_omega_from_quat(gt_quat_eci_full, dt)#
    # velocities = torch.zeros((1, T, N, 3))
    velocities = gt_vel_eci[time_idx].unsqueeze(0).double() # torch.zeros(1, T, 3)
    ipdb.set_trace()
    omegas = torch.zeros((1, T, N, 3))
    accelerations = torch.zeros((1, T, N, 3))
    # ipdb.set_trace()
    # velocities[:, :, 0, :] = gt_vel_eci.unsqueeze(0).double()
    # omegas[:, :, 0, :] = gt_omega.unsqueeze(0).double()
    # accelerations[:, :, 0, :] = gt_acceleration.unsqueeze(0).double()
    for i in range(1, T):
        # velocities[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_vel_eci[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        omegas[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_omega[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
        accelerations[:, i-1, :time_idx[i]-time_idx[i-1], :] = gt_acceleration[time_idx[i-1]:time_idx[i], :].unsqueeze(0).double()
    imu_meas = torch.cat((omegas, accelerations), dim=-1)   # for now, assume that the IMU gives us the accurate angular velocity and acceleration
    position_offset = torch.randn((T, 3))*0#*100
    # position_offset[0, :] = 0
    orientation_offset = torch.randn([T, 3])*0#.2
    # orientation_offset[0, :] = 0
    velocity_offset = torch.randn([T, 3])*velocities.abs().mean()*0#.1
    position = poses_gt_eci.double()[:, :3] + position_offset
    orientation = quaternion_exp(quaternion_log(poses_gt_eci.double()[:, 3:]) + orientation_offset)
    vels = velocities.double() + velocity_offset.unsqueeze(0)
    poses = torch.cat([position, orientation], dim=1).unsqueeze(0)
    states = torch.cat([position, vels[0]], dim=-1).unsqueeze(0)
    states_orig = torch.cat([position, orientation, vels[0]], dim=-1).unsqueeze(0)
    # poses = poses_gt_eci.unsqueeze(0).double() + offset# torch.zeros(1, T, 7)
    # velocities = gt_vel_eci.unsqueeze(0).double() # torch.zeros(1, T, 3)
    # imu_meas = imu_meas.unsqueeze(0)
    landmarks_uv = landmarks_uv.unsqueeze(0)
    landmarks_xyz = landmarks_xyz.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0)
    lamda_init = 1e-4
    # pos_pred, vel_pred = propagate_orbit_dynamics(states[:, :, :3], states[:, :, 7:], time_idx, 1)
    # ipdb.set_trace()

    for i in range(num_iters):
        # states, velocities, lamda_init = BA(i, states, velocities, imu_meas, landmarks_uv, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci)
        iter = i
        landmarks = landmarks_uv
        states = states.double()
        # ipdb.set_trace()
        v = velocities.double()
        imu_meas = imu_meas.double()
        landmarks = landmarks.double()
        landmarks_xyz = landmarks_xyz.double()
        intrinsics = intrinsics.double()
        # time_idx = time_idx.double()
        quat_coeff = 100 #+ min(iter*10, 900)
        vel_coeff = 1

        bsz = states.shape[0]
        landmark_est, Jg = landmark_project(states_orig, landmarks_xyz, intrinsics, ii, jacobian=True)
        r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf = predict_orbit(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True) 
        # ipdb.set_trace()
        # print(r_pred[:, :, -1].abs().mean(), (pose_pred[0,:-1,3:]*states[0,1:,3:]).sum(dim=-1).mean())
        r_obs = (landmarks - landmark_est)
        # ipdb.set_trace()	
        alpha = max(1 - (2*(iter/5) - 1), -4)
        c_obs = r_obs.abs().median() + 1e-10#1000 
        wts_obs = (((((r_obs/c_obs)**2)/abs(alpha-2) + 1)**(alpha/2 - 1)) / ((c_obs)**2)).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)[0]
        # ipdb.set_trace()
        wts_obs = (wts_obs/wts_obs.max())*confidences.unsqueeze(-1).unsqueeze(-1)*0 + 1
        # ipdb.set_trace()
        # r_full = torch.cat([r_obs, r_pred], dim = 1)
        Sigma = 1000*(iter+1)**2#1#00
        V = 1
        dim_base = 6
        dim = 6
        Jg = Jg.reshape(bsz, -1, 9)[:, :, :9].reshape(-1, 2, 9)
        Jg = torch.cat([Jg[:, :, :3], Jg[:, :, 6:]], dim=-1)
        
        JgTwJg = torch.bmm((Jg*wts_obs).transpose(1,2), Jg).reshape(bsz, -1, dim, dim)
        # scatter sum of Bii, Bij, Bji, Bjj and JgTwJg
        n = states.shape[1]
        ii_t = torch.tensor(ii, dtype=torch.long, device=states.device)
        JgTwJg = safe_scatter_add_vec(JgTwJg, ii_t, n).view(bsz, n, dim, dim)
        JgTwJg = torch.block_diag(*JgTwJg[0].unbind(dim=0)).unsqueeze(0)

        # ipdb.set_trace()
        dim2 = min(6, dim)
        Jf = Jf.view(bsz, (n-1)*dim2, n*dim)
        JfTwJf = torch.bmm((Jf*Sigma).transpose(1,2), Jf)
        # Hq = Sigma*torch.block_diag(*Hq[0].unbind(dim=0)).unsqueeze(0)*0

        # J_full = torch.cat([Jg, Jf], axis=1)
        # wts = torch.cat([V, Sigma], 1).unsqueeze(-1)

        # JTwJ = torch.bmm((J_full*wts).transpose(1,2), J_full)

        # JTr = torch.bmm((J_full*wts).transpose(1,2), r_full.unsqueeze(-1))

        # ipdb.set_trace()	
        r_pred = r_pred[:, :,  :dim]
        JgT_robs = safe_scatter_add_vec((Jg.reshape(bsz, -1, 2, dim)*wts_obs[None]*r_obs.unsqueeze(-1)).sum(dim=-2), ii_t, n).view(bsz, n,dim)
        # ipdb.set_trace()
        r_pred_x = r_pred[:, :, :6].clone()
        # r_pred_x[:, :, 3:] = 1
        JfT_rpred = (r_pred_x.reshape(bsz, -1, 1) * Sigma * Jf).sum(dim=1).reshape(bsz, n, dim)*(-1)#.01
        JTr = (JfT_rpred  + JgT_robs).reshape(bsz, -1)#+ JfT_rpred + JgT_robs - Sigma*qgrad

        lamda = lamda_init
        init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).abs().mean()#(r_pred*np.sqrt(Sigma)).abs().mean()#
        # if iter > 15:
        # 	ipdb.set_trace()
        while True:

            JTwJ =  torch.eye(n*dim)[None]*lamda + JfTwJf + JgTwJg# + Sigma*Hq#*100#.01 #+ torch.eye(n*dim)[None]*1e-5+#.1 # + JgTwJg
            # ipdb.set_trace()
            # JTwJ.view(bsz, n, dim, n, dim)[:, :, 3:, :, 3:] += Hq.view(bsz, n, 3, n, 3)
            # try:
            dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)
        
            ipdb.set_trace()
            position = states[:,:,:3] + dpose[:,:,:3]
            vels = states[:,:,3:] + dpose[:,:,3:]
            position[:, 0] = states[:, 0, :3].clone()
            # rotation = quaternion_multiply(states[:,:,3:7], quaternion_exp(dpose[:,:,3:6]))
            # rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
            # rotation[:, 0] = states[:, 0, 3:].clone()
            states_new = torch.cat([position, vels], 2)
            states_orig_new = torch.cat([position, orientation[None], vels], 2)
            landmark_est = landmark_project(states_orig_new, landmarks_xyz, intrinsics, ii, jacobian=False)
            r_pred1, _, _ = predict_orbit(states_new, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=False) 
            r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
            r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)#*0.01
            r_obs1 = r_obs1.reshape(bsz, -1)
            residual = torch.cat([r_obs1, r_pred1], dim = 1)# r_pred1#
            print("lamda: ", lamda, r_pred1.abs().mean())#, r_obs1.abs().mean())
            # except:
            #     lamda = lamda*10
            #     continue

            lamda = lamda*10
            if (residual.abs().mean()) < init_residual:
                break
            if lamda > 1e4:
                print("lamda too large")
                break
            
        lamda_init = max(min(1e-1, lamda*0.01), 1e-4)

        ## backtracking line search
        alpha = 1
        init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).norm()
        # init_residual = (r_obs.abs()).sum() + (r_pred.abs()).sum()*np.sqrt(Sigma)
        print("alpha: ", alpha, r_pred.abs().mean()* np.sqrt(Sigma), r_obs.abs().mean())
        # while True:
        # 	position = states[:,:,:3] + alpha*dpose[:,:,:3]
        # 	rotation = quaternion_multiply(states[:,:,3:], quaternion_exp(alpha*dpose[:,:,3:]))
        # 	rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
        # 	states_new = torch.cat([position, rotation], 2)
        # 	landmark_est = landmark_project(states_new, landmarks_xyz, intrinsics, ii, jacobian=False)
        # 	r_pred1, _, _ = predict(states_new, velocities, imu_meas, time_idx, quat_coeff, jacobian=False) 
        # 	r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
        # 	r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)#*0.01
        # 	r_obs1 = r_obs1.reshape(bsz, -1)
        # 	residual = torch.cat([r_obs1, r_pred1], dim = 1)
        # 	if (residual.norm()) < init_residual:
        # 		break
        # 	else:
        # 		alpha = alpha/2
        # 	print("alpha: ", alpha, r_pred1.abs().mean(), r_obs1.abs().mean())
        # 	if alpha < 1e-1:
        # 		print("alpha too small")
        # 		break
            

        # position = states[:,:,:3] + dpose[:,:,:3]
        # rotation = quaternion_multiply(states[:,:,3:], quaternion_exp(dpose[:,:,3:]))
        # # ipdb.set_trace()
        # # rotation = rotation[:, 1:]
        # # rotation = states[:,1:,3:] + dpose[:,1:,3:]
        # rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
        # states_new = torch.cat([position, rotation], 2)
        # # states_new = torch.cat([states[:,:1], states_new], 1)
        # # print("final: ", (states_new[0,:3, :3]-states_gt_eci[:3, :3]).abs().mean(dim=0))
        # # print("init: ", (states[0,:3, :3] - states_gt_eci[:3, :3]).abs().mean(dim=0))
        # # print("r_pred :", r_pred[0,:3].abs().mean(dim=0))
        # # print("r_obs :", r_obs[0,:3].abs().mean(dim=0))

        # print("final quat: ", (states_new[0,:, 3:7]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states_new[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
        # print("final: ", (poses_new2[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
        # print("final: ", (poses_new3[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
        # print("init quat: ", (states[0,:, 3:7] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
        print("final pos: ", (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0))
        print("init pos: ", (states[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
        print("final vels: ", (states_new[0,:, 3:]-velocities[0]).abs().mean(dim=0))
        print("init vels: ", (states[0,:, 3:] - velocities[0]).abs().mean(dim=0))
        print("r_pred :", r_pred[0,:].abs().mean(dim=0))
        # print("r_obs :", r_obs[0,:].abs().mean(dim=0))
        states = states_new
        if i%5==0:
            ipdb.set_trace()
