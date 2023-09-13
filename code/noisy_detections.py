from BA.BA_utils import *
from trajgen_pipe import *
#from BA.utils import *
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
import ipdb



def noisy_detections():
    for iter in range (1):
        traj, tsamp = generate_new_traj(orbit_type='polar')
        intrinsics = np.genfromtxt("landmarks/intrinsics.csv", delimiter=',')[0]
        #boxes10S = np.load("landmarks/10S_outboxes.npy")
        #boxes10S_center = centroid_boxes(boxes10S)
        #print(boxes10S_center.shape)
        #for centroid in boxes10S_center:
        #eci_pos = convert_latlong_to_cartesian(boxes10S_center[:,1], boxes10S_center[:,0],0)#np.zeros(50))
        boxes = load_boxes()
        boxes = np.concatenate(boxes, axis = 0)
        boxes_centroid = centroid_boxes(boxes)
        sample_ratio = 1
        boxes_centroid_sample = boxes_centroid[range(0,len(boxes_centroid),sample_ratio)]
        #pose_test = np.concatenate((traj[0,:3],traj[0,6:10]), axis=None)
        #eci_boxes = convert_latlong_to_cartesian(boxes_centroid[:,1], boxes_centroid[:,0],0)
        #uv = check_projection(pose_test, eci_pos[0,:],intrinsics)
        #print(uv)
        # ipdb.set_trace()
        detections = []
        positions = traj[:,:3]
        vels = traj[:,4:6]
        zc,yc,xc = convert_quaternion_to_xyz_orientation(traj[:,6:10], tsamp)
        camera_vecs = np.concatenate((zc,yc,xc),axis=1)
        #aprint(positions.shape)
        seq = np.concatenate((positions,camera_vecs),axis=1)
        for t in range(len(tsamp)):
            quat = np.concatenate((traj[t,7:10],traj[t,6:7]), axis=None)
            pose = np.concatenate((traj[t,:3],quat), axis=None)
            eci_boxes_frame = convert_latlong_to_cartesian(boxes_centroid_sample[:,1], boxes_centroid_sample[:,0],tsamp[t])
            # check_projection_torch(traj, boxes, intrinsics, tsamp[t] )
            poses = pose[None].repeat(len(eci_boxes_frame), axis=0)
            uv, mask = check_projections(poses, eci_boxes_frame, intrinsics)
        
            uv = uv[mask,:]
            detection = np.concatenate((t*np.ones((uv.shape[0],1)),uv,boxes_centroid_sample[mask,:],np.ones((uv.shape[0],1))),axis=1)

            detections.append(detection)
            # for k in range(len(eci_boxes_frame)):
            #     feature = eci_boxes_frame[k]
            #     #print(np.linalg.norm(pose[:3]),np.linalg.norm(feature))
            #     ipdb.set_trace()
            #     uv = check_projection(pose, feature,intrinsics)
            #     #print(uv)
            #     if uv is not None:
            #         #print(uv, pose[:3], feature)
            #         box_k = boxes_centroid_sample[k,:]
            #         #print(box_k)
            #         detection = np.zeros(6)
            #         detection[0], detection[1:3], detection[3:5], detection[5:] = t,uv,box_k,1
            #         detections.append(detection)
        detections = np.concatenate(detections, axis=0)
        ipdb.set_trace()
        #print(seq[0,:])
        np.save("landmarks/rand_detections_%s"%iter, detections)
        np.save("landmarks/rand_seq_%s"%iter, seq)
        #np.savetxt("seq_%s"%iter, seq, fmt='%.16f',)
        print(detections)

def check_projection_torch(orbit, boxes, intrinsics, tsamp ):
    gt_pos_eci = orbit[:,:3]
    landmarks_xyz = convert_latlong_to_cartesian(boxes[:,0], boxes[:,1],tsamp)
    #convert_latlong_to_cartesian(orbit_lat_long[:,1], orbit_lat_long[:,0], altitudes)
    gt_vel_eci = compute_velocity_from_pos(orbit[:,:3], 1) 
    zc, yc, xc = convert_quaternion_to_xyz_orientation(orbit[:,6:10], np.arange(len(orbit)))
    #zc, yc, xc = orbit[:, 3:6], orbit[:, 6:9], orbit[:, 9:12]
    gt_quat_eci_full = convert_xyz_orientation_to_quat(xc, yc, zc, np.arange(len(orbit)))
    gt_quat_eci = gt_quat_eci_full[:, :]
    poses_gt_eci = np.concatenate([gt_pos_eci, gt_quat_eci], axis=1)
    # poses_gt_eci = np.concatenate([orbit[:,:3], orbit[:,6:10]], axis=1)
    intrinsics = torch.tensor(intrinsics).unsqueeze(0).repeat(len(gt_pos_eci), 1)
    gt_acceleration = compute_velocity_from_pos(gt_vel_eci, 1)
    gt_acceleration = torch.tensor(gt_acceleration)
    gt_pos_eci = torch.tensor(gt_pos_eci)
    gt_vel_eci = torch.tensor(gt_vel_eci)
    poses_gt_eci = torch.tensor(poses_gt_eci)
    gt_quat_eci = torch.tensor(gt_quat_eci)
    gt_quat_eci_full = torch.tensor(gt_quat_eci_full)
    landmarks_xyz = torch.tensor(landmarks_xyz)
    states_gt_eci = torch.cat([poses_gt_eci, gt_vel_eci], dim=-1)
    landmark_uv_proj = landmark_project(states_gt_eci.unsqueeze(0), landmarks_xyz.unsqueeze(0), intrinsics.unsqueeze(0), tsamp, jacobian=False)
    print(landmark_uv_proj)

def check_projection(pose, point, intrinsics):
    x_pose = pose[:3]
    quat_pose = pose[3:]
    translation = point - x_pose
    pixel_error = 0
    if np.linalg.norm(translation)>1e3:
        return None
    point_rotated = apply_inverse_pose_transformation_np(point, quat_pose, x_pose)
    #print(point_rotated)
    uv = proj_np(point_rotated, intrinsics)
    if (0<uv[0]<2600 and 0<uv[1]<2000):
        #print(point, x_pose)
        noise = np.round(pixel_error*np.random.randn(2))
        uv = uv+noise
        return np.array(uv)
    else:
        return None

def check_projections(pose, point, intrinsics):
    x_pose = pose[:,:3]
    quat_pose = pose[:,3:]
    translation = point - x_pose
    pixel_error = 0
    # if np.linalg.norm(translation)>1e3:
    #     return None
    dist_mask = np.linalg.norm(translation, axis=1) < 1e3
    point_rotated = apply_inverse_pose_transformation_np(point, quat_pose, x_pose)
    #print(point_rotated)
    uv = proj_np(point_rotated, intrinsics)
    noise = np.round(pixel_error*np.random.randn(uv.shape[0],2))
    uv = uv+noise
    mask = np.logical_and(np.logical_and(0<uv[:,0],uv[:,0]<2600),np.logical_and(0<uv[:,1],uv[:,1]<2000))
    mask = np.logical_and(mask, dist_mask)
    return uv, mask
    # if (0<uv[0]<2600 and 0<uv[1]<2000):
    #     #print(point, x_pose)
    #     noise = np.round(pixel_error*np.random.randn(2))
    #     uv = uv+noise
    #     return np.array(uv)
    # else:
    #     return None


def centroid_boxes(boxes):
    boxes_centroids = []
    for box in boxes:
        clon = (box[0]+box[2])/2
        clat = (box[1]+box[3])/2
        boxes_centroids.append([clon,clat])
    return(np.array(boxes_centroids))

def load_boxes():
    landmarks = []
    for fle in glob("landmarks/*_outboxes.npy"):
        landmarks.append(np.load(fle))
    landmarks = np.array(landmarks)
    return landmarks
    
def apply_inverse_pose_transformation_np(point, rotation_quaternion, translation=None):
    """Apply pose transformation (translation and quaternion rotation) on a point."""

    if translation is not None:
        point = point - translation
    # Normalize the quaternion
    #q_norm = rotation_quaternion/rotation_quaternion.norm(dim=-1).unsqueeze(-1)
    q_norm = rotation_quaternion/(np.linalg.norm(rotation_quaternion, axis=-1, keepdims=True))

    # Convert the point to a quaternion representation (w=0)
    #v = torch.cat([point, torch.zeros_like(point[:,:,:1])], dim=-1)
    v = np.stack([point[...,0],point[...,1],point[...,2],np.zeros_like(point[...,0])] , axis=-1)
    # Compute the quaternion conjugate
    q_conj = np.stack([-q_norm[...,0],-q_norm[...,1],-q_norm[...,2], q_norm[...,3]], axis=-1)

    # Apply the quaternion rotation using quaternion multiplication
    v_rotated = quaternion_multiply_np(q_conj, quaternion_multiply_np(v, q_norm))

    return v_rotated
def quaternion_multiply_np(q1, q2):
    """Multiply two quaternions."""
    #print(q1,q2)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w_mul = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_mul = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_mul = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_mul = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([x_mul, y_mul, z_mul, w_mul], axis=-1)
def proj_np(X, intrinsics):
    """ projection """

    X, Y, Z, W = X[..., 0],X[..., 1], X[...,2], X[...,3]
    fx, fy, cx, cy = intrinsics[0],intrinsics[1], intrinsics[2], intrinsics[3] #[...,None,None]

    d = 1.0 / np.clip(Z,0.1,None)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    return np.stack([x, y], axis=-1)

noisy_detections()