import torch
# from lietorch.groups import SO3, SE3
from BA.BA_utils import *

### Bundle Adjustment
# TODOs
# 0. Use orbital dynamics to obtain acceleration but use angular velocity from IMU
# 1. Obtain reasonable values for each of the inputs by going through Paulo's code
# 2. Write testing functions
# 3. Try out different values for the V and Sigma
# 4. Integrate with the DL detection stack
# 5. Implement differentiability for the BA function

def BA(iter, poses, velocities, imu_meas, landmarks, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci):
	poses = poses.double()
	# ipdb.set_trace()
	v = velocities.double()
	imu_meas = imu_meas.double()
	landmarks = landmarks.double()
	landmarks_xyz = landmarks_xyz.double()
	intrinsics = intrinsics.double()
	# time_idx = time_idx.double()
	quat_coeff = 1#00 + min(iter*10, 900)

	bsz = poses.shape[0]
	landmark_est, Jg = landmark_project(poses, landmarks_xyz, intrinsics, ii, jacobian=True)
	r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq = predict(poses, velocities, imu_meas, time_idx, quat_coeff, jacobian=True) 
	r_obs = (landmarks - landmark_est)
	alpha = max(1 - (2*(iter/5) - 1), -4)
	c_obs = r_obs.abs().median()
	wts_obs = (((((r_obs/c_obs)**2)/abs(alpha-2) + 1)**(alpha/2 - 1)) / ((c_obs)**2)).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)[0]
	# ipdb.set_trace()
	wts_obs = (wts_obs/wts_obs.max())*confidences.unsqueeze(-1).unsqueeze(-1)#*0 + 1
	# ipdb.set_trace()
	# r_full = torch.cat([r_obs, r_pred], dim = 1)
	Sigma = 1000#*(iter+1)*4#1#00
	V = 1
	dim_base = 6
	dim = 6
	Jg = Jg.reshape(bsz, -1, dim_base)[:, :, :dim].reshape(-1, 2, dim)
	
	JgTwJg = torch.bmm((Jg*wts_obs).transpose(1,2), Jg).reshape(bsz, -1, dim, dim)#*0
	# wJiT = (Sigma * Ji).transpose(2,3)
	# wJi_1T = (Sigma * Ji_1).transpose(2,3)
	# Bii = torch.matmul(wJiT, Ji)
	# Bij = torch.matmul(wJiT, Ji_1)
	# Bji = torch.matmul(wJi_1T, Ji)
	# Bjj = torch.matmul(wJi_1T, Ji_1)

	# scatter sum of Bii, Bij, Bji, Bjj and JgTwJg
	n = poses.shape[1]
	# i_1 = torch.arange(0, n-1).unsqueeze(0)
	# i = i_1 + 1
	# JfTwJf = safe_scatter_add_mat(Bii, i, i, n, n).view(bsz, n, n, 6, 6) + \
    #     safe_scatter_add_mat(Bij, i, i_1, n, n).view(bsz, n, n, 6, 6) + \
    #     safe_scatter_add_mat(Bji, i_1, i, n, n).view(bsz, n, n, 6, 6) + \
    #     safe_scatter_add_mat(Bjj, i_1, i_1, n, n).view(bsz, n, n, 6, 6)
	# JfTwJf = JfTwJf.transpose(2,3).reshape(bsz, n*6, n*6)
	# # JgTwJg = JfTwJf.transpose(2,3).reshape(bsz, n*6, n*6)
	# ipdb.set_trace()
	ii_t = torch.tensor(ii, dtype=torch.long, device=poses.device)
	JgTwJg = safe_scatter_add_vec(JgTwJg, ii_t, n).view(bsz, n, dim, dim)
	JgTwJg = torch.block_diag(*JgTwJg[0].unbind(dim=0)).unsqueeze(0)
	# JfTwJf = scatter_sum(B, i*n + i_1, dim=1, dim_size=n*n) 

	# ipdb.set_trace()
	dim2 = min(4, dim)
	Jf = Jf.view(bsz, (n-1)*dim2, n*dim)
	Jf_x = Jf.view(bsz, n-1, dim2, n*dim)[:, :, :3, :]
	# ipdb.set_trace()
	Jf_x = Jf_x.reshape((bsz, -1, n*dim))
	JfTwJf = torch.bmm((Jf*Sigma).transpose(1,2), Jf)
	Hq = Sigma*torch.block_diag(*Hq[0].unbind(dim=0)).unsqueeze(0)*0

	# J_full = torch.cat([Jg, Jf], axis=1)
	# wts = torch.cat([V, Sigma], 1).unsqueeze(-1)

	# JTwJ = torch.bmm((J_full*wts).transpose(1,2), J_full)

	# JTr = torch.bmm((J_full*wts).transpose(1,2), r_full.unsqueeze(-1))

	# ipdb.set_trace()	
	r_pred = r_pred[:, :,  :dim]
	JgT_robs = safe_scatter_add_vec((Jg.reshape(bsz, -1, 2, dim)*wts_obs[None]*r_obs.unsqueeze(-1)).sum(dim=-2), ii_t, n).view(bsz, n,dim)
	# ipdb.set_trace()
	r_pred_x = r_pred.clone()
	r_pred_x[:, :, 3:] = 1
	JfT_rpred = (r_pred.reshape(bsz, -1, 1) * Sigma * Jf).sum(dim=1).reshape(bsz, n, dim)*(-1)#.01
	JTr = (JgT_robs + JfT_rpred).reshape(bsz, -1)#+ JfT_rpred

	lamda = lamda_init
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).norm()
	while True:

		JTwJ =  torch.eye(n*dim)[None]*lamda+JgTwJg + JfTwJf#*100#.01 #+ torch.eye(n*dim)[None]*1e-5+#.1 # 
		# JTwJ.view(bsz, n, dim, n, dim)[:, :, 3:, :, 3:] += Hq.view(bsz, n, 3, n, 3)
		try:
			dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)
		
			# ipdb.set_trace()
			position = poses[:,:,:3] + dpose[:,:,:3]
			rotation = quaternion_multiply(poses[:,:,3:], quaternion_exp(dpose[:,:,3:]))
			rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
			# rotation[:, 0] = poses[:, 0, 3:].clone()
			poses_new = torch.cat([position, rotation], 2)
			landmark_est = landmark_project(poses_new, landmarks_xyz, intrinsics, ii, jacobian=False)
			r_pred1, _, _ = predict(poses_new, velocities, imu_meas, time_idx, quat_coeff, jacobian=False) 
			r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
			r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)#*0.01
			r_obs1 = r_obs1.reshape(bsz, -1)
			residual = torch.cat([r_obs1, r_pred1], dim = 1)
			print("lamda: ", lamda, r_pred1.abs().mean(), r_obs1.abs().mean())
		except:
			lamda = lamda*10
			continue

		lamda = lamda*10
		if (residual.norm()) < init_residual:
			break
		if lamda > 1e4:
			print("lamda too large")
			break
		
	lamda_init = max(min(1e-1, lamda*0.01), lamda_init*0.1)

	## backtracking line search
	alpha = 1
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).norm()
	# init_residual = (r_obs.abs()).sum() + (r_pred.abs()).sum()*np.sqrt(Sigma)
	print("alpha: ", alpha, r_pred.abs().mean()*np.sqrt(Sigma), r_obs.abs().mean())
	# while True:
	# 	position = poses[:,:,:3] + alpha*dpose[:,:,:3]
	# 	rotation = quaternion_multiply(poses[:,:,3:], quaternion_exp(alpha*dpose[:,:,3:]))
	# 	rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
	# 	poses_new = torch.cat([position, rotation], 2)
	# 	landmark_est = landmark_project(poses_new, landmarks_xyz, intrinsics, ii, jacobian=False)
	# 	r_pred1, _, _ = predict(poses_new, velocities, imu_meas, time_idx, quat_coeff, jacobian=False) 
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
		

	# position = poses[:,:,:3] + dpose[:,:,:3]
	# rotation = quaternion_multiply(poses[:,:,3:], quaternion_exp(dpose[:,:,3:]))
	# # ipdb.set_trace()
	# # rotation = rotation[:, 1:]
	# # rotation = poses[:,1:,3:] + dpose[:,1:,3:]
	# rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
	# poses_new = torch.cat([position, rotation], 2)
	# # poses_new = torch.cat([poses[:,:1], poses_new], 1)
	# # print("final: ", (poses_new[0,:3, :3]-poses_gt_eci[:3, :3]).abs().mean(dim=0))
	# # print("init: ", (poses[0,:3, :3] - poses_gt_eci[:3, :3]).abs().mean(dim=0))
	# # print("r_pred :", r_pred[0,:3].abs().mean(dim=0))
	# # print("r_obs :", r_obs[0,:3].abs().mean(dim=0))

	print("final quat: ", (poses_new[0,:, 3:]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((poses_new[0,:, 3:] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	# print("final: ", (poses_new2[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
	# print("final: ", (poses_new3[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
	print("init quat: ", (poses[0,:, 3:] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((poses[0,:, 3:] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	print("final pos: ", (poses_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0))
	print("init pos: ", (poses[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
	print("r_pred :", r_pred[0,:].abs().mean(dim=0))
	print("r_obs :", r_obs[0,:].abs().mean(dim=0))
	# ipdb.set_trace()
	return poses_new, velocities, lamda_init