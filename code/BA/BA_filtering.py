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

def BA(iter, states, velocities, imu_meas, landmarks, landmarks_xyz, ii, time_idx, intrinsics, confidences, Sigma, V, lamda_init, poses_gt_eci):
	states = states.double()
	# ipdb.set_trace()
	v = velocities.double()
	imu_meas = imu_meas.double()
	landmarks = landmarks.double()
	landmarks_xyz = landmarks_xyz.double()
	intrinsics = intrinsics.double()
	# time_idx = time_idx.double()
	quat_coeff = 100 #+ min(iter*10, 900)
	vel_coeff = 100

	bsz = states.shape[0]
	landmark_est, Jg = landmark_project(states, landmarks_xyz, intrinsics, ii, jacobian=True)
	r_pred, pose_pred, vel_pred, Ji, Ji_1, Jf, Hq, qgrad = predict(states, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=True) 
	r_obs = (landmarks - landmark_est)
	alpha = max(1 - (2*(iter/5) - 1), 0)
	c_obs = r_obs.abs().median()#1000
	wts_obs = (((((r_obs/c_obs)**2)/abs(alpha-2) + 1)**(alpha/2 - 1)) / ((c_obs)**2)).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)[0]
	wts_obs = (wts_obs/wts_obs.max())*confidences.unsqueeze(-1).unsqueeze(-1)#*0 + 1
	Sigma = min(100*(iter+1)**2, 10000)#1#00
	V = 1
	dim_base = 9
	dim = 9
	Jg = Jg.reshape(bsz, -1, dim_base)[:, :, :dim].reshape(-1, 2, dim)
	
	JgTwJg = torch.bmm((Jg*wts_obs).transpose(1,2), Jg).reshape(bsz, -1, dim, dim)

	n = states.shape[1]
	ii_t = torch.tensor(ii, dtype=torch.long, device=states.device)
	JgTwJg = safe_scatter_add_vec(JgTwJg, ii_t, n).view(bsz, n, dim, dim)
	JgTwJg = torch.block_diag(*JgTwJg[0].unbind(dim=0)).unsqueeze(0)

	dim2 = min(6, dim)
	Jf = Jf.view(bsz, (n-1)*dim2, n*dim)
	JfTwJf = torch.bmm((Jf*Sigma).transpose(1,2), Jf)

	r_pred = r_pred[:, :,  :dim]
	JgT_robs = safe_scatter_add_vec((Jg.reshape(bsz, -1, 2, dim)*wts_obs[None]*r_obs.unsqueeze(-1)).sum(dim=-2), ii_t, n).view(bsz, n,dim)
	r_pred_x = r_pred[:, :, :6].clone()
	JfT_rpred = (r_pred_x.reshape(bsz, -1, 1) * Sigma * Jf).sum(dim=1).reshape(bsz, n, dim)*(-1)#.01
	JTr = (JgT_robs + JfT_rpred - Sigma*qgrad).reshape(bsz, -1)#+ JfT_rpred

	lamda = lamda_init
	init_residual = torch.cat([r_obs.reshape(-1), r_pred.reshape(-1)*np.sqrt(Sigma)], dim = 0).abs().mean()
	while True:

		JTwJ =  torch.eye(n*dim)[None]*lamda+JgTwJg + JfTwJf + Sigma*Hq#*100#.01 #+ torch.eye(n*dim)[None]*1e-5+#.1 # 
		# try:
		dpose = torch.linalg.solve(JTwJ, JTr).reshape(bsz, n, dim)
	
		position = states[:,:,:3] + dpose[:,:,:3]
		vels = states[:,:,7:] + dpose[:,:,6:]
		rotation = quaternion_multiply(states[:,:,3:7], quaternion_exp(dpose[:,:,3:6]))
		rotation = rotation / torch.norm(rotation, dim=2, keepdim=True)
		states_new = torch.cat([position, rotation, vels], 2)
		landmark_est = landmark_project(states_new, landmarks_xyz, intrinsics, ii, jacobian=False)
		r_pred1, _, _ = predict(states_new, imu_meas, time_idx, quat_coeff, vel_coeff, jacobian=False) 
		r_obs1 = (landmarks - landmark_est)*wts_obs[None, :, 0]
		r_pred1 = r_pred1[:, :,  :dim].reshape(bsz, -1) * np.sqrt(Sigma)#*0.01
		r_obs1 = r_obs1.reshape(bsz, -1)
		residual = torch.cat([r_obs1, r_pred1], dim = 1)
		print("lamda: ", lamda, r_pred1.abs().mean(), r_obs1.abs().mean())
		# except:
		# 	lamda = lamda*10
		# 	continue

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

	print("final quat: ", (states_new[0,:, 3:7]-poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states_new[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	# print("final: ", (poses_new2[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
	# print("final: ", (poses_new3[0,:3, 3:]-poses_gt_eci[:3, 3:]).abs().mean(dim=0))
	print("init quat: ", (states[0,:, 3:7] - poses_gt_eci[:, 3:]).abs().mean(dim=0), (1 - torch.abs((states[0,:, 3:7] * poses_gt_eci[:, 3:]).sum(dim=-1))).mean())
	print("final pos: ", (states_new[0,:, :3]-poses_gt_eci[:, :3]).abs().mean(dim=0))
	print("init pos: ", (states[0,:, :3] - poses_gt_eci[:, :3]).abs().mean(dim=0))
	print("final vels: ", (states_new[0,:, 7:]-velocities[0]).abs().mean(dim=0))
	print("init vels: ", (states[0,:, 7:] - velocities[0]).abs().mean(dim=0))
	print("r_pred :", r_pred[0,:].abs().mean(dim=0), r_pred[0,:].abs().mean(dim=0)[-1].item())
	print("r_obs :", r_obs[0,:].abs().mean(dim=0))
	# ipdb.set_trace()
	return states_new, velocities, lamda_init