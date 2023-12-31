import torch
# from lietorch.groups import SO3, SE3
import numpy as np
from BA.utils import *
from scipy.spatial import transform
from torch_scatter import scatter_sum

def proj(X, intrinsics):
    """ projection """

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics.unbind(dim=2) #[...,None,None]

    d = 1.0 / Z.clamp(min=0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy
    # ipdb.set_trace()

    return torch.stack([x, y], dim=-1)

def attitude_jacobian(q):
    """ attitude jacobian """
    q1, q2, q3, q0 = q.unbind(dim=-1)
    Gq = torch.stack([
        torch.stack([q0, -q3, q2], dim=-1),
        torch.stack([q3, q0, -q1], dim=-1),
        torch.stack([-q2, q1, q0], dim=-1),  
        torch.stack([-q1, -q2, -q3], dim=-1),     
    ], dim=-2)
    return Gq

def landmark_project(poses, landmarks_xyz, intrinsics, ii, jacobian=True):
    # ipdb.set_trace()
    # landmark_est = torch.zeros_like(landmarks_xyz[:, :, :2])
    # Jg = torch.zeros(landmarks_xyz.shape[0], landmarks_xyz.shape[1], 3*poses.shape[1])
    poses_ii = poses[:, ii, :]
    bsz = poses_ii.shape[0]
    poses_ii = poses_ii.reshape(-1, 7).requires_grad_(True)
    def project_ii(poses_ii):
        poses_ii = poses_ii.reshape(bsz, -1, 7)
        X1 = apply_inverse_pose_transformation(landmarks_xyz, poses_ii[:, :, 3:], poses_ii[:, :, :3])
        X1 = torch.stack([X1[...,0], X1[...,1], X1[...,2], torch.ones_like(X1[...,1])], dim=-1)
        landmark_est = proj(X1, intrinsics[:, ii])
        landmark_est = landmark_est.reshape(-1, 2)
        return landmark_est
    def project_ii_sum(poses_ii):
        return project_ii(poses_ii).sum(dim=0)
    # X1 = apply_inverse_pose_transformation(landmarks_xyz, poses_ii[:, :, 3:], poses_ii[:, :, :3])
    # X1 = torch.stack([X1[...,0], X1[...,1], X1[...,2], torch.ones_like(X1[...,1])], dim=-1)
    # landmark_est = proj(X1, intrinsics[:, ii])
    landmark_est = project_ii(poses_ii).reshape(bsz, -1, 2)
    # ipdb.set_trace()
    if jacobian:
        Gq = attitude_jacobian(poses_ii[:,3:])
        # Jg = torch.autograd.functional.jacobian(landmark_est.reshape(-1, 2), poses_ii.reshape(-1,7))
        Jg = torch.autograd.functional.jacobian(project_ii_sum, poses_ii, vectorize=True).transpose(0,1)
        ipdb.set_trace()
        Jg = torch.cat([Jg[:,:,:3], torch.bmm(Jg[:,:,3:], Gq)], dim=2).reshape(-1, 2, 6)
        return landmark_est, Jg
    return landmark_est

def predict(poses, velocities, imu_meas, dt=1, jacobian=True):
    w, a = imu_meas[..., :3], imu_meas[..., 3:]
    # phi = quaternion_log(poses[:,:,3:])
    # position = poses[:,:,:3]
    # rotation = poses[:,:,3:]
    # vel = velocities + dt * (apply_pose_transformation_quat(a.unsqueeze(-1), rotation)).squeeze(-1) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
    # pos_pred = position + dt * vel
    # phi_pred = phi + dt * w
    # q_pred = quaternion_exp(phi_pred)#.data  # TODO: why .data?
    # pose_pred = torch.cat([pos_pred, quaternion_exp(phi_pred)], 2) # TODO: why .data? for quaternion_exp().data
    # vel_pred = vel
    # res_pred = torch.cat([pos_pred[:,:-1] - position[:,1:], 1 - torch.abs(q_pred*rotation[:,1:])], 2)
    bsz = poses.shape[0]
    N = poses.shape[1]
    def res_preds(poses):
        phi = quaternion_log(poses[:,:,3:])
        position = poses[:,:,:3]
        rotation = poses[:,:,3:]
        vel = velocities + dt * a#apply_pose_transformation_quat(a, rotation) # SO3(rotation).act(a.unsqueeze(-1)).squeeze(-1)
        pos_pred = position + dt * velocities
        phi_pred = phi + dt * w
        q_pred = quaternion_exp(phi_pred)#.data  # TODO: why .data?
        pose_pred = torch.cat([pos_pred, quaternion_exp(phi_pred)], 2) # TODO: why .data? for quaternion_exp().data
        vel_pred = vel
        res_pred = torch.cat([pos_pred[:,:-1] - position[:,1:], 1 - torch.abs(q_pred[:,:-1]*rotation[:,1:]).sum(dim=-1).unsqueeze(-1)], 2)
        # ipdb.set_trace()
        return res_pred, pose_pred, vel_pred
    def res_preds_sum(poses):
        poses = poses.reshape(bsz, -1, 7)
        return res_preds(poses)[0].sum(dim=0).reshape(-1)
    res_pred, pose_pred, vel_pred = res_preds(poses)
    if jacobian:
        # Jf = torch.zeros(4*poses.shape[0], 3*poses.shape[1])
        # Jf[0::4, 0::3] = torch.eye(poses.shape[0])
        Gq = attitude_jacobian(poses[:,:,3:])
        Jf = torch.autograd.functional.jacobian(res_preds_sum, poses.reshape(bsz, -1)).reshape(bsz, -1, 7)
        Gq = Gq[:, None].repeat(bsz, (N-1)*4, 1, 1, 1).reshape(bsz, -1, 4, 3)
        Jf = torch.cat([Jf[:,:,:3], (Jf[:,:,3:,None] * Gq).sum(dim=2)], dim=2)
        Jf = Jf.reshape(bsz, (N-1)*4, N*6)
        # Jf = Jf.reshape(bsz, (N-1), 4, N, 6)[:, :, :3, :, :3].reshape(bsz, (N-1)*3, N*3)
        return res_pred, pose_pred, vel_pred, 0, 0, Jf
    
    return res_pred, pose_pred, vel_pred

def get_r_sun_moon_PN(r_suns, r_moons, PNs, h, t):
    idx = ((2*t)//h).long()
    r_sun = r_suns[idx]
    r_moon = r_moons[idx]
    PN = PNs[idx]
    return r_sun, r_moon, PN

#Some constants
RE = 6378.0 #Radius of the Earth (km)
μ = 398600.0 #Standard gravitational parameter of Earth


def ground_truth_sat_dynamics(x, t, params):
    
    r = x[:,:3] #satellite position in inertial frame
    v = x[:,3:] #satellite velocity in inertial frame

    r_moons, r_suns, PNs, h = params

    r_sun, r_moon, PN = get_r_sun_moon_PN(r_suns, r_moons, PNs, h, t)
        
    # #look up this term. seems to give a rotation matrix
    # PN = bias_precession_nutation(epc)
    
    # #Compute the sun and moon positions in ECI frame
    # r_sun = sun_position(epc)
    # r_moon = moon_position(epc)
    
    #define the acceleration variable
    a = torch.zeros((x.shape[0],3)).to(x)
    
    #compute acceleration caused by Earth gravity (includes J2)
    #modeled by a spherical harmonic gravity field
    #look up this term. seems to give a rotation matrix
    # PN = bias_precession_nutation(epc)
    # Earth_r    = earth_rotation(epc)
    # rpm  = polar_motion(epc) 

    # R = rpm*Earth_r*PN
    # n_grav = 10
    # m_grav = 10
    # #main contribution in acceleration (seemed to not be equal to the Series Expansion of gravity)
    # a+= accel_gravity(x, R, n_grav, m_grav)
    
    
    #this is the gravity code that is working
    ###########################################################################################################
    #compute the gravitational acceleration based off the series expansion up to J2
    mu = 3.986004418e14 #m3/s2
    J2 = 1.08264e-3 
        
    a_2bp = (-mu*r)/(r.norm(dim=-1)**3).unsqueeze(-1)
    
    Iz = torch.tensor([0,0,1]).to(x).unsqueeze(0)
    
    a_J2 = ((3*mu*J2*R_EARTH**2)/(2*(r.norm(dim=-1)**5))).unsqueeze(-1)*((((5*((r*Iz).sum(dim=-1)**2))/r.norm(dim=-1)**2)-1).unsqueeze(-1)*r - 2*(r*Iz).sum(dim=-1).unsqueeze(-1)*Iz)     

    a_grav = a_2bp + a_J2
    
    a += a_grav
    ############################################################################################################
    
    #atmospheric drag
    #compute the atmospheric density from density harris priester model
    rho = density_harris_priester(r, r_sun)
    #ρ = 1.15e-12 #fixed atmospheric density in kg/m3

    
    #computes acceleration due to drag in inertial directions
    cd = 2.0 #drag coefficient
    area_drag = 0.1 #in m2 #area normal to the velocity direction
    m = 1.0
    
    
    a_drag = accel_drag(x, rho, m, area_drag, cd, PN)

    a += a_drag  #accel_drag(x, rho, m, area_drag, cd, PN)
    
    #Solar Radiation Pressure
    area_srp = 1.0
    coef_srp = 1.8
    a_srp = accel_srp(x, r_sun, m, area_srp, coef_srp)
    a += a_srp #accel_srp(x, r_sun, m, area_srp, coef_srp)
    
    
    #acceleration due to external bodies
    a_sun = accel_thirdbody_sun(x, r_sun)
    a+= a_sun#accel_thirdbody_sun(x, r_sun)
    
    #COMMENTED FOR TESTING
    a_moon = accel_thirdbody_moon(x, r_moon)
    a+= a_moon #accel_thirdbody_moon(x, r_moon)
    
    a_unmodeled = a_srp + a_sun + a_moon
            
    xdot = x[:,3:6]
    vdot = a
    
    x_dot = torch.cat([xdot, vdot], dim=-1)
    
    # return x_dot, a_unmodeled, rho, 
    return x_dot, a


rho_max = 5e-11 #in kg/m3
rho_min = 2e-14 #in kg/m3


def RK4(x, t, h, params):
    
    f1, _ = ground_truth_sat_dynamics(x, t, params) 
    f2, _ = ground_truth_sat_dynamics(x+0.5*h*f1, t+h/2, params)
    f3, _ = ground_truth_sat_dynamics(x+0.5*h*f2, t+h/2, params)
    f4, _ = ground_truth_sat_dynamics(x+h*f3, t+h, params)
    
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
        
    return xnext
     
def RK4_avg(x, t, h, params):
    f1, _ = ground_truth_sat_dynamics(x, t, params) 
    f2, _ = ground_truth_sat_dynamics(x+0.5*h*f1, t+h/2, params)
    f3, _ = ground_truth_sat_dynamics(x+0.5*h*f2, t+h/2, params)
    f4, _ = ground_truth_sat_dynamics(x+h*f3, t+h, params)
    
    favg = (1/6.0)*(f1+2*f2+2*f3+f4)
        
    return favg[:,3:6]
     

def get_all_r_sun_moon_PN():
    h = 1 #1 Hz the timestep
    # from julia import Main
    # import julia
    # jl = julia.Julia(compile_modules=False)
    # Main.include("BA/julia_utils.jl")
    # # initial time for sim
    # # epc0 = Epoch(2012, 11, 8, 12, 0, 0, 0.0)
    # # r_moons, r_suns, PNs = jl.eval("get_r_moon_sun_PNs()")
    # r_moons, r_suns, PNs = Main.get_r_moon_sun_PNs()
    # r_moons, r_suns, PNs = r_moons.transpose(), r_suns.transpose(), PNs.transpose(2,0,1)

    # np.save("data/r_moons.npy", r_moons)
    # np.save("data/r_suns.npy", r_suns)
    # np.save("data/PNs.npy", PNs)
    r_moons = np.load("data/r_moons.npy")
    r_suns = np.load("data/r_suns.npy")
    PNs = np.load("data/PNs.npy")
    ipdb.set_trace()
    r_moons, r_suns, PNs = torch.tensor(r_moons), torch.tensor(r_suns), torch.tensor(PNs)
    t = torch.arange(0, len(r_moons)/2)
    params = (r_moons, r_suns, PNs, h, t)
    return params

def quaternion_log(q):
    """
    Calculate the logarithm of a quaternion.
    
    Args:
        q: Input quaternion [x, y, z, w].
        
    Returns:
        d_theta : The quaternion logarithm
    """

    # q = (q/np.linalg.norm(q)).clamp(-1,1)
    q = (q/q.norm(dim=-1).unsqueeze(-1)).clamp(-1,1)
    # norm = q[:, :-1].norm(dim=-1)
    # theta = 2 * torch.atan2(norm, q[:, -1])
    # return q[:, :-1]*(theta/norm).unsqueeze(-1)
    theta = 2 * torch.arccos(q[..., -1])
    n = q[..., :-1] / torch.sin(theta/2).unsqueeze(-1)
    return n * theta.unsqueeze(-1)


def quaternion_exp(d_theta):
    """
    Calculate the exponential of a small change in angle (d_theta).
    
    Args:
        d_theta (np.ndarray): The small change in angle [d_theta_x, d_theta_y, d_theta_z].
        
    Returns:
        np.ndarray: The resulting quaternion [w, x, y, z].
    """
    theta = d_theta.norm(dim=-1).unsqueeze(-1)
    mask = (theta < 1e-6).float()
    Identity = torch.cat([torch.zeros_like(d_theta), torch.ones_like(theta)], dim=-1)  # Identity quaternion when theta is close to zero.
    q = torch.cat([d_theta * torch.sin(theta / 2) / (theta + 1e-6), torch.cos(theta / 2)], dim=-1)
    q = Identity*mask + q*(1-mask)
    return q

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    # return np.array([q[0], -q[1], -q[2], -q[3]])
    return torch.cat([-q[..., :-1], q[..., -1:]], dim=-1)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    w_mul = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_mul = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_mul = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_mul = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x_mul, y_mul, z_mul, w_mul], dim=-1)

def quaternion_multiply_np(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = np.split(q1, q1.shape[-1], axis=-1)
    x2, y2, z2, w2 = np.split(q2, q2.shape[-1], axis=-1)
    w_mul = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x_mul = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y_mul = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z_mul = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([x_mul, y_mul, z_mul, w_mul], axis=-1)

def apply_pose_transformation_quat(point, rotation_quaternion, translation=None):
    """Apply pose transformation (translation and quaternion rotation) on a point."""
    # Normalize the quaternion
    q_norm = rotation_quaternion/rotation_quaternion.norm(dim=-1).unsqueeze(-1)

    # Convert the point to a quaternion representation (w=0)
    v = torch.cat([point, torch.zeros_like(point[:,:,:1])], dim=-1)

    # Compute the quaternion conjugate
    q_conj = quaternion_conjugate(q_norm)

    # Apply the quaternion rotation using quaternion multiplication
    v_rotated = quaternion_multiply(q_norm, quaternion_multiply(v, q_conj))

    # Extract the transformed point coordinates
    if translation is not None:
        v_final = v_rotated[:,:,:-1] +  translation
    else:
        v_final = v_rotated[:,:,:-1]

    return v_final

def apply_pose_transformation_quat_np(point, rotation_quaternion, translation=None):
    point = torch.tensor(point).unsqueeze(0)
    rotation_quaternion = torch.tensor(rotation_quaternion).unsqueeze(0)
    if translation is not None:
        translation = torch.tensor(translation).unsqueeze(0)
    point_transformed = apply_pose_transformation_quat(point, rotation_quaternion, translation)
    return point_transformed.squeeze(0).numpy()

def apply_inverse_pose_transformation(point, rotation_quaternion, translation=None):
    """Apply pose transformation (translation and quaternion rotation) on a point."""

    if translation is not None:
        point = point - translation
    # Normalize the quaternion
    q_norm = rotation_quaternion/rotation_quaternion.norm(dim=-1).unsqueeze(-1)

    # Convert the point to a quaternion representation (w=0)
    v = torch.cat([point, torch.zeros_like(point[:,:,:1])], dim=-1)

    # Compute the quaternion conjugate
    q_conj = quaternion_conjugate(q_norm)

    # Apply the quaternion rotation using quaternion multiplication
    v_rotated = quaternion_multiply(q_conj, quaternion_multiply(v, q_norm))

    return v_rotated


#Implement this function in spherical coordinates
def gravitational_potential_new(s):
    # input: position in spherical coordinates 
    # s = [r, θ, ϕ]
    # output: gravitational potential
    
    #J2 = mu (in km) * radius of Earth^2 (km2)* J2 term
    #Constants
    mu = 3.986004418e5 #km3/s2
    J2 = 1.08264e-3 
    
    # unpack input
    r = s[:, 0]
    theta = s[:, 1]
    
    m = 1.0 #added in
    
    #only a function of the latitude
    U = (mu/r)*(1+((J2*R_EARTH**2)/(2*(r**2)))*(1-3*(torch.sin(theta))**2))
    
    return U.sum()

# conversion from cartesian coordinates to spherical coordinates
def cartesian_to_spherical(x):
    r = x[:,:3].norm(dim=-1) #torch.sqrt(x[1:3]'*x[1:3])
    theta = torch.atan2(x[:, 2],x[:, :2].norm(dim=-1)) #torch.sqrt(x[1:2]'*x[1:2]))
    phi = torch.atan2(x[:,1],x[:,0])
    
    return torch.stack([r, theta, phi], dim=-1)

def gravitational_acceleration(x):
    # input: position in cartesian coordiantes 
    # output: acceleration in cartesian coordiantes 
    
    
    q_c = x[:,:3]
    #q_d = x[7:9]
    
    v_c = x[:,3:6]
    #v_d = x[10:12]
    
    #a_d = x[7:9]
    
    #c_d = 2.0 #drag coefficient (dimensionless)
    
    #A = 0.1 
    
    #rotation of the earth (rad/s)
    #ω_earth = [0,0, OMEGA_EARTH]
    
    #v_rel = v - cross(ω_earth, q)
    
    #f_drag = -0.5*c_d*(A)*ρ*norm(v_rel)*v_rel
    
    # a_c = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_c))#+ a_d
    q_c.requires_grad_(True)
    a_c = torch.autograd.grad(gravitational_potential_new(cartesian_to_spherical(q_c)), q_c, create_graph=True)[0]
    #a_d = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_d))+ a_d
    
    return a_c

def orbit_dynamics(x):
    
    q_c = x[:, :3]
    #q_d = x[7:9]
    
    v_c = x[:, 3:6]
    #v_d = x[10:12]
    
    a = gravitational_acceleration(x) #obtain the gravitational acceleration given the position q
    
    x_dot = torch.cat([v_c, a[:,:3]], dim=-1)#; zeros(3)] #x dot is velocity and acceleration
    
    return x_dot


def RK4_satellite_potential(x,h):

    #h = 1.0 #time step
    f1 = orbit_dynamics(x)
    f2 = orbit_dynamics(x+0.5*h*f1)
    f3 = orbit_dynamics(x+0.5*h*f2)
    f4 = orbit_dynamics(x+h*f3)
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
    return xnext
    
     
def RK4_orbit_dynamics_avg(x, h):
    f1 = orbit_dynamics(x) 
    f2 = orbit_dynamics(x+0.5*h*f1)
    f3 = orbit_dynamics(x+0.5*h*f2)
    f4 = orbit_dynamics(x+h*f3)
    
    favg = (1/6.0)*(f1+2*f2+2*f3+f4)
        
    return favg[:,3:6]
     


# gmst_deg = 100.0
theta_G0_deg = 280.16  # GMST at J2000.0 epoch in degrees
omega_earth_deg_per_sec = 360 / 86164.100352  # Earth's average rotational velocity in degrees per second



# Constants for the Earth's shape
a = 6378.137  # Earth's equatorial radius in kilometers
b = 6356.752  # Earth's polar radius in kilometers
e = np.sqrt(1 - (b**2 / a**2))

def deg_to_rad(deg):
    return np.deg2rad(deg)

def ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst):
    # Step 1: Convert ECEF to ECI coordinates
    theta = gmst #+ deg_to_rad(90.0)  # Convert GMST to radians and add 90 degrees

    x_eci = x_ecef * np.cos(theta) - y_ecef * np.sin(theta)
    y_eci = x_ecef * np.sin(theta) + y_ecef * np.cos(theta)
    z_eci = z_ecef

    return x_eci, y_eci, z_eci

def get_Rz(times):
    theta_G_rad = np.deg2rad(theta_G0_deg + omega_earth_deg_per_sec * times)
    ZERO = np.zeros_like(theta_G_rad)
    ONE = np.ones_like(theta_G_rad)

    # Rotation matrix
    Rz = np.stack([
        np.stack([np.cos(theta_G_rad), np.sin(theta_G_rad), ZERO],axis=-1),
        np.stack([-np.sin(theta_G_rad), np.cos(theta_G_rad), ZERO],axis=-1),
        np.stack([ZERO, ZERO, ONE], axis=-1)
    ], axis=-2)
    return Rz

def eci_to_ecef(r_eci, times):

    # Rotation matrix
    Rz = get_Rz(times)

    # Convert ECI to ECEF
    r_ecef = Rz * r_eci[:, None, :]

    return r_ecef
    # return x_eci, y_eci, z_eci

def geodetic_to_ecef(latitude, longitude, altitude):
    # Step 1: Convert latitude and longitude to geocentric latitude
    phi = deg_to_rad(latitude)
    lambda_ = deg_to_rad(longitude)

    # Step 2: Calculate the Earth's radius of curvature in the prime vertical
    N = a / np.sqrt(1 - (e**2 * np.sin(phi)**2))
    # N = a 

    # Step 3: Convert latitude, longitude, altitude to ECEF coordinates
    x_ecef = (N + altitude) * np.cos(phi) * np.cos(lambda_)
    y_ecef = (N + altitude) * np.cos(phi) * np.sin(lambda_)
    # z_ecef = ((b**2 / a**2) * N + altitude) * torch.sin(phi)
    z_ecef = (N + altitude) * np.sin(phi)

    return x_ecef, y_ecef, z_ecef

def convert_latlong_to_cartesian(lat, long, times, altitude=None):
    if altitude is None:
        altitude = np.zeros(lat.shape[0])

    # ipdb.set_trace()
    latitude, longitude = lat, long
    # Step 1: Convert latitude, longitude, altitude to ECEF coordinates
    x_ecef, y_ecef, z_ecef = geodetic_to_ecef(latitude, longitude, altitude)

    gmst_deg = theta_G0_deg + omega_earth_deg_per_sec * times
    # Step 2: Convert ECEF to ECI coordinates
    x_eci, y_eci, z_eci = ecef_to_eci(x_ecef, y_ecef, z_ecef, gmst_deg)

    return np.stack([x_eci, y_eci, z_eci], axis=-1)
    # return np.stack([x_ecef, y_ecef, z_ecef], axis=-1)


def unit_vector_to_quaternion(unit_vector1, unit_vector2):
    # Step 1: Calculate the rotation axis
    rotation_axis = np.cross(unit_vector1, unit_vector2, axis=-1)
    rotation_axis /= np.linalg.norm(rotation_axis, axis=-1)[..., None]
    
    # Step 2: Calculate the rotation angle
    rotation_angle = np.arccos(np.sum(unit_vector1*unit_vector2, axis=-1))

    # Step 3: Create the quaternion
    half_rotation_angle = 0.5 * rotation_angle
    sin_half_angle = np.sin(half_rotation_angle)

    quaternion = np.stack([
        sin_half_angle * rotation_axis[:, 0],
        sin_half_angle * rotation_axis[:,1],
        sin_half_angle * rotation_axis[:,2],
        np.cos(half_rotation_angle)
    ], axis=-1)

    return quaternion

def convert_pos_to_quaternion(pos_eci):
    # Step 1: Calculate the satellite's direction vector
    zc = direction_vector = - pos_eci / (np.linalg.norm(pos_eci, axis=-1)[..., None])

    # Step 2: Calculate the quaternion orientation
    # Compute the angle between the satellite's local Z-axis and the ECI Z-axis
    north_pole_eci = np.array([0, 0, 1])[None]
    axis_of_rotation_z = np.cross(north_pole_eci, direction_vector)
    rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z, axis=-1)[..., None]
    xc = -rc

    # compute the vector pointing to the north from camera
    yc = south_vector = np.cross(rc, zc)
    R = np.stack([xc, yc, zc], axis=-1)
    rot = transform.Rotation.from_matrix(R)
    quaternion = rot.as_quat()
    return quaternion

def convert_quaternion_to_xyz_orientation(quat, times):
    # Step 1: convert quat to rotation matrix
    rot = transform.Rotation.from_quat(quat)
    R = rot.as_matrix()

    # Step 2: comvert to ECEF from ECI
    Rz = get_Rz(times)
    R = Rz @ R


    # Step 3: compute the x, y, z axis
    xc, yc, zc = R[:, 0], R[:, 1], R[:, 2]
    right_vector = xc
    up_vector = -yc
    forward_vector = zc


    # zc = direction_vector = - pos_eci / (np.linalg.norm(pos_eci, axis=-1)[..., None])

    # # Step 2: Calculate the quaternion orientation
    # # Compute the angle between the satellite's local Z-axis and the ECI Z-axis
    # north_pole_eci = np.array([0, 0, 1])[None]
    # axis_of_rotation_z = np.cross(north_pole_eci, direction_vector)
    # rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z, axis=-1)[..., None]
    # xc = -rc

    # # compute the vector pointing to the north from camera
    # yc = south_vector = np.cross(rc, zc)
    # R = np.stack([xc, yc, zc], axis=-1)
    # rot = transform.Rotation.from_matrix(R)
    # quaternion = rot.as_quat()
    return forward_vector, up_vector, right_vector 


def compute_omega_from_quat(quat, dt):
    phis = quaternion_log(quat)
    omega = (phis[1:] - phis[:-1]) / dt
    omega = torch.cat([omega, torch.zeros((1, 3))], dim=0)
    return omega

def compute_velocity_from_pos(gt_pos_eci, dt):
    # ipdb.set_trace()
    gt_vel_eci = (gt_pos_eci[1:] - gt_pos_eci[:-1]) / dt
    gt_vel_eci = np.concatenate([gt_vel_eci, np.zeros((1, 3))], axis=0)
    return gt_vel_eci

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    return scatter_sum(A, ii*m + jj, dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    return scatter_sum(b, ii, dim=1, dim_size=n)