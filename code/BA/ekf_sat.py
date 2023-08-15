from utils import *
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt


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
    a = torch.zeros(3).to(x)
    
    #compute acceleration caused by Earth gravity (includes J2)
    #modeled by a spherical harmonic gravity field
    #look up this term. seems to give a rotation matrix
#     PN = bias_precession_nutation(epc)
#     Earth_r    = earth_rotation(epc)
#     rpm  = polar_motion(epc) 

#     R = rpm*Earth_r*PN
#     n_grav = 10
#     m_grav = 10
#     #main contribution in acceleration (seemed to not be equal to the Series Expansion of gravity)
#     a+= accel_gravity(x, R, n_grav, m_grav)
    
    
    #this is the gravity code that is working
    ###########################################################################################################
    #compute the gravitational acceleration based off the series expansion up to J2
    mu = 3.986004418e14 #m3/s2
    J2 = 1.08264e-3 
        
    a_2bp = (-mu*r)/(r.norm(dim=-1)**3)
    
    Iz = torch.tensor([0,0,1]).to(x).unsqueeze(0)
    
    a_J2 = ((3*mu*J2*R_EARTH**2)/(2*(r.norm(dim=-1)**5)))*((((5*((r*Iz).sum(dim=-1)**2))/r.norm(dim=-1)**2)-1)*r - 2*(r*Iz).sum(dim=-1)*Iz)     

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
    
    return x_dot, a_unmodeled, rho, a_drag


rho_max = 5e-11 #in kg/m3
rho_min = 2e-14 #in kg/m3


def RK4(x, t, h, params):
    
    f1,_, _, _ = ground_truth_sat_dynamics(x, t, params) 
    f2,_, _, _ = ground_truth_sat_dynamics(x+0.5*h*f1, t+h/2, params)
    f3,_, _, _ = ground_truth_sat_dynamics(x+0.5*h*f2, t+h/2, params)
    f4,_, _, _ = ground_truth_sat_dynamics(x+h*f3, t+h, params)
    
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
        
    return xnext
     

h = 1 #1 Hz the timestep

from julia import Main
import julia
jl = julia.Julia(compile_modules=False)
Main.include("julia_utils.jl")
# initial time for sim
# epc0 = Epoch(2012, 11, 8, 12, 0, 0, 0.0)
r_moons, r_suns, PNs = jl.eval("get_r_moon_sun_PNs()")
r_moons, r_suns, PNs = Main.get_r_moon_sun_PNs()
params = (r_moons, r_suns, PNs, h)


#Test of some of the SatelliteDynamics functions

#Orbit we want (around ISS orbit conditions)
iss1 = [6871e3, 0.00064, 51.6450, 1.0804, 27.7899, 190]

# Convert osculating elements to Cartesean state
# returns position and velocity (m, m/s). This is the intial position
eci0_1 = sOSCtoCART(iss1, use_degrees=True)

#find the period of the orbit (seconds). only dependent on semi major axis
T = orbit_period(iss1[1])

#final time of simulation
# epcf = epc0 + T



x_0 = eci0_1

#x_0d = zeros(size(x_0)[1])
#x_0d[1:3] = x_0[1:3]+1000*randn(3)
#x_0d[4:6] = x_0[4:6]+10*randn(3)
# display(x_0)
#display(x_0d)
print(x_0)

#number of orbits to simulate
orbit_num = 0.1

#final time for the simulation (period x number of orbits)
Tf = T*orbit_num

#run the rk4
t = np.arange(0,Tf, step=h) #create a range to final time at constant time step
    
# all_x_c = torch.zeros((x_0.shape[1], len(t))) #variable to store all x of chief
#all_x_d = zeros(length(x_0), length(t)) #variable to store all x of deputy
    
all_x_c = [x_0] #set the initial state of chief
#all_x_d[:,1] = x_0d #set the initial state of deputy

for k in range(len(t) - 1):
            
    current_t = t[k]#epc0+t[k] #calculate the current time
        
    all_x_c.append(RK4(all_x_c[-1], current_t, h, params)) #calculate the next state for chief
    #all_x_d[:,k+1] = RK4(all_x_d[:,k], current_t, h) #calculate the next state for chief

#contains all the ground truth states
x_hist = torch.stack(all_x_c, dim=-1)

x_hist_scaled = x_hist*1e-3 # get the measurement to km (if needed)

#plot(x_hist[1,:], x_hist[2,:], x_hist[3,:], title="Ground Truth", label="satellite trajectory")
# t = Array(range(0,Tf, step=h)) 

all_a_unmodeled = []#torch.zeros((3, len(t)))

all_drag = []#torch.zeros((3, len(t)))

rho_all = []#torch.zeros(len(t))

for i in range(len(t)):
    
    current_time = t[i] # + epc0
    
    ẋ, a_unmodeled, rho_t, a_drag = ground_truth_sat_dynamics(x_hist[:,i], current_time, params)
    
    rho_all.append(rho_t)
    all_a_unmodeled.append(a_unmodeled)
    all_drag.append(a_drag)
 
rho_all = torch.stack(rho_all, dim=-1)
all_a_unmodeled = torch.stack(all_a_unmodeled, dim=-1)
all_drag = torch.stack(all_drag, dim=-1)

#Ground truth data that concatenates satellite pose with the truth drag force using SatelliteDynamics.jl
# truth_data = [x_hist; all_drag]

#standard deviation associated to the measurement
std_gps_measurement = 1 #*1e-3 #in km. ~10 m

#assume that it is additive noise (measurement noise)
R = np.eye(3)*((std_gps_measurement)**2)/3
R_sqrt = torch.tensor(scipy.linalg.sqrtm(R).real).unsqueeze(0)

# R = convert(Matrix{Float64}, R)

## Print this?
# sqrt(R)*randn(3)
     

#standard deviation associated to the measurement
std_cv_measurement = 360 #*1e-3 #in km. ~10 m

#assume that it is additive noise (measurement noise)
R_cv = np.eye(6)*((std_cv_measurement)**2)/3
R_cv_sqrt = torch.tensor(scipy.linalg.sqrtm(R_cv).real).unsqueeze(0)
bsz = x_hist.shape[0]

# R_cv = convert(Matrix{Float64}, R_cv)

## Print this?
# sqrt(R_cv)*randn(6)
     

GPS_num = x_hist_scaled.shape[2]

# GPS_measurements = []#torch.zeros((3, GPS_num)).to(x_hist)

# for i in range(GPS_num):
    # GPS_measurements.append(x_hist[:, :3, i] + R_sqrt*torch.randn(b,3).unsqueeze(0))
GPS_measurements = x_hist[:, :3] + (R_sqrt.unsqueeze(-1)*torch.randn(bsz,3,GPS_num)[:,None]).sum(dim=2) # b,3,GPS_num
    

#Random positions for the features (trying to make this roughly corespond to points on the surface of the Earth in ECI
z1 = torch.randn((bsz,3))
z1 = RE*z1/(z1.norm(dim=-1).unsqueeze(0))

z2 = z1+(100*torch.randn((bsz,3)))
z2 = RE*z2/(z2.norm(dim=-1).unsqueeze(0))

z_stack = torch.cat([z1,z2], dim=-1)

def cd_measurement_function(x):
    measurement = torch.zeros((x.shape[0], 4))
    measurement[:, :3] = x[:, :3]
    measurement[:, 4] = (x[:,:3] - x[:,6:9]).norm(dim=-1)

    return measurement
    

def cv_measurement_function(x,z):
    #x_pos = x[1:3]
    z1 = z[:,:3]
    z2 = z[:,3:6]
    y1 = z1-x[:,:3]
    y1 = y1/y1.norm(dim=1)

    y2 = z2-x[:,:3]
    y2 = y2/y2.norm(dim=1)
    
    return torch.cat([y1, y2], dim=-1)


CV_num = x_hist_scaled.shape[1]

CV_measurements = torch.zeros((bsz, 6, CV_num))

for i in range(CV_num):
    CV_measurements[:, :, i] = cv_measurement_function(x_hist[:, :3, i],z_stack) + (R_cv_sqrt*torch.randn((1,6))).sum(dim=-1)
         

def cv_measurement_jacobian(x,z):
    x.requires_grad_(True)
    C = torch.autograd.jacobian(cv_measurement_function(x,z),x)[0]
    return C
     
x_test = x_hist[:,:3,0]
y_test = cv_measurement_function(x_test,z_stack)
     

#estimated GPS measurement given the estimated state (deterministic) 
# def measurement_function(x):
    
#     C = torch.tensor([I zeros((3,6))]
#     #measurement only gives you the position of the satellite
#     measurement = C*x 
    
#     #only return the state of the spacecraft
#     return measurement, C

pose_std_dynamics = 4e-6#*1e-3 #get to km
velocity_std_dynamics = 8e-6 #*1e-3 #get to km/s
a_d_noise = 2e-7

#process noise 
# Q = I(9).*[ones(3)*(pose_std_dynamics^2)/3; ones(3)*(velocity_std_dynamics^2)/3; ones(3)*(a_d_noise)^2]
Q = torch.diag([(pose_std_dynamics**2)/3]*3+ [(velocity_std_dynamics**2)/3]*3 +  [(a_d_noise)**2]*3)
Q_sqrt = Q.sqrt().unsqueeze(0)
#jacobian of the measurement function
#H = [I zeros((3,6))]
# def measurement_jacobian(x):
#     H = [I, zeros(3,12)]
#     range_norm = (x[:,:3]-x[6:9]).norm(dim=-1)
#     range_line = [1, 1, 1, -1, -1, -1, zeros(9)']
#     H = [H,range_line/range_norm ]
# MJ = measurement_jacobian(truth_data[:,1])
# MJ[4,:]


#Implement this function in spherical coordinates
def gravitational_potential_new(s):
    # input: position in spherical coordinates 
    # s = [r, θ, ϕ]
    # output: gravitational potential
    
    #J2 = mu (in km) * radius of Earth^2 (km2)* J2 term
    #Constants
    mu = 3.986004418e14 #m3/s2
    J2 = 1.08264e-3 
    
    # unpack input
    r = s[:,0]
    th = s[:,1]
    
    m = 1.0 #added in
    
    #only a function of the latitude
    U = (mu/r)*(1+((J2*R_EARTH**2)/(2*r**2))*(1-3*(torch.sin(th))**2))
    
    return U


# conversion from cartesian coordinates to spherical coordinates
def cartesian_to_spherical(x):
    r = torch.sqrt((x[:,:3]*x[:,:3]).sum(dim=-1))
    th = torch.atan(x[:,2],torch.sqrt((x[:,:2]*x[:,:2]).sum(dim=-1)))
    phi = torch.atan(x[:,1],x[:,0])
    
    return torch.stack([r, th, phi], dim=-1)


def gravitational_acceleration(x):
    # input: position in cartesian coordiantes 
    # output: acceleration in cartesian coordiantes 
    
    
    q_c = x[:,:3]
    #q_d = x[7:9]
    
    v_c = x[:,3:6]
    #v_d = x[10:12]
    
    a_d = x[:,6:9]
    
    #c_d = 2.0 #drag coefficient (dimensionless)
    
    #A = 0.1 
    
    #rotation of the earth (rad/s)
    #ω_earth = [0,0, OMEGA_EARTH]
    
    #v_rel = v - cross(ω_earth, q)
    
    #f_drag = -0.5*c_d*(A)*ρ*norm(v_rel)*v_rel
    
    # a_c = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_c))+ a_d
    a_c = torch.autograd.grad(gravitational_potential_new(cartesian_to_spherical(q_c)), q_c)[0] + a_d
    #a_d = (ForwardDiff.gradient(_x -> gravitational_potential_new(cartesian_to_spherical(_x)), q_d))+ a_d
    
    return a_c
    

def orbit_dynamics(x):
    
    q_c = x[:,:3]
    #q_d = x[7:9]
    
    v_c = x[:,3:6]
    #v_d = x[10:12]
    
    a = gravitational_acceleration(x) #obtain the gravitational acceleration given the position q
    
    xdot = torch.cat([v_c, a[:,:3], torch.zeros((bsz,3))], dim=-1)    #x dot is velocity and acceleration
    
    return xdot
    


def RK4_satellite_potential(x):

    h = 1.0 #time step
    f1 = orbit_dynamics(x)
    f2 = orbit_dynamics(x+0.5*h*f1)
    f3 = orbit_dynamics(x+0.5*h*f2)
    f4 = orbit_dynamics(x+h*f3)
    xnext = x+(h/6.0)*(f1+2*f2+2*f3+f4)
    return xnext
    
    


def find_jacobian_discrete(x_0):
    
    F = torch.autograd.functional.jacobian(RK4_satellite_potential(x_0), x_0)[0]
    
    return F



def find_jacobian_continuous(x):
    
    F = torch.autograd.functional.jacobian(orbit_dynamics(x), x)
    
    return F



#SQRT QR EKF Formulation 
def EKF_satellite_QR(X, F, k):
    
    #Prediction Step
    
    #state prediction
    X_pre = RK4_satellite_potential(X)

    A = find_jacobian_discrete(X) #the A matrix from the dynamics
    
    n = torch.stack([torch.bmm(F,A.transpose(-1,-2)), Q_sqrt], dim=-1)  ########## TODO : Verify concat dimension is correct
    
    _, F_pre = torch.linalg.qr(A)    # qr(n)
        
    zk_hat = cv_measurement_function(X_pre,CV_measurements[:,k]) #get the measurement from the state prediction 
    C = cv_measurement_jacobian(X_pre,CV_measurements[:,k])
    
    #### C IS THE JACOBIAN OF THE MEASUREMENT FUNCTION
        
    innovation = CV_measurements[:, k+1] - zk_hat
    
    _, G = torch.linalg.qr([torch.bmm(F_pre,C.transpose(-1,-2)),R_cv_sqrt]) #mxm where m is measurement size
    
    GtC = torch.linalg.solve(G.transpose(-1,-2),C)  # G'\C
    GtCF = torch.bmm(GtC,torch.bmm(F_pre.transpose(-1,-2),F_pre))
    L_inside = torch.linalg.solve(G,GtCF) # G\(G'\C*F_pre'*F_pre)
    
    L = L_inside.transpose(-1,-2) 
        
    X = X_pre + torch.bmm(L, innovation)

    m = C.shape[-1]
    
    _, F = torch.linalg.qr([torch.bmm(F_pre,(torch.eye(m)[None]-torch.bmm(L,C)).transpose(-1,-2)), torch.bmm(R_cv_sqrt,L.transpose(-1,-2))]) #qr([F_pre*(I-L*C)', R_cv_sqrt*L'])

    #obtain the fisher information matrix
    #Jk1 = inv(Φ)'*J*inv(Φ) + C'*inv(R)*C
    
    #Jk1 = inv(Φ[1:3, 1:3])'*J[1:3, 1:3]*inv(Φ[1:3, 1:3])+ C[1:3, 1:3]'*inv(R[1:3, 1:3])*C[1:3, 1:3]
    
    #Jk1 = inv(R + Φ*(J)*(Φ)')+ C'*inv(Q)*C;
    
    return X, F
    
    

true_first_pose_c = x_0
#true_first_pose_d = x_0d

gps_noise = torch.randn((bsz,3))*100

velocity_noise = torch.randn((bsz,3))*0.1 

std_velocity = 1.0
std_accel = 5e-7

#seems to work
X_0 = torch.stack([true_first_pose_c[:,1]+gps_noise[:,1],true_first_pose_c[:,2]+gps_noise[:,2],true_first_pose_c[:,3]+gps_noise[:,3],true_first_pose_c[:,4]+velocity_noise[:,1],true_first_pose_c[:,5]+velocity_noise[:,2],true_first_pose_c[:,6]+velocity_noise[:,3],torch.ones(bsz)*1e-6, torch.ones(bsz)*2e-6, torch.ones(bsz)*3e-6], dim=-1)
#X_0 = [true_first_pose[1]+gps_noise[1],true_first_pose[2]+gps_noise[2],true_first_pose[3]+gps_noise[3],true_first_pose[4]+velocity_noise[1],true_first_pose[5]+velocity_noise[2],true_first_pose[6]+velocity_noise[3],1e-5, 2e-5, 3e-5]

# P_0 = I(9).*[ones(3)*((std_cv_measurement)^2)/3; ones(3)*((std_velocity)^2)/3; ones(3)*(std_accel)^2] 
P_0 = torch.diag([((std_cv_measurement)**2)/3]*3 + [((std_velocity)**2)/3]*3 + [(std_accel)**2]*3)

#take the cholesky factorization
F_0 = P_0.sqrt()    #P_0)

all_states = torch.zeros(bsz, 9, CV_num)

cov_sqrt_all = torch.zeros((bsz, CV_num, 9, 9))# for i in CV_num]

#set first value to X_0
all_states[:,:,0] = X_0

cov_sqrt_all[:,0] = F_0


#run EKF for every time step

for k in range(CV_num-1):
    
    #obtain the updated and state at the next timestep (this is original, works)
    X, F = EKF_satellite_QR(all_states[:,:,k], cov_sqrt_all[:,k], k)
    
    #X_0 = X
    
    #F_0 = F
    
    #all_states[:,k+1] = X_0
    
    #cov_sqrt_all[k+1] = F_0
    
    all_states[:,:,k+1] = X
    
    cov_sqrt_all[:,k+1] = F
    
    #J_all[k+1] = Jk1
    
    

# sample = Array(range(1,size(all_drag[1,:])[1], step=100)) #create a range to final time at constant time step
