using LinearAlgebra

struct OrbitalElements
    # Semi-major axis
    a::Number
    # Eccentricity
    e::Number
    # Inclination
    i::Number
    # Right Ascension of Ascending Node
    Ω::Number
    # Argument of Periapsis
    ω::Number
    # True Anomaly
    ν::Number
end

function oe2eci(oe::OrbitalElements, μ=398600.4418)
    
    P = oe.a*(1-oe.e^2);                # Semi-Latus Rectum
    r_mag = P/(1+oe.e*cos(oe.ν));       # Distance from Earth to orbiting body

    n = sqrt(μ/oe.a^3)
    E = anom2E(oe.ν,oe.e)
    # R in perifocial coordinates [P Q W]'
    # r_peri = [r_mag*cos(oe.ν); r_mag*sin(oe.ν); 0];
    # v_peri = sqrt(μ/P)*[-sin(oe.ν); (oe.e+cos(oe.ν)); 0];
    r_peri = [oe.a*(cos(E) - oe.e); oe.a*sqrt(1 - oe.e^2)*sin(E);0];
    v_periComp = [-sin(E);sqrt(1 - oe.e^2)*cos(E);0];
    v_peri = (oe.a*n)/(1 - oe.e*cos(E))*v_periComp;
    if oe.i == 0 && oe.e != 0         # Equitorial and Elliptical
        R1 = 1;
        R2 = 1;
        R3 = rotz(oe.ω);
    elseif oe.e == 0 && oe.i != 0     # Circular and Inclined
        R1 = rotz(oe.Ω);
        R2 = rotx(oe.i);
        R3 = 1;
    elseif oe.i == 0 && oe.e == 0     # Equitorial and Circular
        R1 = 1;
        R2 = 1;
        R3 = 1;
    else                              # Not Circular or Inclined
        R1 = rotz(oe.Ω);
        R2 = rotx(oe.i);
        R3 = rotz(oe.ω);
    end
    R = R1*R2*R3;                     # Full rotation matrix
    r_eci = R*r_peri;
    v_eci = R*v_peri;
    return [r_eci; v_eci]
end

function anom2E(ν,e)
    E = acos((e + cos(ν))/(1 + e*cos(ν)));
    if ν > π
        E = 2π - E;
    end
    return E
end

function rotz(γ)
    rotmat = [cos(γ) -sin(γ) 0; sin(γ) cos(γ) 0; 0 0 1];
    return rotmat
end

function rotx(α)
    rotmat = [1 0 0;0 cos(α) -sin(α); 0 sin(α) cos(α)];
    return rotmat
end

function  eci2oe(x, μ=398600.4418)

    R = x[1:3]
    V = x[4:6]
    
    r = norm(R)
    v = norm(V)

    H = cross(R,V)
    h = norm(H)

    N = cross([0;0;1],H)
    n = norm(N)
    e_vec = 1/μ*((v^2-μ/r).*R-dot(R,V).*V)
    e = norm(e_vec)

    # Mechanical Energy to determine size
    ϵ = 0.5*v^2 - μ/r
    if e != 1
        a = -μ/(2*ϵ)
    else
        a = inf # Semi-major axis undefined for parabolas
    end

    # Orbital Inclination (always less than 180 deg)
    i = acos(H[3]/h)

    # Rignt Ascension of Ascending Node
    Ω = acos(N[1]/n)
    if N[2] < 0             # If Nⱼ is greater than 0 Om is less than 180
        Ω = 2π- Ω
    end

    # Argument of periapsis
    term = dot(N,e_vec)/(n*e)
    ϵ = 1e-10
    if abs(term) > 1 # checking precision of values
        if abs(term)-1 < ϵ
            if term < 0 term = -1 else term = 1 end
        end
    end
    ω = acos(term)
    if e_vec[3] < 0         # If e(k) is greater than 0 w is less than 180
        ω = 2π - ω;
    end

    # True anomaly
    term = dot(e_vec,R)/(e*r)
    ϵ = 1e-10
    if abs(term) > 1
        if abs(term)-1 < ϵ
            if term < 0 term = -1 else term = 1 end
        end
    end

    ν = acos(term);
    if dot(R,V) < 0         # If R dot V is greater than zero nu is less than 180
        ν = 2π - ν;
    end

    # Special Cases, use different angles
    if i == 0 && e != 0 # Elliptical equatorial
        # Provide the longitude of periapsis (PI = Om + w)
        ang = acos(e_vec[1]/e)
        if e_vec[2] < 0
            ang = 2π - ang;
        end
    elseif i != 0 && e == 0 # Circular inclined
        # Provide the argument of latitude (u = w + anom)
        ang = acos(dot(N,R)/(n*r))
        if r_eci[3] < 0
            ang = 2π - ang;
        end
    elseif i == 0 && e == 0 # Circular equatorial
        # Provide the true latitude (lambda = Om + w + anom)
        ang = acos(R[1]/r)
        if R[2] < 0
            ang = 2π - ang;
        end
    else
        # Default output for ang
        ang = NaN;
    end

    oe = OrbitalElements(a,e,i,Ω,ω,ν)
    return oe, ang
end

function orbit_dynamics(x_orbit, μ=398600.4418, J2=1.75553e10)
    r = x_orbit[1:3]
    v = x_orbit[4:6]
    
    v̇ = -(μ/(norm(r)^3))*r + (J2/(norm(r)^7))*(r.*([6 -1.5 -1.5; 6 -1.5 -1.5; 3 -4.5 -4.5]*(r.*r)))
    
    return [v; v̇]
end

function orbit_step(xk, h)
    #Take a single RK4 step on the orbit dynamics
    
    f1 = orbit_dynamics(xk)
    f2 = orbit_dynamics(xk+0.5*h*f1)
    f3 = orbit_dynamics(xk+0.5*h*f2)
    f4 = orbit_dynamics(xk+h*f3)
    
    xn = xk + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end
     

#Random initial conditions
#Polar Orbit
oe_polar = OrbitalElements(600.0+6378.0+(100*rand()-50), 0.0+0.01*rand(), (pi/2)+(0.2*rand()-0.1), 2*pi*rand(), 2*pi*rand(), 2*pi*rand());
# oe_polar = OrbitalElements(600.0+6378.0+(100*0.5-50), 0.0+0.01*0.5, (pi/2)+(0.2*0.5-0.1), 2*pi*0.5, 2*pi*0.5, 2*pi*0.5);

#ISS~ish Orbit
#eo_iss = OrbitalElements(420.0+6378.0+(100*rand()-50) ,0.00034+0.01*rand(), (51.5*pi/180)+(0.2*rand()-0.1), 2*pi*rand(), 2*pi*rand(), 2*pi*rand());

x0_orbit = oe2eci(oe_polar)

#simulate for 3 hours (~2 orbits)
tf = 3*60*60
tsamp = 0:tf

xtraj_orbit = zeros(6, length(tsamp))
xtraj_orbit[:,1] .= x0_orbit

for k = 1:(length(tsamp)-1)
    xtraj_orbit[:,k+1] .= orbit_step(xtraj_orbit[:,k], 1.0)
end
     

#3U CubeSat Inertia (MKS units)
m = 4.0;
J = Diagonal([(m/12)*(.1^2+.34^2); (m/12)*(.1^2+.34^2); (m/12)*(.1^2+.1^2)])

#Quaternion stuff
function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end
function L(q)
    s = q[1]
    v = q[2:4]
    L = [s    -v';
         v  s*I+hat(v)]
    return L
end
T = Diagonal([1; -ones(3)])
H = [zeros(1,3); I]
function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end
function G(q)
    G = L(q)*H
end
function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end
function qtorp(q)
    q[2:4]/q[1]
end

function attitude_dynamics(x_attitude, J = Diagonal((1/3)*[(.1^2+.34^2); (.1^2+.34^2); (.1^2+.1^2)]))
    q = x_attitude[1:4]
    q = q/norm(q)
    ω = x_attitude[5:7]
    
    q̇ = 0.5*G(q)*ω
    ω̇ = -J\(hat(ω)*J*ω)
    return [q̇; ω̇]
end

function attitude_step(xk, h)
    #Take a single RK4 step on the attitude dynamics
    
    f1 = attitude_dynamics(xk)
    f2 = attitude_dynamics(xk+0.5*h*f1)
    f3 = attitude_dynamics(xk+0.5*h*f2)
    f4 = attitude_dynamics(xk+h*f3)
    
    xn = xk + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xn[1:4] .= xn[1:4]/norm(xn[1:4]) #re-normalize quaternion
    
    return xn
end

#Random initial conditions
q0 = randn(4)
q0 = q0/norm(q0)
ω0 = 2*(pi/180)*randn(3)
x0_attitude = [q0; ω0]
     

#simulate

xtraj_attitude = zeros(7, length(tsamp))
xtraj_attitude[:,1] .= x0_attitude

for k = 1:(length(tsamp)-1)
    xtraj_attitude[:,k+1] .= attitude_step(xtraj_attitude[:,k], 1.0)
end
     