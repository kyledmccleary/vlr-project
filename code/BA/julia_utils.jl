using Pkg; Pkg.activate("."); Pkg.instantiate()
using SatelliteDynamics
using LinearAlgebra

function get_r_moon_sun_PNs()

    # initial time for sim
    epc0 = Epoch(2012, 11, 8, 12, 0, 0, 0.0)

    #Test of some of the SatelliteDynamics functions

    #Orbit we want (around ISS orbit conditions)
    iss1 = [6871e3, 0.00064, 51.6450, 1.0804, 27.7899, 190];

    # Convert osculating elements to Cartesean state
    # returns position and velocity (m, m/s). This is the intial position
    eci0_1 = sOSCtoCART(iss1, use_degrees=true)

    #find the period of the orbit (seconds). only dependent on semi major axis
    T = orbit_period(iss1[1])

    #final time of simulation
    epcf = epc0 + T

    h = 1 #1 Hz the timestep
    orbit_num = 0.1 #number of orbits to simulate

    #final time for the simulation (period x number of orbits)
    Tf = T*orbit_num

    #run the rk4
    t = Array(range(0,Tf, step=h)) #create a range to final time at constant time step

    PNs = zeros(3,3,length(t)*2) #preallocate the nutation matrix
    r_suns = zeros(3,length(t)*2) #preallocate the sun position
    r_moons = zeros(3,length(t)*2) #preallocate the moon position
        
    for k=1:(length(t))
        current_t = epc0+t[k] #calculate the current time
        #look up this term. seems to give a rotation matrix
        PNs[:,:,k*2-1] = bias_precession_nutation(current_t)
        #Compute the sun and moon positions in ECI frame
        r_suns[:,k*2-1] = sun_position(current_t)
        r_moons[:,k*2-1] = moon_position(current_t)

        current_t = current_t + h/2
        #look up this term. seems to give a rotation matrix
        PNs[:,:,k*2] = bias_precession_nutation(current_t)
        #Compute the sun and moon positions in ECI frame
        r_suns[:,k*2] = sun_position(current_t)
        r_moons[:,k*2] = moon_position(current_t)
    end

    return r_moons, r_suns, PNs
end