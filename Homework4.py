# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Astro 585 Homework 4

# <headingcell level=2>

# Moupiya Maji

# <headingcell level=3>

# 1. Profiling

# <codecell>

# HW4_Q3_leapfrog.jl

function update_derivs_pos!(state::Vector{Float64}, derivs::Vector{Float64} )
 # Input: state = [x,y,vx,vy], a vector of two 2-d positions and velocities for a test particle
 # Output: The derivatives of the position are updated in the preallocated array derivs.
  @assert length(state) == 4
  @assert length(derivs) == 4
  v_x = state[3]
  v_y = state[4]
  derivs[1] = v_x
  derivs[2] = v_y
  return derivs;
end

function update_derivs_vel!(state::Vector{Float64}, derivs::Vector{Float64} )
  # Input: state = [x,y,vx,vy], a vector of two 2-d positions and velocities for a test particle
  # Output: The derivatives of the velocity are updated in the preallocated array derivs.  
  @assert length(state) == 4
  @assert length(derivs) == 4
  GM = 1.0
  r_x = state[1]
  r_y = state[2]
  r2 = r_x*r_x+r_y*r_y
  a = -GM/r2
  r = sqrt(r2)
  a_x = a * r_x/r
  a_y = a * r_y/r
  derivs[3] = a_x
  derivs[4] = a_y
  return derivs;
end

function update_derivs!(state::Vector{Float64}, derivs::Vector{Float64} )
  # Input: state = [x,y,vx,vy], a vector of two 2-d positions and velocities for a test particle
  # Output: The derivatives are updated in the preallocated array derivs.  
  update_derivs_vel!(state,derivs)
  update_derivs_pos!(state,derivs)
  return derivs
end
  
function advance_leapfrog!(state::Vector{Float64},derivs::Vector{Float64}, dt::Float64; derivs_current::Bool = false)
  # Input/Output: state = array of two 2-d positions and velocities for a test particle
  # Temporary space: The derivatives are updated in the preallocated array derivs.
  # Input: dt is the fixed time step 
  # Optional param: derivs_current: whether need to calculate derivatives at beginning
  @assert length(state) == length(derivs)
  
  if !derivs_current 
    update_derivs_pos!(state,derivs);
  end
  {state[i] = state[i] + 0.5*dt*derivs[i]  for i in 1:2}
  update_derivs_vel!(state,derivs);
  {state[i] = state[i] + dt*derivs[i]      for i in 3:4}
  update_derivs_pos!(state,derivs);
  {state[i] = state[i] + 0.5*dt*derivs[i]  for i in 1:2}    
end

# Input/Output: state = [x,y,vx,vy], an array of two 2-d positions and velocities for a test particle
# Input: dt is the fixed time step 
# Input: duration is the total 
function integrate_leapfrog!(state::Vector{Float64}, dt::Float64, duration::Float64; max_num_log::Integer = 100000)
  @assert(length(state)==4) 
  @assert(dt>0.0)
  @assert(duration>0.0)
  
  # Preallocate array to hold data log (including  initial state)
  nsteps = iceil(duration/dt);
  nskip = (nsteps<max_num_log) ? 1 : iceil(nsteps/(max_num_log-1))
  num_log = iceil(nsteps/nskip)+1   
  log = Array(Float64,(num_log,length(state)));

  # Pre-allocate and pre-compute derivaties
  derivs = Array(Float64,4);  
  update_derivs!(state,derivs);

  # Log initial state 
  log_pos = 1
  log[log_pos,:] = deepcopy(state) 

  n = 0
  t = 0.0
  while t<duration
    # ensure don't integrate for more than duration
    dt_tmp = (t+dt<=duration) ? dt : duration-t;

	# advance system by one time step
    advance_leapfrog!(state,derivs,dt_tmp, derivs_current=true)
    t = t + dt_tmp
    n = n + 1

    if (n%nskip==0) # Log data
	   log_pos += 1
   	   @assert( log_pos<=length(log) )
	   @assert( length(log[log_pos,:])==length(state) )
	   log[log_pos,:] = deepcopy(state) 
	end
  end
  return log
end

function calc_error_leapfrog_old(dur::Float64, dt::Float64 = 2pi/200.0)
  state = [1.,0.,0.,1.];  
  integrate_leapfrog!(state,dt,dur*2pi);
  dist = state[1]^2+state[2]^2
  phase = atan2(state[2],state[1])
  offset = sum((state[1:2].-[1.0,0.0]).^2)
  return (dist-1.0,phase,offset)
end

function calc_end_distance_leapfrog(dur::Integer, dt::Float64 = 2pi/200.0, state::Vector{Float64} = [1., 0., 0., 1.] )
  #state = [1.,0.,0.,1.];  
  dist_init = sqrt(state[1]^2+state[2]^2)
  phase_init = atan2(state[2],state[1])
  integrate_leapfrog!(state,dt,dur*2pi);
  # Calculate three metrics of the accuracy of the integration  
  dist = sqrt(state[1]^2+state[2]^2)
  phase = atan2(state[2],state[1])
  offset = sqrt(sum((state[1:2].-[1.0,0.0]).^2))
  return (dist-dist_init,phase-phase_init,offset)
end

using Base.Test
function test_leapfrog(dur::Integer, dt::Float64 = 2pi/200.0)  
  err = calc_end_distance_leapfrog(dur,dt)
  @test_approx_eq_eps(err[1], 0., 1e-6)
  @test_approx_eq_eps(err[2], 0., 1e-1)
  @test_approx_eq_eps(err[3], 0., 1e-1)
end

# <codecell>

function derrivatives(state::Array,GM)
    r = sqrt(state[1]^2+state[2]^2)
    dvx = -GM*state[1]/r^3
    dvy = -GM*state[2]/r^3
    drx = state[3]
    dry = state[4]  
  return [drx,dry,dvx,dvy]
end

function leapfrog_update_pos(state::Array,dt)  
  Rx = state[1] + .5*dt*state[3]
  Ry = state[2] + .5*dt*state[4]  
  newstate = [Rx,Ry,state[3],state[4]]
  return newstate
end

function leapfrog_update_both(state::Array,ds,dt)
  vx = state[3]+dt*ds[3]
  vy = state[4]+dt*ds[4]
  [state[1]+.5*dt*vx,state[2]+.5*dt*vy,vx,vy]
end


function integrate_leapfrog_student(state::Array,dt,duration,GM=1) 
  r_x,r_y,v_x,v_y = [],[],[],[]
  N = iceil(duration/dt) 
  t = 0.
  for i in 0:N
    dt_tmp = (t+dt<=duration) ? dt : duration-t;

    state = leapfrog_update_pos(state,dt_tmp)
    ds = derrivatives(state,GM)
    state = leapfrog_update_both(state,ds,dt_tmp)
    t += dt_tmp

    r_x = vcat(r_x,[state[1]])
    r_y = vcat(r_y,[state[2]])
    v_x = vcat(v_x,[state[3]])
    v_y = vcat(v_y,[state[4]])
  end
  return (r_x,r_y,v_x,v_y)
end



function calc_end_distance_leapfrog_student(dur::Integer, dt::Float64 = 2pi/200.0, state::Vector{Float64} = [1., 0., 0., 1.] )
  #state = [1.,0.,0.,1.];  
  dist_init = sqrt(state[1]^2+state[2]^2)
  phase_init = atan2(state[2],state[1])
  datalog = integrate_leapfrog_student(state,dt,dur*2pi)
  # Since student version didn't modify state (no !), we need to update state with the final value
  state = [datalog[1][end], datalog[2][end], datalog[3][end], datalog[4][end]]
  # Calculate three metrics of the accuracy of the integration
  dist = sqrt(state[1]^2+state[2]^2)
  phase = atan2(state[2],state[1])
  offset = sqrt(sum((state[1:2].-[1.0,0.0]).^2))
  return (dist-dist_init,phase-phase_init,offset)
end

using Base.Test
  function test_leapfrog_student(dur::Integer, dt::Float64 = 2pi/200.0)
  err = calc_end_distance_leapfrog_student(dur,dt)
  @test_approx_eq_eps(err[1], 0., 1e-6)
  @test_approx_eq_eps(err[2], 0., 1e-1)
  @test_approx_eq_eps(err[3], 0., 1e-1)
end

# <markdowncell>

# 1b. In the test_leapfrog function, I think the while loop in the integrate leapfrog function where it integrates the function by calling advance leapfrog is the most time consuming part. It will probably take 90% of the time.
# 
# In the test_leapfrog_student function, I think the for loop in integrate leapfrog student takes the longest time, possibly 90%.

# <markdowncell>

# 1c.

# <markdowncell>

# Profiling function Integrate_leapfrog!

# <codecell>

Profile.clear()

# <codecell>

@profile (for i=1:1000; integrate_leapfrog!([1.,0.,0.,1.],0.01,1.0); end)

# <codecell>

Profile.print()

# <markdowncell>

# 1d. In this function the most time consuming line is the calling the advance_leapfrog function inside the while loop in the function, line 87 is the most time consuming part. This is what I have guessed above. Also the deepcopy of the state in each loop inside the while takes a lot of time, which I have not predicted. The deepcopy takes time because iot copies the whole structure of the state at each time, which will be time consuming.

# <codecell>

Profile.clear()

# <markdowncell>

# Profiling Integrate_leapfrog_student

# <codecell>

@profile (for i=1:1000; integrate_leapfrog_student([1.,0.,0.,1.],2pi/200.,6.,1); end)

# <codecell>

Profile.print()

# <markdowncell>

# In this function, we see that the steps inside the for loop in the function, where it updates the state and increments rx,ry,vx and vy are the most time consuming. Especially the concatenation of vx takes the most amount of time. this is similar to what I have predicted above.

# <markdowncell>

# 1e. To speed up the code we need to get a way to bypass the while loop. Coming back to it.

# <headingcell level=2>

# 3. Effect of branching

# <codecell>

function triad(b::Vector, c::Vector, d::Vector)
  assert(length(b)==length(c)==length(d))
  a = similar(b)
  for i in 1:length(a)
     a[i] = b[i] + c[i] * d[i]
  end
  return a
end

# <codecell>

function triad_twist1(b::Vector, c::Vector, d::Vector)
  assert(length(b)==length(c)==length(d))
  a = similar(b)
  for i in 1:length(a)
     if c[i]<0. 
       a[i] = b[i] - c[i] * d[i]
     else 
       a[i] = b[i] + c[i] * d[i]
     end
  end
  return a
end

# <codecell>


function triad_twist2(b::Vector, c::Vector, d::Vector)
  assert(length(b)==length(c)==length(d))
  a = similar(b)
  for i in 1:length(a)
     if c[i]<0. 
       a[i] = b[i] - c[i] * d[i]
     end
  end
  for i in 1:length(a)
     if c[i]>0. 
       a[i] = b[i] + c[i] * d[i]
     end
  end
  return a
end

# <codecell>

function triad_twist3(b::Vector, c::Vector, d::Vector)
  assert(length(b)==length(c)==length(d))
  a = similar(b)
  for i in 1:length(a)
     cc = abs(c[i])
     a[i] = b[i] + cc * d[i]
  end
  return a
end

# <markdowncell>

# 3a. The triad function creates a vercor through some combination of three input vectors (basically, a=b+cd). The twist variation of triad tries to take care of the effect of sign of c (+ve or -ve). 
# 
# In the triad_twist1, it is achieved in a simple if else loop, where if an element of c is negative, we will change the operation to a=b-cd to get the right answer. 
# 
# In triad_twist2 function, there are two for loops, one for positive c values and one for negative values, both runs for the entire length of c array. Clearly, this is more time consuming than the twist 1 version because each element of the arrays will be grabbed from memory twice where in twist 1 it is accessed only once. this means, as the size of arrays change to 10^3 to 10^6 twist will take much more time to run. For twist2, equally positive or negative components in c will take more time that having all negative or all positive values. The reason is that if all values are one kind, it does not have to enter in one of the loops at all, so it does not have to grab elements of b and d twice. On the other hand, in twist 1 function, it does not matter if all c value signs are +ve, -ve or mixed as it will access all elements once only.
# 
# In triad_twist3 function, it just takes the absolute value of each elements in c upfront and then calculates a. Since it does not have to go through any decision process, it will take less time than triad twist1 function and this will be more prominent for larger array sizes. The number of +ve or -ve c value will not have any effect, similar as in twist 1. (There might be some effect if absolute function take a significant time for -ve elements.) Comparing to twist 2, twist 3 will take shorter time, as it does not have to access array elements twice.

# <markdowncell>

# Effect of Array size

# <codecell>

b1=rand(800);
c1=rand(800);
d1=rand(800);
triad(b1,c1,d1);
print("Array size small=800\n")
print("triad:         ",  @elapsed x=triad(b1,c1,d1));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b1,c1,d1));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b1,c1,d1));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b1,c1,d1));

# <codecell>

b2=rand(8000000);
c2=rand(8000000);
d2=rand(8000000);
print("Array size large=8*10^6\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <markdowncell>

# As expected, both for small arrays and large arrays, triad twist 2 takes longest time among twisted functions and triad twist 3 takes shortest time. Twist 1 takes time in between, more similar to twist 3. The effects are more pronounced for the larger array, of course. Triad function takes shortest time, as it does not compute any extra parameters like absolute values etc.

# <markdowncell>

# Effect of Number of positive/negative elements

# <codecell>

b2=rand(8000000);
c2=rand(8000000);
d2=rand(8000000);
print("All positive\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(8000000);
c2=-rand(8000000);
d2=rand(8000000)
print("All negative\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(8000000);
c2=[-rand(4000000),rand(4000000)];
d2=rand(8000000)
print("Half negative and half positive\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(8000000);
c2=[-rand(1000000),rand(7000000)];
d2=rand(8000000)
print("88% positive c values\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <markdowncell>

# Here we see, that the times are very similar for all positive or all negative elements in c. However for twist2, half negative and half positive case, it takes much more time. This is consistent with what I discussed before. 

# <markdowncell>

# Trying the ame variations with small array sizes.

# <codecell>

b2=rand(800);
c2=rand(800);
d2=rand(800);
print("All positive\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(800);
c2=-rand(800);
d2=rand(800)
print("All negative\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(800);
c2=[-rand(400),rand(400)];
d2=rand(800)
print("Half negative and half positive\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <codecell>

b2=rand(800);
c2=[-rand(100),rand(700)];
d2=rand(800)
print("88% positive c values\n")
print("triad:         ",  @elapsed x=triad(b2,c2,d2));
print("\ntriad_twist1:  ",  @elapsed x=triad_twist1(b2,c2,d2));
print("\ntriad_twist2:  ",  @elapsed x=triad_twist2(b2,c2,d2));
print("\ntriad_twist3:  ",  @elapsed x=triad_twist3(b2,c2,d2));

# <markdowncell>

# For smaller array sizes, we see the same effects, but in smaller amounts.

# <markdowncell>

# Profiling

# <codecell>

Profile.clear()

# <codecell>

b2=rand(8000000);
c2=rand(8000000);
d2=rand(8000000);
Profile.clear()
#@profile triad(b2,c2,d2);
print (for i=1:2; @profile triad(b2,c2,d2); end)
Profile.print()

# <codecell>

Profile.clear()
print (for i=1:2; @profile triad_twist1(b2,c2,d2); end)
Profile.print()

# <codecell>

Profile.clear()
print (for i=1:2; @profile triad_twist2(b2,c2,d2); end)
Profile.print()

# <codecell>

Profile.clear()
print (for i=1:2; @profile triad_twist3(b2,c2,d2); end)
Profile.print()

# <markdowncell>

# Profiling with half -ve and half +ve elements

# <codecell>

b2=rand(8000000);
c2=[-rand(4000000),rand(4000000)];
d2=rand(8000000)
Profile.clear()
print (for i=1:2; @profile triad_twist1(b2,c2,d2); end)
Profile.print()

# <codecell>

Profile.clear()
print (for i=1:2; @profile triad_twist2(b2,c2,d2); end)
Profile.print()

# <codecell>

Profile.clear()
print (for i=1:2; @profile triad_twist3(b2,c2,d2); end)
Profile.print()

# <markdowncell>

# Profiling the function with two types of arrays we see that for twist2 and twist 3 it takes the longest time in calculating the elements of the array. Interestingly, when half the elements are positive only, the twist2 function takes more time in the negative loop though it should take same amount of time in both loops. I think it's a statistical effect. But more samples are making the buffer full so I did not try it. 

# <markdowncell>

# In esssence we see that avoiding branching as far as possible makes the code more efficient.

# <headingcell level=2>

# 2.  Loop vs “Vectorized” vs Map vs MapReduce vs “Devectorized”

# <markdowncell>

# 2a.

# <codecell>

function fn_normal(x::Float64)
    y=exp(-0.5*x^2)/sqrt(2pi)
end

# <markdowncell>

# 2b.

# <codecell>

function int_for(a::Float64,b::Float64)
    N=1000;
    h=(b-a)/N;
    sum=fn_normal(a);
    x=a;
    @parallel for i=1:N-1
        x=x+h;
        sum=sum+2*fn_normal(x);
    end
    return (sum+fn_normal(b))*(h/2)
end
        

# <codecell>

@elapsed x=int_for(0.,1.)

# <markdowncell>

# Without parallel, time taken is 1e^-4, but when I add parallel in front of for, the time reduces to 10^-5!

