from casadi import *
import casadi as ca
import scipy as sc
import scipy.interpolate as scint
import matplotlib.pyplot as plt
import numpy as np
import bspline
import bspline.splinelab as splinelab
import random as r
import matplotlib.animation as animation
import matplotlib.patches as patches


# functions to fulfill my demands 
def bsder(knots, k): 
    """
    Computes a transformation matrix T that can be used to express the
    derivative of a B-spline by using the same coefficients that the spline has.
    If f(x)=bscol(knots, k, x) * coeffs, then f'(x)=bscol(knots, k-1, x) * T * coeffs.
    
    Parameters:
        knots (list or np.array): Knot vector for the B-spline.
        k (int): Order of the B-spline.
    
    Returns:
        T (ca.DM): Transformation matrix for the derivative of the B-spline.
    """
    m = len(knots)
    k= k+1
    def zerocheck(a):
        return ca.if_else(a == 0, 1, a)

    T = ca.DM.zeros(m - k + 1, m - k)

    for i in range(m - k):
        T[i, i] = (k - 1) / zerocheck(knots[i + k - 1] - knots[i])
        T[i + 1, i] = -(k - 1) / zerocheck(knots[i + k] - knots[i + 1])

    return T

def bscol(knots, k, tau):
    """
    Evaluates the B-spline basis functions defined by (knots, k) at tau.
    
    Parameters:
        knots (list or np.array): Knot vector for the B-spline.
        k (int): Order of the B-spline.
        tau (list or np.array): Points at which to evaluate the basis functions.
    
    Returns:
        col (ca.DM or np.array): Evaluated B-spline basis functions at tau.
    """
    tau = np.transpose(tau)
    n = len(tau)
    m = len(knots)
    k = k+1
    def zerocheck(a):
        return ca.if_else(a == 0, 1, a)
    
    N = np.zeros((m,k,n))


    for t_idx in range(n):
        for l in range(1, k + 1):
            for i in range(m - l):
                if l == 1:
                    N[i, l - 1, t_idx] = ca.if_else((knots[i] <= tau[t_idx]) & (tau[t_idx] <= knots[i + 1]), 1, 0)
                else:
                    term1 = (tau[t_idx] - knots[i]) / zerocheck(knots[i + l - 1] - knots[i])
                    term1 = term1 * N[i, l - 2, t_idx]

                    term2 = (knots[i + l] - tau[t_idx]) / zerocheck(knots[i + l] - knots[i + 1])
                    term2 = term2 * N[i + 1, l - 2, t_idx]

                    N[i, l - 1, t_idx] = term1 + term2
    
    col = np.array([N[0:m - k, k - 1, t_idx] for t_idx in range(n)])

    if n == 1:
        col = col.T
    
    return ca.horzcat(*col) if n > 1 else col.flatten()

class Obstacle:
    def __init__(self,x,y):
        self.cp = [x,y]

class Circle (Obstacle):
    '''
    for the circular obstacles -> lacks movement yet
    
    '''
    def __init__ (self, x,y,r):
        self.nv =1
        self.v =np.array([x,y])
        self.r = r
    
class Rectangle (Obstacle):
    '''
    rectangular obstacles
    '''
    def __init__(self,x,y,w,h):
        self.nv = 4
        self.v = np.array([[x-w/2,y-h/2],[x-w/2,y+h/2],[x+w/2,y+h/2],[x+w/2,y-h/2]])
        self.r = 0
        self.csucs = [x-w/2,y-h/2]
        self.w = w
        self.h = h
         
def openfig(): # making an animation plot scene with the correct boundaries
    fig, ax = plt.subplots()
    ax.grid(color='k', linestyle=':', linewidth=0.5)
    ax.plot(q_start[0],q_start[1],'xr', ms ="10")
    ax.plot(q_end[0],q_end[1],'xr', ms ="10")
    ax.set(xlim=(bounds[0][0], bounds[0][1]), ylim=(bounds[1][0], bounds[1][1]))
    ax.axis('equal')
    return fig, ax

def animate_scene(mobs): # plotting the obstacles for the animation
    fig, vx = openfig()
    mobs_art = []
    for i in range(0, len(mobs)): # plotting the mobs
        if mobs[i].nv == 4:
            mobs_art.append(patches.Rectangle(mobs[i].csucs, mobs[i].w,mobs[i].h, facecolor='r'))
            vx.add_patch(mobs_art[i])
        elif mobs[i].nv ==1:
            mobs_art.append(patches.Circle(mobs[i].v, mobs[i].r, facecolor='r'))
            vx.add_patch(mobs_art[i])
    
    ani_t = 200                 # the animation's samplesize -> will vary with the 
    ani_x = np.zeros(ani_t)     
    ani_y = np.zeros(ani_t)
    for i in range(ani_t):      # the bsplines' points are collected into arrays -> the spline is a function
        ani_x[i] = bsp_x(i/ani_t)
        ani_y[i] = bsp_y(i/ani_t)
    ani_x[0] = q_start[0]
    ani_y[0] = q_start[1]
    ani_x[-1] = q_end[0]
    ani_y[-1] = q_end[1]
    drone = plt.Circle(xy=(ani_x[0],ani_y[0]),radius=rad,ec='k',color="b")
    line, = vx.plot([],[],color='k', linestyle='--', linewidth=1)
    vx.add_patch(drone)

    def update_data(frame):     # function which give us the appropriate points 
        line.set_data([ani_x[:frame]],[ani_y[:frame]])
        drone.set_center(xy=(ani_x[frame],ani_y[frame]))

    anim = animation.FuncAnimation(    
                        fig = fig,
                        func=update_data,
                        frames = ani_t,
                        interval = 2
                        )
    #anim.save("optimization_2D.gif","ffmpeg",60,150) 
    plt.show()
    
# -------- points: starting point and goal --------
q_start = [0,0]     # starting point in [m]
q_end = [3,3]       # goal in [m]
bounds = [[-1,4],[-1,4]]
rad = 0.1           # radius of the drone [m]
safety = 5e-2

# -------- B-spline allocation --------
nknots = 10
N = 2000 # number of control intervals
k = 3   # degree of the spline
knots_0 = np.linspace(0,1,nknots)
knots = splinelab.augknt(knots_0,k) # knots 
nb = nknots + k-1                     # number of basis functions

'''
#  -------- getting the positions of the coefficients --------
x = np.linspace(0,1,1001)
colmat = splinelab.spcol(knots, k,x)
coeffs_poz = np.zeros(nb)
max =0
for i in range(1,nb-1):
    for j in range (0,len(x)):
        if max < colmat[j,i]:
            max = colmat[j,i]
            coeffs_poz[i]=j/len(x)
        else:
            max = colmat[j,i]

coeffs_poz[0]=0; coeffs_poz[-1]=1
'''

# -------- Matrixes for the b-splines ------
c_max=11                    # number of the points where we calculate the b-splines -> minimal time gridding
c = np.linspace(0,1,c_max)
dmx = bscol(knots, k, c)


g = bscol(knots, k-1,c) # derivate's collocation matrix
P =  bsder(knots, k)    

# -------- initialization of the obstacles -------
mob = Rectangle(1,2.5,0.1,4) # declare the obstacle
mobs = []
mobs.append(mob)
#mob2 = Rectangle(2.3,0,0.2,4)
#mobs.append(mob2)

# ------- optimization problem -------
opti = Opti()

# ------- decision variables ------
Coeffs = opti.variable(nb,2)        # coefficients -> number of basis functions are given
q = opti.variable(c_max,2)          # location -> since the g, dmx matrixes are expressed with c_max, the dim is c_max too
vel = opti.variable(c_max,2)        # velocity
T = opti.variable()                 # final time

r_veh = opti.parameter()            # drone's radius 
opti.set_value(r_veh,rad)

# ------- objective --------
opti.minimize(T)

# ------- kinematic constrains --------
opti.subject_to(q[:,0]==dmx.T@Coeffs[:,0]) # getting q_x as a b-spline
opti.subject_to(q[:,1]==dmx.T@Coeffs[:,1])

# position
opti.subject_to(opti.bounded(bounds[0][0]+r_veh,Coeffs[:,0],bounds[0][1]-r_veh))      # bounding where it should stay -> boundaries of the map
opti.subject_to(opti.bounded(bounds[1][0]+r_veh,Coeffs[:,1],bounds[1][1]-r_veh))      

opti.subject_to(q[0,0]==q_start[0])                                                   # start and final positions -> can imporve the time in some cases, otherwise the coeff limitations should be enough
opti.subject_to(q[0,1]==q_start[1]) 
opti.subject_to(q[-1,0]==q_end[0])
opti.subject_to(q[-1,1]==q_end[1])

# limiting coefficients
opti.subject_to(Coeffs[0,0]==q_start[0])            # first and last coeffs should be the points themselves
opti.subject_to(Coeffs[0,1]==q_start[1])
opti.subject_to(Coeffs[-1,0]==q_end[0])
opti.subject_to(Coeffs[-1,1]==q_end[1])

# velocity
opti.subject_to(vel[:,0]*T == g.T@P@Coeffs[:,0])       # expressing the velocities as b-splines
opti.subject_to(vel[:,1]*T == g.T@P@Coeffs[:,1])

opti.subject_to(vel[0,0]==0)                          # begining from a halt and halting at the destination
opti.subject_to(vel[0,1]==0)
opti.subject_to(vel[-1,0]==0)
opti.subject_to(vel[-1,1]==0)
vel_max = 0.5                                       # velocity limitations -----> m/s
#opti.subject_to(opti.bounded(0,vel_x,vel_max))     # for component limits
#opti.subject_to(opti.bounded(0,vel_y,vel_max))

opti.subject_to(opti.bounded(-vel_max**2,vel[:,0]**2+vel[:,1]**2,
                             vel_max**2))           # can't use sqrt() in casadi

# ------- obstacle avoidance --------
'''
még nincs megoldva -> time gridding formában működik jelenleg -> kevés grid esetén nem működik vagy sok grid esetén lassú
'''
Coeffs_a = opti.variable(nb,2)                      # separting hyperplane's b-spline
a= opti.variable(c_max,2)                           # the normal vector of the line
Coeffs_b = opti.variable(nb,1)       
b = opti.variable(c_max,1)                          # line's offset
opti.subject_to(a[:,0]==dmx.T@Coeffs_a[:,0])        # parameterized as b-splines
opti.subject_to(a[:,1]==dmx.T@Coeffs_a[:,1])
opti.subject_to(b==dmx.T@Coeffs_b)

for p in range(c_max):                              # separating hyperplane problem
    if mob.nv==4:    
        for i in range(mob.nv):
            opti.subject_to(a[p,:]@mob.v[i]-b[p]>=-mob.r+safety)
        opti.subject_to(a[p,:]@q[p,:].T-b[p]<=-r_veh)
    elif mob.nv == 1:
        opti.subject_to((q[p,0]-mob.v[0])**2+(q[p,1]-mob.v[1])**2>=(mob.r+r_veh+safety)**2)
    
opti.subject_to(a[:,0]**2+a[:,1]**2<=1)             # normalizing the n vector

# ------- mob 2 ----------
'''
Coeffs_a2 = opti.variable(nb,2)
a2= opti.variable(c_max,2)
Coeffs_b2 = opti.variable(nb,1)       
b2 = opti.variable(c_max,1)

opti.subject_to(a2[:,0]==dmx.T@Coeffs_a2[:,0])          # parameterized as b-splines
opti.subject_to(a2[:,1]==dmx.T@Coeffs_a2[:,1])
opti.subject_to(b2==dmx.T@Coeffs_b2)

for p in range(c_max):  
    if mob2.nv==4:    
        for i in range(mob2.nv):
            opti.subject_to(a2[p,:]@mob2.v[i]-b2[p]>=mob2.r+safety)
        opti.subject_to(a2[p,:]@q[p,:].T-b2[p]<=-r_veh)
    elif mob2.nv == 1:
        opti.subject_to((q[p,0]-mob2.v[0])**2+(q[p,1]-mob2.v[1])**2>=(mob2.r+r_veh+safety)**2)
        #opti.subject_to((q[p,1]-mob.v[1])**2>=(mob.r-r_veh)**2)   
  
opti.subject_to(a2[:,0]**2+a2[:,1]**2<=1)             # normalizing the n vector
'''

# ------- time constraint --------
opti.subject_to(T>0)            # T should be bigger than 0

# ------- initial values for solver -------        # can help where to search for the solution -> putting the coeffs on a line
priority = 0
dx_lin = (q_end[0]-q_start[0])/(nb-priority)
dy_lin = (q_end[1]-q_start[1])/(nb-priority)
for i in range(nb):
    opti.set_initial(Coeffs[i,0],i*dx_lin)
    opti.set_initial(Coeffs[i,1],i*dy_lin)

# -------- solving the problem -------
p_opts = {"expand": True}
s_opts = {"max_iter": 5000}
opti.solver("ipopt",p_opts,s_opts)
sol = opti.solve()

# -------- extracting the solution --------
megoldas = sol.value(Coeffs)
tf = sol.value(T)
print("Az ossz ido:")
print(tf)

speed_x = sol.value(vel[:,0])
speed_y = sol.value(vel[:,1])
#coeffs_a = sol.value(Coeffs_a)
#coeffs_b =sol.value(Coeffs_b)

# -------- plotting --------
fig,bx= plt.subplots(1,2,figsize=(12, 4))
xx = np.linspace(0,1,N+1)

# -------- x spline --------
coeffs_x = sol.value(Coeffs[:,0])
bsp_x = scint.BSpline(knots, coeffs_x,k)
bx[0].plot(bsp_x(xx),xx )
bx[0].set_title('Az X komponens')
bx[0].set_ylabel('S paraméter [-]', loc='center')
bx[0].set_xlabel('Az X értéke [m]', loc='center')

# -------- y spline --------
coeffs_y = sol.value(Coeffs[:,1])
bsp_y = scint.BSpline(knots, coeffs_y,k)
bx[1].plot(xx,bsp_y(xx))
bx[1].set_title('Az Y komponens')
bx[1].set_xlabel('S paraméter [-]', loc='center')
bx[1].set_ylabel('Az Y értéke [m]', loc='center')

# -------- acceleration --------
# x component
fig,lx= plt.subplots(1,2,figsize=(12, 4))
t_x = scint.BSpline(knots, coeffs_x,k)
print(t_x)
acc_x = scint.splder(t_x,2)
lx[0].plot(xx,acc_x(xx))
lx[0].set_title('Az X gyorsulása')
lx[0].set_xlabel('S paraméter [-]', loc='center')
lx[0].set_ylabel('Az Y értéke [m]', loc='center')
# y components
t_y = scint.BSpline(knots, coeffs_y,k)
acc_y = scint.splder(t_y,2)
lx[1].plot(xx,acc_y(xx))
lx[1].set_title('Az Y gyorsulása')
lx[1].set_xlabel('S paraméter [-]', loc='center')
lx[1].set_ylabel('Az Y értéke [m]', loc='center')

# -------- plotting the path --------
fig, cx = plt.subplots()
cx.axis('equal')
cx.plot(bsp_x(xx),bsp_y(xx),'--k',lw = '1.5')  
cx.set_title('A kettő együtt')
cx.set_xlabel('X értéke [m]', loc='center')
cx.set_ylabel('Y értéke [m]', loc='center')   
if mob.nv == 4:
            rajz = patches.Rectangle(mob.csucs, mob.w,mob.h, facecolor='r')
            cx.add_patch(rajz)
elif mob.nv ==1:
            rajz = patches.Circle(mob.v, mob.r, facecolor='r')
            cx.add_patch(rajz)

# derivates
fig, dx = plt.subplots(1,2,figsize=(12, 4))

dx[0].plot(c*tf,g.T@P@coeffs_x,"y",lw="2")
dx[0].set_title('Az X komponens sebessége')
dx[0].set_xlabel('Az idő [s]', loc='center')
dx[0].set_ylabel('A sebesség [m/s]', loc='center')   

h = coeffs_y/tf
dx[1].plot(c*tf,speed_y,"y",lw="2")
dx[1].set_title('Az Y komponens sebessége')
dx[1].set_xlabel('Az idő [s]', loc='center')
dx[1].set_ylabel('A sebesség [m/s]', loc='center')   

bx[0].grid(color='k', linestyle=':', linewidth=0.5)
bx[1].grid(color='k', linestyle=':', linewidth=0.5)
cx.grid(color='k', linestyle=':', linewidth=0.5)
dx[0].grid(color='k', linestyle=':', linewidth=0.5)
dx[1].grid(color='k', linestyle=':', linewidth=0.5)

# feladatok:    b-spline relaxation -> limiting coeffs insted of the velocity/accelaration
#               how to make the second derivate of the spline with the same coeffs
#               moving obstacles -> needs B-spline relaxation

animate_scene(mobs)


