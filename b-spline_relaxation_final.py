from casadi import *
import casadi as ca
import scipy as sc
import scipy.interpolate as scint
import matplotlib.pyplot as plt
import numpy as np
import bspline.splinelab as splinelab
import matplotlib.animation as animation
import matplotlib.patches as patches
import random 

# functions to fulfill my demands 
def zerocheck(a):
        '''
        Return 1 insted of zero. Helps to prevent dividing with zero.
        '''
        return ca.if_else(a == 0, 1, a)

def DerivCollMx(knots, k, tau): 
    '''
    Evaluates the new B-spline basis functions of the B-spline's first and second derivatives defined by (knots, k) at tau.
    Based on: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html

    Parameters:
        knots (list or np.array): Knot vector for the B-spline.
        k (int): Order of the B-spline.
        tau (list or np.array): Points at which to evaluate the basis functions.
    
    Returns:
        base_1 (ca.DM or np.array): Evaluated matrix of the new B-spline basis functions of the first derivative at tau.
        base_2 (ca.DM or np.array): Evaluated matrix of the new B-spline basis functions of the second derivative at tau.
    '''
    tau = np.transpose(tau)
    n = len(tau)
    m = len(knots)
    k = k+1
    
    N = np.zeros((m,k,n))
    N_first_d = np.zeros((m,k,n)) 
    N_sec_d = np.zeros((m,k,n))
    for t_idx in range(n): 
        for l in range(1, k + 1): 
            for i in range(m - l): 
                if l == 1:
                    N[i, l - 1, t_idx] = ca.if_else((knots[i] <= tau[t_idx]) & (tau[t_idx] <= knots[i + 1]), 1, 0)
                else:
                    term1 = (tau[t_idx] - knots[i]) / zerocheck(knots[i + l - 1] - knots[i])
                    term2 = (knots[i + l] - tau[t_idx]) / zerocheck(knots[i + l] - knots[i + 1])

                    term1 = term1 * N[i, l - 2, t_idx]
                    term2 = term2 * N[i + 1, l - 2, t_idx]
                    N[i, l - 1, t_idx] = ca.if_else(term1+term2 > 1, (term1+term2)/2, term1+term2) 

                    N_first_d [i,l-1,t_idx] = N[i + 1, l - 2, t_idx]
                    if l>=2:
                        N_sec_d[i,l-1,t_idx] = N[i + 2, l - 3, t_idx]    

                    
    base_1 = np.array([N_first_d[0:m - k-1, k - 1, t_idx] for t_idx in range(n)])
    base_2 = np.array([N_sec_d[0:m - k-2, k - 1, t_idx] for t_idx in range(n)])
    return base_1,base_2

def bscol(knots, k, tau):
    """
    Evaluates the B-spline basis functions defined by (knots, k) at tau 
    and evalutes the first and second derivatives' 
    
    Parameters:
        knots (list or np.array): Knot vector for the B-spline.
        k (int): Order of the B-spline.
        tau (list or np.array): Points at which to evaluate the basis functions.
    
    Returns:
        col (ca.DM or np.array): Evaluated B-spline basis functions at tau.
        col_der (ca.DM or np.array): Evaluated B-spline basis functions at tau.
        col_der2 (ca.DM or np.array): Evaluated B-spline basis functions at tau.
    """
    tau = np.transpose(tau)
    n = len(tau)
    m = len(knots)
    k = k+1

    N = np.zeros((m,k,n))
    #N_deriv = np.zeros((m,k,n)) 
    #N_deriv_2 = np.zeros((m,k,n)) 
   
    for t_idx in range(n):
        for l in range(1, k + 1):
            for i in range(m - l):
                if l == 1:
                    N[i, l - 1, t_idx] = ca.if_else((knots[i] <= tau[t_idx]) & (tau[t_idx] <= knots[i + 1]), 1, 0)
                else:
                    term1 = (tau[t_idx] - knots[i]) / zerocheck(knots[i + l - 1] - knots[i])
                    term2 = (knots[i + l] - tau[t_idx]) / zerocheck(knots[i + l] - knots[i + 1])

                    term1 = term1 * N[i, l - 2, t_idx]
                    term2 = term2 * N[i + 1, l - 2, t_idx]

                    #term_d_1 = (l-1) / zerocheck(knots[i + l - 1] - knots[i])
                    #term_d_2 = (l-1) / zerocheck(knots[i + l] - knots[i + 1])

                    #term_d_1 = term_d_1 * N[i, l - 2, t_idx]
                    #term_d_2 = term_d_2 * N[i + 1, l - 2, t_idx]
                    #if l>=2:
                         
                        #term_d2_1 = (l-1) / zerocheck(knots[i + l - 1] - knots[i])
                        #term_d2_2 = (l-1) / zerocheck(knots[i + l] - knots[i + 1])

                        #term_d2_1 = term_d2_1 * N_deriv[i, l - 2, t_idx]
                        #term_d2_2 = term_d2_2 * N_deriv[i + 1, l - 2, t_idx]
                        #N_deriv_2[i, l - 1, t_idx] = term_d2_1-term_d2_2

                    N[i, l - 1, t_idx] = term1 + term2
                    #N_deriv[i, l - 1, t_idx] = term_d_1-term_d_2
                    
    col = np.array([N[0:m - k, k - 1, t_idx] for t_idx in range(n)])
    #col_der = np.array([N_deriv[0:m - k, k - 1, t_idx] for t_idx in range(n)])
    #col_der2 = np.array([N_deriv_2[0:m - k, k - 1, t_idx] for t_idx in range(n)])

    if n == 1:
        col = col.T

    return ca.horzcat(*col) if n > 1 else col.flatten()#, ca.horzcat(*col_der) if n > 1 else col_der.flatten(), ca.horzcat(*col_der2) if n > 1 else col_der2.flatten()

class Obstacle: # parent of the obstacles
    def __init__(self,x,y):
        self.cp = [x,y]

class Circle (Obstacle):
    '''
    Class for the circular obstacles. 
    '''
    def __init__ (self,x,y,r,vel=0.5):
        self.nv =1                  # number of vertices -> in this case the center point
        self.v =np.array([x,y])     # the vertice's coordinate
        self.r = r                  # radius
        self.vel = vel              # velocity

class Rectangle (Obstacle):    
    '''
    Rectangular obstacles.
    '''
    def __init__(self,x,y,w,h,vel,knumber):
        self.nv = 4                 # number of vertices
        self.v = np.array([[x-w/2,y-h/2],[x-w/2,y+h/2],[x+w/2,y+h/2],[x+w/2,y-h/2]])    # the coordinates of the vertices
        self.r = 0                  # radius
        self.point = [x-w/2,y-h/2]  # left corner's coordinate, for the plotting
        self.w = w                  # width of the rectangle
        self.h = h                  # height of the rectangle
        self.vel = vel              # velocity
        self.coeffs_a = opti.variable(knumber,2)    # hyperplane's coefficients
        self.coeffs_b = opti.variable(knumber)      # hyperplane's offset
 
def openfig(): 
    '''
    Making an animation plot scene with the correct boundaries. 
    '''
    fig, ax = plt.subplots()
    ax.grid(color='k', linestyle=':', linewidth=0.5)
    ax.plot(q_start[0],q_start[1],'xr', ms ="10")
    ax.plot(q_end[0],q_end[1],'xr', ms ="10")
    ax.set(xlim=(bounds[0][0], bounds[0][1]), ylim=(bounds[1][0], bounds[1][1]))
    ax.axis('equal')
    return fig, ax

def animate_scene(mobs): # plotting the obstacles for the animation
    fig, vx = openfig()
    vx.set_title("Trajectory animation")
    mobs_art = []
    for i in range(0, len(mobs)): # plotting the mobs
        if mobs[i].nv == 4:
            mobs_art.append(plt.Rectangle(mobs[i].point, mobs[i].w,mobs[i].h, facecolor='r'))
            vx.add_patch(mobs_art[i])
        elif mobs[i].nv ==1:
            mobs_art.append(plt.Circle(mobs[i].v, mobs[i].r, facecolor='r'))
            vx.add_patch(mobs_art[i])
    
    ani_t = 200                 # the animation's resolution
    ani_x = np.zeros(ani_t)     
    ani_y = np.zeros(ani_t)
    for i in range(ani_t):      # the bsplines' points are collected into arrays
        ani_x[i] = bsp_x(i/ani_t)
        ani_y[i] = bsp_y(i/ani_t)

    drone = plt.Circle(xy=(ani_x[0],ani_y[0]),radius=r_veh,ec='k',color="b")
    s_dist = plt.Circle(xy=(ani_x[0],ani_y[0]),radius=r_veh+safety,ec='b',color="none",ls='--')
    line, = vx.plot([],[],color='k', linestyle='--', linewidth=1)
    vx.add_patch(s_dist)
    vx.add_patch(drone)

    '''
    #separating hyperplane
    for i in range(len(mobs)):
        nvector = amx.T@sol.value(mobs[i].coeffs_a)
    
    oset =  scint.BSpline(knots_h,sol.value(mobs[0].coeffs_b),k_h)
    print("Az offset:")
    test= linspace(0,1,100)
    print(oset(test))
    x = np.linspace(q_start[0]-3,q_end[0]+3,100)
    nspline_x = scint.BSpline(knots_h,sol.value(mobs[0].coeffs_a[:,0]),k_h)
    nspline_y = scint.BSpline(knots_h,sol.value(mobs[0].coeffs_a[:,1]),k_h)
    h_plane, =  vx.plot([],[],color='r',marker='o',linewidth=1)

    '''
    def update_data(frame): 
        '''
        Function, which will update the animation in each frame
        '''    
        #path
        line.set_data([ani_x[:frame]],[ani_y[:frame]])

        #drone
        drone.set_center(xy=(bsp_x(frame/ani_t),ani_y[frame]))
        s_dist.set_center(xy=(ani_x[frame],ani_y[frame]))

        # obstacles
        for i in range(len(mobs)):
            if mobs[i].nv==4:
                moved =[move[i][0]/ani_t*frame*tf+mobs[i].point[0],move[i][1]/ani_t*frame*tf+mobs[i].point[1]]
                mobs_art[i].set_xy(moved)
            else:
                moved =[move[i][0]/ani_t*frame*tf+mobs[i].v[0],move[i][1]/ani_t*frame*tf+mobs[i].v[1]]
                mobs_art[i].set_center(moved)
        '''
        # hyperplane -> inclomplete
        h_plane.set_data([ani_x[frame]+nspline_x(frame/ani_t)*(safety+r_veh)],[ani_y[frame]+nspline_y(frame/ani_t)*(safety+r_veh)])
        print(nspline_x(frame/ani_t)*oset(frame/ani_t))
        '''
    anim = animation.FuncAnimation(    
                        fig = fig,
                        func=update_data,
                        frames = ani_t,
                        interval = 2
                        )
    #anim.save("optimization_2D.gif","ffmpeg",60,150) 
    plt.show()


# -------- points: starting point and goal --------
q_start = [0,0]         # starting point in [m]
q_end = [3,3]           # goal in [m]
bounds = [[-1,5],[-1,5]]
r_veh = 0.2           # radius of the drone [m]
safety = 1e-1       # safety distance between the obtsacle and the drone

# -------- B-spline allocation --------
nknots = 10
k = 3   # degree of the spline
knots_0 = np.linspace(0,1,nknots)
knots = splinelab.augknt(knots_0,k) # knots 
nb = nknots + k-1                     # number of basis functions

# -------- Matrixes for the b-splines ------
div=30+ 1                    # number of the points where we calculate the b-splines -> minimal time gridding
c = np.linspace(0,1,div)
dmx = bscol(knots, k, c)
base,base_2 = DerivCollMx(knots,k,c)

# -------- B-spline allocation for the hyperplane --------
nknots_h = 12
k_h = 2                                    # degree of the spline
knots_0_h = np.linspace(0,1,nknots_h)
knots_h = splinelab.augknt(knots_0_h,k_h)   # knots 
nb_h = nknots_h + k_h-1                     # number of basis functions
amx = bscol(knots_h, k_h, c)

# ------- optimization problem -------
opti = Opti()

# -------- initialization of the obstacles -------
mobs = []           # mobs and move should be the same length 
move = []

'''
mob = Circle(3,3,0.2,0.2) # declare the obstacle
mobs.append(mob)
move.append([0,mob.vel])
'''
'''
for i in range(30):
    r = random.uniform(0, 3)
    xpo = random.uniform(bounds[0][0], bounds[0][1])
    ypo = random.uniform(bounds[1][0], bounds[1][1])
    mob = Circle(xpo,ypo,0.2,r) 
    mobs.append(mob)
    pr= random.uniform(-1,1)
    pv= random.uniform(-1,1)
    move.append([mob.vel*pr,mob.vel*pv])
  '''




'''
mob = Rectangle(0,-1.2,2,0.2,0,nb_h) 
mobs.append(mob)
move.append([0,0])

mob = Rectangle(-1,0,0.2,2,0,nb_h) 
mobs.append(mob)
move.append([0,0])

'''

for i in range(12):
    r = random.uniform(0, 3)
    xpo = random.uniform(bounds[0][0], bounds[0][1])
    ypo = random.uniform(bounds[1][0], bounds[1][1])
    mob = Rectangle(xpo,ypo,0.2,0.2,r,nb_h)
    mobs.append(mob)
    pr= random.uniform(-1,1)
    pv= random.uniform(-1,1)
    move.append([mob.vel*pr,mob.vel*pv])
  

# ------- decision variables ------
Coeffs = opti.variable(nb,2)        # coefficients -> number of basis functions are given
T = opti.variable()                 # final time

# ------- objective --------
opti.minimize(T)

# ------- kinematic constrains --------
# position
opti.subject_to(opti.bounded(bounds[0][0]+r_veh,Coeffs[:,0],bounds[0][1]-r_veh))      # bounding where it should stay -> boundaries of the map
opti.subject_to(opti.bounded(bounds[1][0]+r_veh,Coeffs[:,1],bounds[1][1]-r_veh))    

# limiting coefficients
opti.subject_to(Coeffs[0,0]==q_start[0])                        # first and last coeffs should be the starting point and the end point
opti.subject_to(Coeffs[0,1]==q_start[1])    
opti.subject_to(Coeffs[-1,0]==q_end[0])
opti.subject_to(Coeffs[-1,1]==q_end[1])

# velocity
vel_max = 10                    # velocity limitations -----> m/s
vel_min = 10
acc_max = 50                    # acceleration limitations: m/s^2
acc_min = 50

# limitations
'''
for i in range(1,nb): # limiting velocity components
    opti.subject_to(opti.bounded(-vel_min*T,(Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]),vel_max*T))
    opti.subject_to(opti.bounded(-vel_min*T,(Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]),vel_max*T))

for i in range(2,nb): # limiting acceleration components
    opti.subject_to(opti.bounded(-acc_min*T**2,
        (((Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,1]-Coeffs[i-2,1])*k/zerocheck(knots[i+k-1]-knots[i-1])))*(k-1)/zerocheck(knots[i+k-1]-knots[i]),
        acc_max*T**2))
    opti.subject_to(opti.bounded(-acc_min*T**2,
        (((Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,0]-Coeffs[i-2,0])*k/zerocheck(knots[i+k-1]-knots[i-1])))*(k-1)/zerocheck(knots[i+k-1]-knots[i]),
        acc_max*T**2))
  
'''
for i in range(1,nb-1): # limiting velocity
    opti.subject_to(((Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]))**2+((Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]))**2<=(vel_max*T)**2)

for i in range(2,nb):   # # limiting acceleration
    opti.subject_to(
        ((((Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,0]-Coeffs[i-2,0])*k/zerocheck(knots[i+k-1]-knots[i])))*(k-1)/zerocheck(knots[i+k-2]-knots[i-1]))**2+\
        ((((Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,1]-Coeffs[i-2,1])*k/zerocheck(knots[i+k-1]-knots[i])))*(k-1)/zerocheck(knots[i+k-2]-knots[i-1]))**2<=
        (acc_max*T**2)**2)

# 0 velocity from the sarting position and at the destination   
opti.subject_to((Coeffs[1,0]-Coeffs[0,0])*k/zerocheck(knots[1+k]-knots[1])==0)
opti.subject_to((Coeffs[1,1]-Coeffs[0,1])*k/zerocheck(knots[1+k]-knots[1])==0)
opti.subject_to((Coeffs[-1,0]-Coeffs[-2,0])*k/zerocheck(knots[nb+k]-knots[nb])==0)
opti.subject_to((Coeffs[-1,1]-Coeffs[-2,1])*k/zerocheck(knots[nb+k]-knots[nb])==0)

# 0 acceleration from the starting position and at the destination   
opti.subject_to((((Coeffs[2,0]-Coeffs[1,0])*k/zerocheck(knots[2+k]-knots[2]))-((Coeffs[1,0]-Coeffs[0,0])*k/zerocheck(knots[2+k-1]-knots[1])))*(k-1)/zerocheck(knots[2+k-1]-knots[2]) == 0)
opti.subject_to((((Coeffs[2,1]-Coeffs[1,1])*k/zerocheck(knots[2+k]-knots[2]))-((Coeffs[1,1]-Coeffs[0,1])*k/zerocheck(knots[2+k-1]-knots[1])))*(k-1)/zerocheck(knots[2+k-1]-knots[2]) == 0)
opti.subject_to((((Coeffs[-1,0]-Coeffs[-2,0])*k/zerocheck(knots[nb+k]-knots[nb]))-((Coeffs[-2,0]-Coeffs[-3,0])*k/zerocheck(knots[nb+k-1]-knots[nb-1])))*(k-1)/zerocheck(knots[nb+k-1]-knots[nb-1]) == 0)
opti.subject_to((((Coeffs[-1,1]-Coeffs[-2,1])*k/zerocheck(knots[nb+k]-knots[nb]))-((Coeffs[-2,1]-Coeffs[-3,1])*k/zerocheck(knots[nb+k-1]-knots[nb-1])))*(k-1)/zerocheck(knots[nb+k-1]-knots[nb-1]) == 0)

# ------- obstacle avoidance --------
for l in range(len(mobs)):
    for p in range(div):                              # separating hyperplane theorem
        if mobs[l].nv==4:    
            for i in range(mobs[l].nv):
                opti.subject_to(amx.T[p,:]@mobs[l].coeffs_a@(mobs[l].v[i]+move[l]@(p*T/div))-amx.T[p,:]@mobs[l].coeffs_b>=safety)
            opti.subject_to((amx.T[p,:]@mobs[l].coeffs_a)@(dmx.T[p,:]@Coeffs[:,:]).T-amx.T[p,:]@mobs[l].coeffs_b<=-r_veh)
        
        elif mobs[l].nv == 1:
            opti.subject_to((dmx.T[p,:]@Coeffs[:,0]-(mobs[l].v[0]+move[l][0]*p*T/div))**2+(dmx.T[p,:]@Coeffs[:,1]-(mobs[l].v[1]+move[l][1]*p*T/div))**2>=(mobs[l].r+r_veh+safety)**2)
            
    if mobs[l].nv==4:
        for p in range(div):     
            if mobs[l].nv==4:    
                opti.subject_to(amx.T[p,:]@mobs[l].coeffs_a[:,0]**2+amx.T[p,:]@mobs[l].coeffs_a[:,1]**2==1)             # normalizing the n vector
    
# ------- time constraint --------
opti.subject_to(T>0)            

# ------- initial values for solver -------        
opti.set_initial(T,(sqrt((q_end[0]-q_start[0])**2+(q_end[1]-q_start[1])**2)/vel_max)+vel_max/acc_max)

#  putting the coeffs on a line
priority =0                         # piority can be changed to get trajectories from counter clockwise or anticlockwise
dx_lin = (q_end[0]-q_start[0])/(nb+priority)
dy_lin = (q_end[1]-q_start[1])/(nb+priority)
for i in range(nb):
    opti.set_initial(Coeffs[i,0],i*dx_lin)
    opti.set_initial(Coeffs[i,1],i*dy_lin)

# -------- solving the problem -------
p_opts = {"expand": True}
s_opts = {"max_iter": 5000}
opti.solver("ipopt",p_opts,s_opts)
sol = opti.solve()

# -------- extracting the solution --------
tf = sol.value(T)
coeffs_x = sol.value(Coeffs[:,0])
coeffs_y = sol.value(Coeffs[:,1])
print("Final time:")
print(tf)

# -------- Calculating the coefficients of the derivatives --------
# 1. derivative
Q_x = np.zeros(len(coeffs_x)-1)
Q_y = np.zeros(len(coeffs_y)-1)

for i in range(1,len(coeffs_x)):
    Q_x[i-1]=sol.value((Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]))/tf
for i in range(2,len(coeffs_y)):
    Q_y[i-1]=sol.value((Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]))/tf

# 2. derivative
Q_x2 = np.zeros(len(coeffs_x)-2)
Q_y2 = np.zeros(len(coeffs_y)-2)

for i in range(2,len(coeffs_x)):
    Q_x2[i-2]=sol.value((((Coeffs[i,0]-Coeffs[i-1,0])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,0]-Coeffs[i-2,0])*k/zerocheck(knots[i+k-1]-knots[i-1])))*(k-1)/zerocheck(knots[i+k-1]-knots[i]))/tf**2

for i in range(2,len(coeffs_y)):
    Q_y2[i-2]=sol.value((((Coeffs[i,1]-Coeffs[i-1,1])*k/zerocheck(knots[i+k]-knots[i]))-((Coeffs[i-1,1]-Coeffs[i-2,1])*k/zerocheck(knots[i+k-1]-knots[i-1])))*(k-1)/zerocheck(knots[i+k-1]-knots[i]))/tf**2

# -------- plotting --------
fig,bx= plt.subplots(1,2,figsize=(12, 4))
xx = np.linspace(0,1,1001)

# -------- x spline --------
bsp_x = scint.BSpline(knots, coeffs_x,k)
bx[0].plot(bsp_x(xx),xx )
bx[0].set_title('The X component')
bx[0].set_ylabel('S parameter [-]', loc='center')
bx[0].set_xlabel('Value of X [m]', loc='center')

# -------- y spline --------
bsp_y = scint.BSpline(knots, coeffs_y,k)
bx[1].plot(xx,bsp_y(xx))
bx[1].set_title('The Y component')
bx[1].set_xlabel('S parameter [-]', loc='center')
bx[1].set_ylabel('Value of Y [m]', loc='center')

# -------- plotting the path --------
fig, cx = plt.subplots()
cx.axis('equal')
cx.plot(bsp_x(xx),bsp_y(xx),'--k',lw = '1.5')  
cx.set_title('The trajectory')
cx.set_xlabel('Value of X [m]', loc='center')
cx.set_ylabel('Value of Y [m]', loc='center')
cx.plot(q_start[0],q_start[1],'xr', ms ="10")
cx.plot(q_end[0],q_end[1],'xr', ms ="10")  

for l in range(len(mobs)):
    if mobs[l].nv == 4:
                rajz = patches.Rectangle(mobs[l].point, mobs[l].w,mobs[l].h, facecolor='r')
                cx.add_patch(rajz)
    elif mobs[l].nv ==1:
                rajz = patches.Circle(mobs[l].v, mobs[l].r, facecolor='r')
                cx.add_patch(rajz)

# derivatives
fig, dx = plt.subplots(1,2,figsize=(12, 4))
# x component
dx[0].plot(c*tf,base@Q_x,"y",lw="2")
dx[0].set_title('Velocity of the X component')
dx[0].set_xlabel('Time [s]', loc='center')
dx[0].set_ylabel('Velocity [m/s]', loc='center')   
bsp_deriv = scint.splder(bsp_x,1)

# y component
dx[1].plot(c*tf,base@Q_y,"y",lw="2")
dx[1].set_title('Velocity of the Y component')
dx[1].set_xlabel('Time [s]', loc='center')
dx[1].set_ylabel('Velocity [m/s]', loc='center')  

# -------- acceleration --------
# x component
fig,lx= plt.subplots(1,2,figsize=(12, 4))

lx[0].plot(c*tf,base_2@Q_x2)
lx[0].set_title('Acceleration of X')
lx[0].set_xlabel('Time [s]', loc='center')
lx[0].set_ylabel('Acceleration of X [m/s^2]', loc='center')
bsp_deriv2 = scint.splder(bsp_x,2)
lx[0].plot(c*tf,bsp_deriv2(c)/tf**2)

# y components
lx[1].plot(c*tf,base_2@Q_y2)
lx[1].set_title('Acceleration of Y')
lx[1].set_xlabel('Time [s]', loc='center')
lx[1].set_ylabel('Acceleration of Y [m/s^2]', loc='center')

bx[0].grid(color='k', linestyle=':', linewidth=0.5)
bx[1].grid(color='k', linestyle=':', linewidth=0.5)
cx.grid(color='k', linestyle=':', linewidth=0.5)
dx[0].grid(color='k', linestyle=':', linewidth=0.5)
dx[1].grid(color='k', linestyle=':', linewidth=0.5)
lx[0].grid(color='k', linestyle=':', linewidth=0.5)
lx[1].grid(color='k', linestyle=':', linewidth=0.5)

animate_scene(mobs)

