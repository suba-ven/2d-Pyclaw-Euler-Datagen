import numpy as np
from scipy import integrate
from scipy.optimize import minimize, differential_evolution

from clawpack import riemann, pyclaw
from clawpack.riemann.euler_5wave_2D_constants import density, x_momentum, y_momentum, energy, num_eqn

from SBIConfig import SBIWarmStartDataConfig

from time import time

class ShockBubbleInteractionVarEst():
    def __init__(self, config:SBIWarmStartDataConfig, state_params_list=[]):
        """Initialize the ShockBubbleInteraction class.

        Args:
            config (ShockBubbleInteractionConfig): Default parameters for the simulation
            state_params_list (list, string): List of the names of uncertain parameters. Defaults to [].
        """
        self.t_final = config.t_final
        self.n_steps = config.n_steps
        
        self.x = pyclaw.Dimension(config.x_domain[0], config.x_domain[1], config.nx, name='x')
        self.y = pyclaw.Dimension(config.y_domain[0], config.y_domain[1], config.ny, name='y')
        self.domain = pyclaw.Domain([self.x, self.y])
        self.num_aux = 1
        self.order = ["density", "x_momentum", "y_momentum", "energy", "tracer"]
        
        self.state_params_list = state_params_list
        self.default_params_dict = {
            "x0": config.x0,
            "y0": config.y0,
            "r0": config.r0,
            "gamma": config.gamma,
            "rhoin": config.rhoin,
            "pinf": config.pinf,
            "rhoout": config.rhoout,
            "pout": config.pout,
            "pin": config.pin,
            "xshock": config.xshock
        }
        
        self.threshold = config.threshold
        #self.ll_eps = config.ll_eps
        #self.gaussian_filter_sigma = config.gaussian_filter_sigma
        
        self.obs_start_idx = config.obs_start_idx
               
    def forward_model(self, state_params):
        """ Run the forward model with the given state parameters.

        Args:
            state_params (list): list of state parameters

        Returns:
            list: list of frames from the solution
        """
        temp_params = self.default_params_dict.copy()
        temp_params.update(dict(zip(self.state_params_list, state_params)))
        
        x0 = temp_params["x0"]
        y0 = temp_params["y0"]
        r0 = temp_params["r0"]
        gamma = temp_params["gamma"]
        rhoin = temp_params["rhoin"]
        pinf = temp_params["pinf"]
        rhoout = temp_params["rhoout"]
        pout = temp_params["pout"]
        pin = temp_params["pin"]
        xshock = temp_params["xshock"]
        
        # Helper functions
        def ycirc(x,ymin,ymax):
            if ((x-x0)**2)<(r0**2):
                return max(min(y0 + np.sqrt(r0**2-(x-x0)**2),ymax) - ymin,0.)
            else:
                return 0
            
        def qinit(state):
            gamma1 = gamma - 1.

            grid = state.grid

            rinf = (gamma1 + pinf*(gamma+1.))/ ((gamma+1.) + gamma1*pinf)
            vinf = 1./np.sqrt(gamma) * (pinf - 1.) / np.sqrt(0.5*((gamma+1.)/gamma) * pinf+0.5*gamma1/gamma)
            einf = 0.5*rinf*vinf**2 + pinf/gamma1
            
            X, Y = grid.p_centers

            r = np.sqrt((X-x0)**2 + (Y-y0)**2)

            #First set the values for the cells that don't intersect the bubble boundary
            state.q[0,:,:] = rinf*(X<xshock) + rhoin*(r<=r0) + rhoout*(r>r0)*(X>=xshock)
            state.q[1,:,:] = rinf*vinf*(X<xshock)
            state.q[2,:,:] = 0.
            state.q[3,:,:] = einf*(X<xshock) + (pin*(r<=r0) + pout*(r>r0)*(X>=xshock))/gamma1
            state.q[4,:,:] = 1.*(r<=r0)

            #Now compute average density for the cells on the edge of the bubble
            d2 = np.linalg.norm(state.grid.delta)/2.
            dx = state.grid.delta[0]
            dy = state.grid.delta[1]
            dx2 = state.grid.delta[0]/2.
            dy2 = state.grid.delta[1]/2.
            for i in range(state.q.shape[1]):
                for j in range(state.q.shape[2]):
                    ydown = Y[i,j]-dy2
                    yup   = Y[i,j]+dy2
                    if abs(r[i,j]-r0)<d2:
                        infrac,abserr = integrate.quad(ycirc,X[i,j]-dx2,X[i,j]+dx2,args=(ydown,yup),epsabs=1.e-8,epsrel=1.e-5)
                        infrac=infrac/(dx*dy)
                        state.q[0,i,j] = rhoin*infrac + rhoout*(1.-infrac)
                        state.q[3,i,j] = (pin*infrac + pout*(1.-infrac))/gamma1
                        state.q[4,i,j] = 1.*infrac
                        
        def auxinit(state):
            """
            aux[1,i,j] = radial coordinate of cell centers for cylindrical source terms
            """
            y = state.grid.y.centers
            for j,r in enumerate(y):
                state.aux[0,:,j] = r

        def incoming_shock(state,dim,t,qbc,auxbc,num_ghost):
            """
            Incoming shock at left boundary.
            """
            gamma1 = gamma - 1.

            pinf=5.
            rinf = (gamma1 + pinf*(gamma+1.))/ ((gamma+1.) + gamma1*pinf)
            vinf = 1./np.sqrt(gamma) * (pinf - 1.) / np.sqrt(0.5*((gamma+1.)/gamma) * pinf+0.5*gamma1/gamma)
            einf = 0.5*rinf*vinf**2 + pinf/gamma1

            for i in range(num_ghost):
                qbc[0,i,...] = rinf
                qbc[1,i,...] = rinf*vinf
                qbc[2,i,...] = 0.
                qbc[3,i,...] = einf
                qbc[4,i,...] = 0.
                
        def step_Euler_radial(solver,state,dt):
            """
            Geometric source terms for Euler equations with cylindrical symmetry.
            Integrated using a 2-stage, 2nd-order Runge-Kutta method.
            This is a Clawpack-style source term routine, which approximates
            the integral of the source terms over a step.
            """
            dt2 = dt/2.

            q = state.q
            rad = state.aux[0,:,:]

            rho = q[0,:,:]
            u   = q[1,:,:]/rho
            v   = q[2,:,:]/rho
            press  = (gamma - 1.) * (q[3,:,:] - 0.5*rho*(u**2 + v**2))

            qstar = np.empty(q.shape)

            qstar[0,:,:] = q[0,:,:] - dt2/rad * q[2,:,:]
            qstar[1,:,:] = q[1,:,:] - dt2/rad * rho*u*v
            qstar[2,:,:] = q[2,:,:] - dt2/rad * rho*v*v
            qstar[3,:,:] = q[3,:,:] - dt2/rad * v * (q[3,:,:] + press)

            rho = qstar[0,:,:]
            u   = qstar[1,:,:]/rho
            v   = qstar[2,:,:]/rho
            press  = (gamma - 1.) * (qstar[3,:,:] - 0.5*rho*(u**2 + v**2))

            q[0,:,:] = q[0,:,:] - dt/rad * qstar[2,:,:]
            q[1,:,:] = q[1,:,:] - dt/rad * rho*u*v
            q[2,:,:] = q[2,:,:] - dt/rad * rho*v*v
            q[3,:,:] = q[3,:,:] - dt/rad * v * (qstar[3,:,:] + press)
        
        
        state = pyclaw.State(self.domain, num_eqn, self.num_aux)
        state.problem_data['gamma']= gamma
    
        qinit(state)
        auxinit(state)
        
        # Solver setup and solution    
        solver = pyclaw.ClawSolver2D(riemann.euler_5wave_2D)
        solver.step_source = step_Euler_radial
        solver.source_split = 1
        solver.limiters = [4,4,4,4,2]
        solver.cfl_max = 0.5
        solver.cfl_desired = 0.45
        solver.user_bc_lower = incoming_shock
        solver.bc_lower[0]=pyclaw.BC.custom
        solver.bc_upper[0]=pyclaw.BC.extrap
        solver.bc_lower[1]=pyclaw.BC.wall
        solver.bc_upper[1]=pyclaw.BC.extrap
        #Aux variable in ghost cells doesn't matter
        solver.aux_bc_lower[0]=pyclaw.BC.extrap
        solver.aux_bc_upper[0]=pyclaw.BC.extrap
        solver.aux_bc_lower[1]=pyclaw.BC.extrap
        solver.aux_bc_upper[1]=pyclaw.BC.extrap

        claw = pyclaw.Controller()
        claw.solution = pyclaw.Solution(state, self.domain)
        claw.solver = solver
        
        claw.keep_copy = True
        claw.output_format = None
        claw.tfinal = self.t_final
        claw.num_output_times = self.n_steps
        claw.verbosity = 1
        
        claw.run()
        return claw.frames
            
    def observation_operator(self, claw_frames, density=True):
        """ Returns a binary array of shock positions in the density field.

        Args:
            claw_frames (list): list of frames from the solution

        Returns:
            (old) np.ndarray: binary array of shock positions in density field
            (new) np.ndarray: array of continuous density field
        """
        density_frame = np.array([frame.q[0] for frame in claw_frames[self.obs_start_idx:]])
        vx, vy = np.gradient(density_frame, axis=-2), np.gradient(density_frame, axis=-1)
        grad = np.sqrt(vx**2 + vy**2)
        if density:
            return density_frame #density_frame
        return (grad > self.threshold).astype(float) # else, returns shock location
    
print('shock')