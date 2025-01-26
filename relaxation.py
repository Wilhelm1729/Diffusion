
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from copy import deepcopy
#import scipy
import sys
import time


class Diffusion():

    def __init__(self, gridsize, D, canal_height = 4, canal_width = 8, canal_wall_thickness = 2, x_padding = 10, y_padding = 20, sample_map=False):
        """
            Initialize: Map, Boundary_type

            Map has the values
            Boundary_type index the boundary conditions:
                0 = out of bounds
                1 = in bounds
                2 = Dirichlet condition
                3 = Von Neumann condition
                4 = Robin condition
        
        """

        # Initialize material parameters
        self.dx = gridsize
        self.D = D
        self.dt = self.dx**2 / 4 / D

        # Robin boundary parameters
        self.K = 1
        self.V_0 = 10

        # Initialize map parameters
        self.h = canal_height
        self.w = canal_width
        self.r = canal_wall_thickness
        self.b = canal_wall_thickness
        self.y_padding = y_padding
        self.x_padding = x_padding

        # Initialize map
        self.sample_map = sample_map

        if self.sample_map:

            pw = int(self.h / self.dx)
            ph = int(self.w / self.dx)

            print("Map dimensions" + str(pw) + " " + str(ph))

            self.map = np.zeros((pw,ph))

            self.map[:,-1] = np.zeros(pw)
            self.map[0,:] = np.zeros(ph)
            self.map[-1,:] = np.zeros(ph)
            self.map[:,0] = np.zeros(pw)

            self.boundary_type = np.ones(self.map.shape)
            self.boundary_type[:,-1] = 2 * np.ones(pw)
            self.boundary_type[0,:] = 3 * np.ones(ph)
            self.boundary_type[-1,:] = 3 * np.ones(ph)
            self.boundary_type[:,0] = 4 * np.ones(pw)

        else:
            self.boundary_type = self._generate_geometry()
            self.map = np.zeros(self.boundary_type.shape)


        # Time evolution
        self.time = []
        self.mean_value = []
        self.sample_values = []
        self.fluxes = []

    def _generate_geometry(self, show=False):
        # Measurements in mm
        h = self.h
        w = self.w
        r = self.r
        b = self.b

        height = b + h + r + self.y_padding #self.height
        width = w + 4 * r + 2 * self.x_padding#self.width

        scale = 1 / self.dx #self.scale

        pheight = int(scale * height)
        pwidth = int(scale * width)

        matrix = np.ones((pwidth, pheight))

        def set_matrix(m, x, y, value, avoid_index=None):
            (xl, yl) = m.shape
            if 0 <= x and x < xl and 0 <= y and y < yl:
                if m[x,y] != avoid_index:
                    m[x,y] = value

        def put_solid(m, x, y):
            set_matrix(m, x, y, 0)
            set_matrix(m, x-1, y, 3, 0)
            set_matrix(m, x+1, y, 3, 0)
            set_matrix(m, x, y-1, 3, 0)
            set_matrix(m, x, y+1, 3, 0)
        
        def put_liquid(m, x, y):
            set_matrix(m, x, y, 0)
            set_matrix(m, x-1, y, 3, 0)
            set_matrix(m, x+1, y, 3, 0)
            set_matrix(m, x, y-1, 4, 0)
            set_matrix(m, x, y+1, 3, 0)
        
        for xx in range(0, pwidth):
            x = xx / scale - width / 2
            for yy in range(0, pheight):
                y = height - yy / scale
                # Check for cross section
                if y < h + r + b and abs(x) < w/2 + 2*r:

                    # Check for walls
                    if y < h + b and abs(x) > w/2 and abs(x) < w/2 + 2*r:
                        put_solid(matrix, xx, yy)
                        continue
                    # Check for floor
                    if y < b:
                        put_liquid(matrix, xx, yy)
                        continue
                    # Check for roundings
                    if (y - h - b) ** 2 + (x - r - w/2)** 2 < r**2 or (y - h - b) ** 2 + (x + r + w/2)** 2 < r**2:
                        put_solid(matrix, xx, yy)
                        continue
                    
                if yy == 0 or yy == pheight-1 or xx == 0 or xx == pwidth-1:
                    set_matrix(matrix, xx, yy, 2)
    
        if show == True:
            plt.imshow(matrix.transpose(), cmap='viridis', interpolation='nearest')
            plt.colorbar()  # Add a color bar to indicate the scale
            plt.title("Matrix Visualization")
            plt.show()

        return matrix

    def get_flux(self, matrix, q):
        """
            Calculates the flux through a cross section of the canal at height q in the canal
            where q is normalized between 0 and 1
        """

        if self.sample_map:
            y = int(matrix.shape[1] * q)
            flux = 0
            for x in range(matrix.shape[0]):
                flux += self.D * (matrix[x,y] - matrix[x,y-1])
            return flux
        else:
            scale = 1 / self.dx

            y = int(scale * (self.b + self.h * (1-q) + self.y_padding))

            xb = int(scale * (2 * self.r + self.x_padding))
            xe = int(scale * (2 * self.r + self.x_padding + self.w))

            flux = 0
            for x in range(xb, xe):

                flux += self.D * (matrix[x,y] - matrix[x,y-1]) #/ self.dx

            #print(y, xb, xe, flux)
            return flux

    def get_sample(self, matrix, q):
        """
            Returns a sample value at height q in the canal where q is normalized between 0 and 1
        """
        scale = 1 / self.dx
        width = self.w + 4 * self.r + 2 * self.x_padding
        return matrix[int(scale * width / 2), int(scale*(self.h * (1-q) + self.r + self.y_padding))]

    def get_mean(self, matrix, p = 0, q = 1):
        """
            Calculates and return the mean value of the concentration within the canal
        """
        if self.sample_map:
            #TODO Add p q support
            return matrix[1:-1,1:-1].mean()
        else:
            scale = 1 / self.dx

            xb = int(scale * (2 * self.r + self.x_padding))
            xe = int(scale * (2 * self.r + self.x_padding + self.w))

            yb = int(scale * (self.b + self.h * (1-q) + self.y_padding))
            ye = int(scale * (self.b + self.r + self.h * (1-p) + self.y_padding))

            mean_area = matrix[xb:xe+1,yb:ye+1]
            mask = self.boundary_type[xb:xe+1,yb:ye+1]
            boolean_mask = mask == 1

            filtered_elements = mean_area[boolean_mask]
            m = filtered_elements.mean()
            return m

    def relaxation(self, v):
        """
            Performs one relaxation step on the matrix v using the boundary data in self.boundary_type
        """
        vnew = np.zeros(self.map.shape)

        (xs, ys) = self.map.shape

        for x in range(0,xs):
            for y in range(0,ys):
                # Inbound relaxation
                if self.boundary_type[x,y] == 1:
                    vnew[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25

                # Dirichlet condition
                elif self.boundary_type[x,y] == 2:
                    vnew[x,y] = v[x,y]

                elif self.boundary_type[x,y] == 3 or self.boundary_type[x,y] == 4:
                    
                    ydir = 0
                    xdir = 0
                    # This does not work for complicated boundary conditions but will work for now
                    if x+1 < xs and self.boundary_type[x+1,y] == 1: xdir +=1
                    if x-1 >= 0 and self.boundary_type[x-1,y] == 1: xdir -= 1
                    if y+1 < ys and self.boundary_type[x,y+1] == 1: ydir += 1
                    if y-1 >= 0 and self.boundary_type[x,y-1] == 1: ydir -= 1
                
                    V_in = v[x + xdir,y + ydir]
                    
                    if self.boundary_type[x,y] == 3:
                        # Neumann condition
                        vnew[x,y] = V_in
                    else:
                        # Robin condition
                        vnew[x,y] = (V_in + self.K * self.dx * self.V_0) / (1 + self.K * self.dx)
        
        for x in range(0,xs):
            for y in range(0,ys):
                v[x,y] = vnew[x,y]


    def relaxation_GS(self, v):
        """
            Performs one (Gauss-Seidel) relaxation step on the matrix v using the boundary data in self.boundary_type
        """
        #vnew = np.zeros(self.map.shape)

        (xs, ys) = self.map.shape

        for x in range(0,xs):
            for y in range(0,ys):
                # Inbound relaxation
                if self.boundary_type[x,y] == 1:
                    v[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25

                # Dirichlet condition
                elif self.boundary_type[x,y] == 2:
                    v[x,y] = v[x,y]

                elif self.boundary_type[x,y] == 3 or self.boundary_type[x,y] == 4:
                    
                    ydir = 0
                    xdir = 0
                    # This does not work for complicated boundary conditions but will work for now
                    if x+1 < xs and self.boundary_type[x+1,y] == 1: xdir +=1
                    if x-1 >= 0 and self.boundary_type[x-1,y] == 1: xdir -= 1
                    if y+1 < ys and self.boundary_type[x,y+1] == 1: ydir += 1
                    if y-1 >= 0 and self.boundary_type[x,y-1] == 1: ydir -= 1
                
                    V_in = v[x + xdir,y + ydir]
                    
                    if self.boundary_type[x,y] == 3:
                        # Neumann condition
                        v[x,y] = V_in
                    else:
                        # Robin condition
                        v[x,y] = (V_in + self.K * self.dx * self.V_0) / (1 + self.K * self.dx)
        
        #for x in range(0,xs):
        #    for y in range(0,ys):
        #        v[x,y] = vnew[x,y]


    def animate_heatmap(self, nsteps=10000):
        """
            Animate the time evolution of the system.
        """
        v = deepcopy(self.map)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(v.transpose(), cmap=None, interpolation='nearest', vmin = 0, vmax=10)
        fig.colorbar(im)
        ax.set_title("Diffusion")
        step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        def update(step):
            print(step)

            if step > 0:
                self.relaxation(v)
            
            step_text.set_text(f'Time: {step * self.dt:.2f}')
            im.set_array(v.transpose())

            return im, step_text

        anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=1,  blit=True, repeat=False)
        plt.show()

    def animate(self, nsteps=10000):
        """
            Animate and plot the time evolution of the system and a sample value and some fluxes as different height
        """
        v = deepcopy(self.map)

        fig = plt.figure() # figsize=(10,5)

        gs = gridspec.GridSpec(2,2)

        # Plot of heatmap
        ax1 = fig.add_subplot(gs[1,:])
        im = ax1.imshow(v.transpose(), cmap=None, interpolation='nearest', vmin = 0, vmax=10)
        fig.colorbar(im, ax=ax1)
        ax1.set_title("Diffusion")
        step_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # Plot of specific value
        ax2 = fig.add_subplot(gs[0,0]) # To plot specific value
        ax3 = fig.add_subplot(gs[0,1]) # To plot fluxes

        line, = ax2.plot([], [], 'r-', label='value')
        line_f1, = ax3.plot([], [], 'r-', label='flux 1')
        line_f2, = ax3.plot([], [], 'b-', label='flux 2')
        line_f3, = ax3.plot([], [], 'g-', label='flux 3')

        # Data
        steps = []
        values = []
        f1_v = []
        f2_v = []
        f3_v = []

        # titles
        #ax2.set_title("Sample value")
        #ax2.set_xlabel("Step")
        #ax2.set_ylabel("Value")
        #ax3.set_title("Sample fluxes")
        #ax3.set_xlabel("Step")
        #ax3.set_ylabel("Flux")

        def init():
            ax2.set_xlim(0,nsteps)
            ax2.set_ylim(0,10)

            ax3.set_xlim(0,nsteps)
            ax3.set_ylim(0,120) # change to appropriate 
            
            ax2.legend()
            ax3.legend()
            return line, line_f1, line_f2, line_f3

        def update(step):
            print(step)

            if step > 0:
                self.relaxation(v)
            
            step_text.set_text(f'Time: {step * self.dt:.2f}')
            im.set_array(v.transpose())

            # Updates plots
            steps.append(step * self.dt)
            values.append(self.get_sample(v, 0.1))
            f1_v.append(self.get_flux(v, 0.1))
            f2_v.append(self.get_flux(v, 0.5))
            f3_v.append(self.get_flux(v, 0.9))
            # Sets data            
            line.set_data(steps, values)
            line_f1.set_data(steps, f1_v)
            line_f2.set_data(steps, f2_v)
            line_f3.set_data(steps, f3_v)
            # Updates xes limit dynamically
            ax2.set_xlim(0, max(steps) + 1)
            ax3.set_xlim(0, max(steps) + 1)

            p2_y_min = min(values)
            p2_y_max = max(values)

            ax2.set_ylim(p2_y_min - 0.5, p2_y_max + 0.5)

            p3_y_min = min(min(f1_v), min(f2_v), min(f3_v))
            p3_y_max = max(max(f1_v), max(f2_v), max(f3_v))
            ax3.set_ylim(p3_y_min - 0.5, p3_y_max + 0.5)

            #ax2.figure.canvas.draw_idle() # Makes sure the ticks are redrawn, better to disable blit
            #ax3.figure.canvas.draw_idle()

            return im, step_text, line, line_f1, line_f2, line_f3

        anim = animation.FuncAnimation(fig, update, frames=nsteps+1, init_func=init, interval=1,  blit=False, repeat=False)

        plt.tight_layout()
        plt.show()

    def run(self, nsteps, timemode = False):
        """
            Runs some timesteps and store the result in the object.
        """
        if timemode:
            nsteps = nsteps / self.dt
            print("Number of steps:" + str(nsteps))
        
        v = self.map

        self.fluxes.append([])
        self.fluxes.append([])
        self.fluxes.append([])

        step = 0

        # Loading bar parameters
        prefix='Progress:'
        length=50
        fill='█'

        while True:
            # Loading bar
            percent = f"{100 * (step / nsteps):.1f}"  # Percent completion
            filled_length = int(length * step // nsteps)  # Length of the filled portion
            bar = fill * filled_length + '-' * (length - filled_length)  # Build the bar
            sys.stdout.write(f"\r{prefix} |{bar}| {percent}%")  # Print the bar
            sys.stdout.flush()

            # Computation
            self.relaxation(v)

            self.time.append(step * self.dt)
            #sample_value.append(self.get_sample(v, 0.1))
            self.mean_value.append(self.get_mean(v))
            self.fluxes[0].append(self.get_flux(v, 0.1))
            self.fluxes[1].append(self.get_flux(v, 0.5))
            self.fluxes[2].append(self.get_flux(v, 0.9))

            step += 1

            if step > nsteps:
                break

        print()
        self.map = v
        print("Finished running")
    

    def run_till_steady_state(self, tolerance):
        """
            Runs some timesteps and store the result in the object.
        """
        v = self.map

        self.fluxes.append([])
        self.fluxes.append([])
        self.fluxes.append([])

        step = 0

        while True:
            # Computation
            if step % 100 == 0:
                print(step)

            previous = deepcopy(v)

            self.relaxation(v)

            self.time.append(step * self.dt)
            #sample_value.append(self.get_sample(v, 0.1))
            self.mean_value.append(self.get_mean(v))
            self.fluxes[0].append(self.get_flux(v, 0.1))
            self.fluxes[1].append(self.get_flux(v, 0.5))
            self.fluxes[2].append(self.get_flux(v, 0.9))

            step += 1

            # Check for steady state

            # Change in mean
            diff = np.abs(previous - v)
            m = np.max(diff)

            if m / self.dt < tolerance:
                break

        self.map = v
        print("Finished running")


    def plot_map(self):
        """
            Plot the data in the object.
        """
        fig = plt.figure() # figsize=(10,5)
        gs = gridspec.GridSpec(2,2)

        ax = fig.add_subplot(gs[1,:])
        im = ax.imshow(self.map.transpose(), cmap=None, interpolation='nearest', vmin = 0, vmax=10)
        fig.colorbar(im, ax=ax)
        ax.set_title("Diffusion")

        ax_sample = fig.add_subplot(gs[0,0])
        ax_sample.plot(self.time, self.mean_value)

        ax_flux = fig.add_subplot(gs[0,1])
        ax_flux.plot(self.time, self.fluxes[0], label = "flux 1")
        ax_flux.plot(self.time, self.fluxes[1], label = "flux 2")
        ax_flux.plot(self.time, self.fluxes[2], label = "flux 3")
        ax_flux.legend()

        
        plt.tight_layout()
        plt.show()

    def save_map(self, filename):
        """
            Saves the object.
        """
        metadata = np.array([self.dx, self.D, self.dt, self.h, self.w, self.r, self.b, self.x_padding, self.y_padding, self.K, self.V_0])

        # Add sample_values later
        np.savez(filename + '.npz', metadata = metadata, boundary=self.boundary_type, data = self.map, 
                 time = np.array(self.time), mean_value = np.array(self.mean_value), fluxes = np.array(self.fluxes))

    def load_map(self, filename):
        """
            Loads an object from file.
        """
        loaded = np.load(filename)

        metadata = loaded['metadata']
        self.dx = metadata[0]
        self.D =  metadata[1]
        self.dt = metadata[2]
        self.h = metadata[3]
        self.w = metadata[4]
        self.r = metadata[5]
        self.b = metadata[6]
        self.x_padding = metadata[7]
        self.y_padding = metadata[8]
        self.K = metadata[9]
        self.V_0 = metadata[10]

        self.boundary_type = loaded["boundary"]
        self.map = loaded['data']

        self.time = loaded['time']
        self.mean_value = loaded["mean_value"]
        self.fluxes = loaded["fluxes"]



def compare_discretization():

    gridsizes = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(gridsizes)
    runtimes = []

    for gridsize in gridsizes:
        d = Diffusion(gridsize=gridsize, D=10, canal_height=10, canal_width=10, sample_map=True)
        start_time = time.time()
        d.run(5, timemode=True)
        end_time = time.time()
        runtimes.append(end_time-start_time)
        d.save_map("Sample_Map_Gridsize_varied_"+str(gridsize))
        plt.plot(d.time, d.mean_value, label="Gridsize "+str(gridsize))

    print(runtimes)
    plt.legend()
    plt.show()


def main():

    #compare_discretization()
    d = Diffusion(gridsize=1, D=10, canal_height=10, canal_width=10, sample_map=True)
    d.load_map("Sample_Map_Gridsize_varied_0.5.npz")
    d.plot_map()

    #d = Diffusion(0.5,10, y_padding=30,x_padding=15)
    #d.load_map("Gridsize_varied_0.2.npz")
    #d.plot_map()

    #d.load_map("test.npz")

    #d.run_till_steady_state(0.1)
    #d.plot_map()

    #d.animate()
    #d.run(40, timemode=True)
    #d.save_map("test")
    #d.plot_map()
    #d.generate_geometry(show=True)



if __name__ == "__main__":
    main()