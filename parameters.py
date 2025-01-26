import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from copy import deepcopy
#import scipy
import sys
import time

import relaxation


# Study parameters

def height_dependence():

    heights = [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60]
    flux = []
    
    for height in heights:
        d = relaxation.Diffusion(1,10, canal_height=height,y_padding=30,x_padding=15)
        d.run_till_steady_state(0.01)
        d.save_map("height_"+str(height)+"_dx_"+str(1)+"_tol_"+str(0.01))
        flux.append(d.fluxes[2][-1])
        #d.plot_map()

    np.save("flux_1_0.01", np.array(flux))

    plt.plot(heights, flux)
    plt.title("Flux out of canal for different heights")
    plt.xlabel("Height [mm]")
    plt.ylabel("Flux [g/s]")
    plt.show()

def height_dependence_plot():

    fst = 13
    fs = 12

    heights = np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60])
    flux = np.load("flux_1_0.01.npy")

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    
    #plt.semilogy(heights, flux1)
    #plt.loglog(heights, 800/heights)
    ax1.plot(heights, flux)
    ax1.set_title("Flux out of canal for different heights", fontsize = fst)
    ax1.set_xlabel("Canal height [mm]", fontsize = fs)
    ax1.set_ylabel("Flux [g/s]", fontsize = fs)

    ax2.plot(heights, 1/flux)
    ax2.set_title("Reciprocal flux out of canal for different heights", fontsize = fst)
    ax2.set_xlabel("Canal height [mm]", fontsize = fs)
    ax2.set_ylabel("1/Flux [s/g]", fontsize = fs)

    plt.tight_layout()
    plt.savefig("height_variation.png")
    plt.show()

def width_dependence():

    widths = [4,8,12,16,20,24,28,32]
    flux = []
    
    for width in widths:
        d = relaxation.Diffusion(1,10, canal_width=width,y_padding=30,x_padding=15)
        d.run_till_steady_state(0.01)
        d.save_map("width_"+str(width)+"_dx_"+str(1)+"_tol_"+str(0.01))
        flux.append(d.fluxes[2][-1])
        #d.plot_map()

    np.save("flux_w_1_0.01", np.array(flux))

    plt.plot(widths, flux)
    plt.title("Flux out of canal for different widths")
    plt.xlabel("width [mm]")
    plt.ylabel("Flux [g/s]")
    plt.show()

def width_dependence_plot():

    fst = 13
    fs = 12

    widths = np.array([4,8,12,16,20,24,28,32])
    flux = np.load("flux_w_1_0.01.npy")

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    
    ax1.plot(widths, flux / widths)
    ax1.set_title("Flux out of canal for different widths", fontsize = fst)
    ax1.set_xlabel("Canal width [mm]", fontsize = fs)
    ax1.set_ylabel("Flux [g/s$\cdot$mm]", fontsize = fs)

    log_f = np.log(flux / widths)
    log_w = np.log(widths)

    slope = 0
    c = 0
    for index in range(0,len(log_f)-1):
        c+=1
        slope += (log_f[index+1]-log_f[index]) / (log_w[index+1]-log_w[index])
    slope = slope / c

    print(slope)

    ax2.loglog(widths, flux / widths)
    ax2.set_title("Flux out of canal for different widths (loglog)", fontsize = fst)
    ax2.set_xlabel("Canal width [mm]", fontsize = fs)
    ax2.set_ylabel("Flux [g/s$\cdot$mm]", fontsize = fs)

    plt.tight_layout()
    plt.savefig("width_variation.png")
    plt.show()

### To study dx-dependence

def compare_discretization():

    #pw = int(self.h / self.dx)
    #ph = int(self.w / self.dx)

    gridsizes = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    print(gridsizes)
    runtimes = []

    for gridsize in gridsizes:
        d = relaxation.Diffusion(gridsize=gridsize, D=10, canal_height=10, canal_width=10, sample_map=2)
        start_time = time.time()
        d.run(5, timemode=True)
        end_time = time.time()
        runtimes.append(end_time-start_time)
        d.save_map("Sample_Map_2_Gridsize_varied_"+str(gridsize))
        plt.plot(d.time, d.mean_value, label="Gridsize "+str(gridsize))

    print(runtimes)
    #np.save("runtimes", np.array(runtimes))
    plt.legend()
    plt.show()

def compare_profiles():

    fs = 14
    fst = 16

    gridsizes = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    print(gridsizes)

    #Reference
    d = relaxation.Diffusion(gridsize=1, D=10, canal_height=10, canal_width=10, sample_map=True)
    d.load_map("Sample_Map_2_Gridsize_varied_0.2.npz")
    if d.map.shape[0] % 2 == 1:
        value = d.map[d.map.shape[0]//2]
    else:
        value = (d.map[d.map.shape[0]//2] + d.map[d.map.shape[0]//2-1])/2
    position = np.linspace(0,1,d.map.shape[1])


    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    errors = []

    #Multiple
    for gridsize in gridsizes:
        d = relaxation.Diffusion(gridsize=gridsize, D=10, canal_height=10, canal_width=10, sample_map=True)
        d.load_map("Sample_Map_2_Gridsize_varied_"+str(gridsize)+".npz")
        #d.plot_map()
        #print(d.map.shape[0])
        if d.map.shape[0] % 2 == 1:
            values = d.map[d.map.shape[0]//2]
        else:
            values = (d.map[d.map.shape[0]//2] + d.map[d.map.shape[0]//2-1])/2

        positions = np.linspace(0,1,d.map.shape[1])

        d = np.interp(positions, position, value)

        plotted = np.abs(values-d)

        errors.append(np.interp(0.2, positions, plotted))

        ax1.plot(positions*10, plotted, label="Gridsize "+str(gridsize))

    ax1.legend()
    ax1.set_title("Error vs. position", fontsize=fst)
    ax1.set_ylabel("Error",fontsize=fs)
    ax1.set_xlabel("Position",fontsize=fs)
    
    l_g = -np.log(np.array(gridsizes))
    l_e = np.log(errors)
    slope = 0
    c = 0
    for index in range(0,len(l_g)-1):
        c += 1
        slope += (l_e[index+1]-l_e[index]) / (l_g[index+1]-l_g[index])
    slope = slope / c

    print(slope)

    ax2.loglog(10/np.array(gridsizes), errors)
    ax2.set_title("Loglog of error vs. gridpoints", fontsize=fst)
    ax2.set_ylabel("Error",fontsize=fs)
    ax2.set_xlabel("Gridpoints",fontsize=fs)


    fig.tight_layout()
    fig.savefig("Error.png")
    fig.show()

def plot_runtime():

    fs = 14
    fst = 16

    runtimes = np.load("runtimes.npy")
    gridsizes = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    log_r = np.log(runtimes)
    log_N = -np.log(gridsizes)

    slope = 0
    for index in range(0,len(runtimes)-1):
        slope += (log_r[index+1]-log_r[index]) / (log_N[index+1]-log_N[index])
    slope = slope / (len(runtimes)-2)

    print(slope)

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    ax1.plot(1/gridsizes, runtimes)
    ax1.set_title("Plot of runtime vs. N",fontsize=fst)
    ax1.set_ylabel("Runtime [s]",fontsize=fs)
    ax1.set_xlabel("N",fontsize=fs)

    ax2.loglog(1/gridsizes, runtimes)
    ax2.set_title("Loglog plot of runtime vs. N",fontsize=fst)
    ax2.set_ylabel("Runtime [s]",fontsize=fs)
    ax2.set_xlabel("N",fontsize=fs)

    #ax1.tick_params(axis='x', labelsize=fs) 
    #ax1.tick_params(axis='y', labelsize=fs)
    #ax2.tick_params(axis='x', labelsize=fs) 
    #ax2.tick_params(axis='y', labelsize=fs)

    plt.tight_layout()
    plt.savefig("Runtime.png")
    plt.show()




def main():

    width_dependence_plot()




if __name__ == "__main__":
    main()