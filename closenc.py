#
#
#

import os
import sys
import shutil
import math
import numpy as np

import rebound

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

import time
from datetime import datetime, timedelta
from astropy import units as u

# from scipy.optimize import curve_fit

import requests
import json
from bs4 import BeautifulSoup
import urllib2



# SOME CONSTANTS

# au in lunar distance
lunar = 389.177939646 

# Earth sidereal revolution period
year_to_day = 365.256363004
# One year in units where G=1 
# (untuk konversi satuan yang digunakan di dalam Rebound)
# (4*pi**2 / (1 + M_earth))

year = 6.2831758714599477 

au = 149597870.7 # km

GOLDEN_RATIO = 0.5*(1. + np.sqrt(5))    


# PLOTTING
matplotlib.rcParams.update({'text.usetex': True})
matplotlib.rcParams.update({'font.size': 20})



# -------------------------


def get_daeiw(sa):
    # calc d, a, e, i, w as function of t
    times = np.zeros(len(sa))
    daeiw = np.zeros((8, len(sa)))
    for i, sim in enumerate(sa):
        times[i] = sim.t
        d = sim.particles[10] - sim.particles[3]
        dist = np.sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
        azimuth = math.atan2(d.y, d.x)
        
        # For rotating frame
        ds = sim.particles[10] - sim.particles[0] # heliocentric distance 
        de = sim.particles[3] - sim.particles[0]
        azs = math.atan2(ds.y, ds.x)
        aze = math.atan2(de.y, de.x)
        azr = azs - aze
        drot = np.sqrt(ds.x*ds.x + ds.y*ds.y + ds.z*ds.z)/np.sqrt(de.x*de.x + de.y*de.y + de.z*de.z) # not in AU, but relative to the Earth-Sun distance

        
        daeiw[:,i] = dist, sim.particles[10].a, sim.particles[10].e, sim.particles[10].inc, sim.particles[10].omega, azimuth, azr, drot
        
    return times, daeiw


def plot_daeiw(t, daeiw, list_enc, resultdir):     
    # plotting
    nRow = 5
    nCol = 1
    xSize = 12
    ySize = xSize/GOLDEN_RATIO #float(nRow)*xSize/GOLDEN_RATIO
    color = ['m', 'r', 'g', 'b', 'k']

    ymin = np.array([daeiw[0,:].min(), daeiw[1,:].min(), daeiw[2,:].min(), daeiw[3,:].min(), daeiw[4,:].min()])
    ymax = np.array([daeiw[0,:].max(), daeiw[1,:].max(), daeiw[2,:].max(), daeiw[3,:].max(), daeiw[4,:].max()])
    ymin[0] = 0.0 # distance to the Earth

    #startY = [0.0, 0.0, 0.0, 0.0]
    delY = (ymax - ymin)/3.
    #delMY = np.array(delY)/2.
    
    xmin = t.min()
    xmax = t.max()
    startX = xmin
    delX = (xmax - xmin)/10.
    delMX = (xmax - xmin)/50.
    ylabel = [r'$d$ [au]', r'$a$ [au]', r'$e$', r'$i$ [rad]', r'$\omega$ [rad]']

    fig = plt.figure(figsize=(xSize,ySize))
    gs = gridspec.GridSpec(nRow, nCol, wspace=0, hspace=0, height_ratios=[2, 1, 1, 1, 1])
    xticks = np.arange(startX, xmax+1.e-3*delX, delX)
    for i in range(nRow):
        ax = plt.subplot(gs[i])
        if i==0:
            ax.scatter(t, daeiw[i,:], c=t/255., s=2, lw=0)
        else:
            ax.plot(t, daeiw[i,:], color[i])
        
        ax.set_xticks(xticks)
        ax.set_xlim(xmin, xmax)
        minorLocator_x = MultipleLocator(delMX)
        ax.xaxis.set_minor_locator(minorLocator_x)
       
        if i == 0:
            yticks = np.arange(ymin[i], ymax[i]+0.05*delY[i], delY[i])
            ax.set_yticks(np.round(yticks, decimals=3))
        else:
            yticks = np.arange(ymin[i], ymax[i]-0.05*delY[i], delY[i])
            ax.set_yticks(np.round(yticks, decimals=3))
        #minorLocator_y = MultipleLocator(delMY[i])
        #ax.yaxis.set_minor_locator(minorLocator_y)
        ax.set_ylim(ymin[i]-0.01*delY[i], ymax[i])
        
        # x-label only the for the last plot
        if (i<(nRow-1)):
            ax.set_xticklabels('')
        if (i==(nRow-1)):
            ax.set_xlabel(r'$t$ [year]', fontsize=18)
        ax.set_ylabel(ylabel[i], fontsize=18)

    plt.tight_layout()
    plt.savefig(resultdir+"daeiw.pdf")


def plot_geocentric(t, daeiw, resultdir, lim=1200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([0.0], [0.0], marker='o', c='black', s=40, lw=0, alpha=0.8)
    ax.scatter(daeiw[0,:lim]*np.cos(daeiw[5,:lim]), daeiw[0,:lim]*np.sin(daeiw[5,:lim]), c=t[:lim]/255., s=5, lw=0, alpha=0.5)
    ax.text(-0.3, -0.3, 'Earth') 
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(resultdir+"geocentric.pdf")


def plot_rotatingframe(t, daeiw, resultdir, lim=1200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([0.0], [0.0], marker='*', c='black', s=40, lw=0)
    ax.scatter([1.0], [0.0], marker='o', c='black', s=40, lw=0)
    ax.scatter(daeiw[7,:lim]*np.cos(daeiw[6,:lim]), daeiw[7,:lim]*np.sin(daeiw[6,:lim]), c=t[:lim]/255., s=5, lw=0, alpha=0.5)
    ax.text(-0.2, -0.3, 'Sun')
    ax.text(0.7, -0.3, 'Earth')
    ax.set_xlabel(r"$x'$")
    ax.set_ylabel(r"$y'$")
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(resultdir+"rotframe.pdf")


def plot_orbit(sim, lim=3.0, outputname="orbit_ce.pdf", particles='', resultdir="result/astername"):
    plt.close()
    matplotlib.rcParams.update({'font.size': 20})
    fig = rebound.OrbitPlot(sim, slices=True, trails=True, color=True, unitlabel="[au]", lim=lim, limz=lim*0.6667, plotparticles=particles)
    plt.tight_layout()
    plt.savefig(os.path.join(resultdir, outputname))


def init_simulation(asteroidname="Florence", initdate="2017-08-17 00:00", tend=100.0, filename='', resultdir="result/asteroidname"):
    print "Initiate Simulation..."
    sim = rebound.Simulation()

    print "Download initial condition at "+initdate+" ...."
    # download initial condition from JPL NASA Horizons 
    sim.add("Sun", date=initdate) 
    sim.add("Mercury", date=initdate)
    sim.add("Venus", date=initdate)
    sim.add("399", date=initdate) # Earth
    sim.add("301", date=initdate) # Moon
    sim.add("Mars", date=initdate)
    sim.add("Jupiter", date=initdate)
    sim.add("Saturn", date=initdate)
    sim.add("Uranus", date=initdate)
    sim.add("Neptune", date=initdate)
    sim.add(asteroidname, date=initdate)
    
    # add radius for collision detection
    # Radius is not retrieved from JPL NASA using above script.
    ps = sim.particles 
    ps[0].r = 695700/au
    ps[1].r = 2440/au
    ps[2].r = 6052/au
    ps[3].r = 6378/au
    ps[4].r = 1737/au
    ps[5].r = 3390/au
    ps[6].r = 69911/au
    ps[7].r = 58232/au
    ps[8].r = 25362/au
    ps[9].r = 24622/au
    
    # with open(resultdir + "initpos_" + asteroidname + ".txt", 'w') as of:
    #     of.write(sim.particles_ascii()) # write to text file, to be used later.

    # with open("initpos_" + asteroidname + ".txt", 'r') as ifile:
    #     data = ifile.read()
    # sim.add_particles_ascii(data)

    print "Plot initial orbit..."
    plot_orbit(sim, outputname="initial_orbit.pdf", resultdir=resultdir)

    # Setting
    sim.integrator = "ias15" # IAS15 is the default integrator, so we actually don't need this line
    sim.dt = 0.001           # IAS15 use the adaptive timestep
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.move_to_com()
    tmax = tend*year # end of simulation (in year)
    Noutput = 10000


    if filename == '':
        asteroidname = asteroidname.replace(" ", "") # if the name given is 2017 MZ8 -> 2017MZ8
        filename = asteroidname+".bin"

    sim.initSimulationArchive(resultdir+filename, interval=tmax/Noutput) 
    # save in binary format (H. rain, D. Tamayo, 2017)
    
    print "Integrate"
    print "Number of particles before integration: ", len(ps)
    print "Integrating..."
    sim.integrate(tmax, exact_finish_time=0) # simulate
    print "Number of particles after integration: ", len(ps)

    print("Integration finish.")
    resolutiontime = (tend)*year_to_day*24./Noutput
    print("Resolution of the saved data = %f hour" % (resolutiontime))


# def quadratic(x, a, b, c):
#     return a*x*x + b*x + c

def find_closeencounter(sa, times, distance, initdate, lim_dist = 0.301, lim_time = 0.3):
    """
    find the close encounter
    d < 0.301
    
    try to approximate the exact time with interpolation
    
    the different of time & distance with JPL result become larger along integration 
    (1-12 minutes) (?) -> not caused by output resolution 
    maybe due to:
        - error in integration
        - Ceres and other massive asteroids are not included
    """
    # list of close encounter
    sortedidx = np.argsort(distance)
    lim_time = lim_time*sa[0].particles[10].P
    
    list_enc = [] # directly from output simulation
    list_encounter = [] # find detail
    
    for index, x in np.ndenumerate(sortedidx):
        if (distance[x] <= lim_dist):
            select = True
            for ind, y in enumerate(list_enc):
                if (np.abs(times[x] - y[1]) < lim_time) :
                    select = False
            
            if select:
                list_enc.append((x, times[x], distance[x]))
    
    if len(list_enc) > 0:
        # try with higher output resolution
        newres = 30.0 # seconds
        newres = newres/(3600.0*24*year_to_day) * year # in 'simulation time'
        for idx, td in enumerate(list_enc):
            id_encounter = td[0]
            # re-simulation +- 1 timestep of close encounter 
            # with higher output resolution.
            # the closest distance must be in that region
            newtimes = []
            newdist = []
            for i in np.arange(times[id_encounter-1], times[id_encounter+1], newres):
                sim = sa.getSimulation(i, mode="exact")
                d = sim.particles[10] - sim.particles[3]
                dist = np.sqrt(d.x*d.x + d.y*d.y + d.z*d.z)
                newtimes.append(i)
                newdist.append(dist)

            newtimes = np.array(newtimes)
            newdist = np.array(newdist)

            idxmin = np.argmin(newdist)
            tminimum = newtimes[idxmin]
            dminimum = newdist[idxmin]

            # try simple interpolation, quadratic equation
            # plusminus = 5 # bigger value will be worse
            # tslice = newtimes[idxmin-plusminus:idxmin+plusminus]
            # dslice = newdist[idxmin-plusminus:idxmin+plusminus]

            # #coef, pcov = curve_fit(quadratic, tslice, dslice, p0=[2.21258442, -1.18020052, 0.20461748])
            # coef = np.polyfit(tslice, dslice, 2)
            # print coef
            # tminimum = -coef[1]/(2.*coef[0]) # 
            # dminimum = coef[2] + coef[1]*tminimum + coef[0]*tminimum*tminimum
            # # print tminimum, dminimum

            # # for plotting
            # tx = np.linspace(tslice.min(), tslice.max(), 100)
            # y = coef[2] + coef[1]*tx + coef[0]*tx*tx     

            # plt.plot(tslice, dslice, 'b.')
            # plt.plot(tx, y, 'r,')
            # #plt.plot([tminimum], [dminimum], 'go')
            # plt.xlabel("t (rebound)")
            # plt.ylabel("distance")
            # plt.show()
            # plt.close()

            # find the close encounter time
            closeencountertime = tminimum/year
            #print("Minimum distance (%f AU or %f LD) occured at time: %f years." % (dminimum, dminimum*lunar, closeencountertime))
            encounterdate = datetime.strptime(initdate, '%Y-%m-%d %H:%M') + timedelta(days=year_to_day*closeencountertime)
            #print closeencountertime, " years (after initial)"
            strtime = encounterdate.strftime("%Y-%m-%d %H:%M")
            print strtime
            dminimum_au = dminimum*u.au
            dminumum_km = dminimum_au.to(u.km)
            dminimum_ld = dminimum_au.value * lunar
            print 'Closest distance = ', dminimum_au, ' = ', dminumum_km, ' = ', dminimum_ld, ' LD'

            list_encounter.append([tminimum, dminimum, strtime, dminimum_au.value, dminumum_km, dminimum_ld])

    return list_encounter   


def plot_simulation(archivename, initdate="2017-08-17 00:00", resultdir="result/asteroidname"):
    print "Open the Simulation Archive: "+archivename

    sa = rebound.SimulationArchive(resultdir+archivename)

    print len(sa), sa.tmin, sa.tmax
    print "Filesize: ", sa.filesize/1000000, " Mb" 

    print "Calculate the distance to the Earth..."
    times, daeiw = get_daeiw(sa)
    
    t = times/year
    
    print "Find close encounter time..."
    list_enc = find_closeencounter(sa, times, daeiw[0,:], initdate=initdate)

    print "Plotting..."
    plot_daeiw(t, daeiw, list_enc, resultdir)

    limitplot = 12.0 # years
    lim = int(limitplot/t[-1] * len(sa)) # limit plotting in number of data point
    print "Limit to first "+str(limitplot)+" years for geocentric and rotating frame plot..("+str(lim)+" steps)"
    plot_geocentric(t, daeiw, resultdir, lim)
    plot_rotatingframe(t, daeiw, resultdir, lim)
        
    return list_enc


def get_about(id_asteroid):
    """ Get information by scrapping the SBDB JPL NASA website"""
    # in this website we can use asteroid name, asteroid ID, and also SPK_ID
    url = 'http://ssd.jpl.nasa.gov/sbdb.cgi?sstr='+id_asteroid
    print "GET "+url
    r = requests.get(url)
    data = r.content

    firstidx = data.find("SPK-ID") + 54
    lastidx = data[firstidx:].find("<")
    SPK_ID = data[firstidx:firstidx+lastidx]

    firstidx = data.find("Classification") + 121
    lastidx = data[firstidx:].find("</a>")
    classy = data[firstidx:firstidx+lastidx]
    if data[firstidx+lastidx+12] == "[":
        first = data[firstidx+lastidx+12:].find(">")
        last = data[firstidx+lastidx+12+first:].find("<")
        classy += " (" + data[firstidx+lastidx+13+first:firstidx+lastidx+12+first+last] + ")"

    firstidx = data.find("Discovered")
    lastidx = data[firstidx:].find("</font>")
    discovered = ''
    if firstidx > 0:
        discovered = "\item " + data[firstidx:firstidx+lastidx]


    """ Get information about asteroid using NASA API https://api.nasa.gov"""
    url = "https://api.nasa.gov/neo/rest/v1/neo/"+SPK_ID+"?&api_key=A6YxGaVtqMjCQbduPKXlwkoXTqOiTFl9koahEGGK"
    print "GET "+url
    r = requests.get(url)
    data = json.loads(r.content)

    name = data['name']
    if (name[0] == "(") and (name[-1] == ")"):
        name = name[1:-1]

    H = data['absolute_magnitude_h']

    try: 
        D_min = data['estimated_diameter']['meters']['estimated_diameter_min']
        D_max = data['estimated_diameter']['meters']['estimated_diameter_max']
        if D_min > 1000:
            estimated_diameter = str(D_min/1000)[:4] + " \--- " + str(D_max/1000)[:4] + " km"
        else:
            estimated_diameter = str(D_min)[:3] + " \--- " + str(D_max)[:3] + " meters"
    except:
        estimated_diameter = 'unknown'

    # Orbital data
    o = data['orbital_data']
    P = o['orbital_period'][:7]

    epoch_osc = o['epoch_osculation']
    a = o['semi_major_axis'][:7]
    ecc = o['eccentricity'][:7]
    inc = o['inclination'][:7]
    Om = o['ascending_node_longitude'][:7]
    om = o['perihelion_argument'][:7]
    M = o['mean_anomaly'][:7]

    orbital = "Semi-major axis ($a$) & & "+a+" \\\\ \n"
    orbital+= "Eccentricity ($e$) & & "+ecc+" \\\\ \n"
    orbital+= "Inclination ($i$) & & "+inc+" \\\\ \n"
    orbital+= "Lon. of ascending node ($\Omega$) & & "+Om+" \\\\ \n"
    orbital+= "Argument of pericenter ($\omega$) & & "+om+" \\\\ \n"
    orbital+= "Mean Anomaly ($M$) & & "+M+" \\\\ \n"


    ca = data['close_approach_data']
    number_of_ca = len(ca)
    table_ca = ''
    if number_of_ca > 0:
        for each_ca in ca:
            if float(each_ca['close_approach_date'][:4]) >= time.localtime().tm_year: # only future close approach
                table_ca += each_ca['close_approach_date'] + ' & ' + each_ca['orbiting_body'] + ' & ' + each_ca['miss_distance']['astronomical'][:6] + ' & ' + each_ca['relative_velocity']['kilometers_per_second'][:5] + ' \\\\' + ' \n'


    return name, discovered, estimated_diameter, classy, H, epoch_osc, P, orbital, table_ca



def make_report(id_asteroid="2004 MN4", initdate="2017-08-17 00:00", tend=100.0, dirname="result"):
    """ 
    - id_asteroid can be asteroid-id-number, name, or SPK-ID
    - initdate is to the date to init the integration -> download initial condition
    - tend = length of integration (in years)
    """

    SPK_ID = id_asteroid
    asteroidname = id_asteroid
    pars = get_about(SPK_ID)
    print "Name: ", pars[0]

    asteroidname = asteroidname.replace(" ", "") # if the name given is 2017 MZ8 -> 2017MZ8
    resultdir = dirname+"/"+asteroidname+"/"

    if os.path.exists(resultdir):
        print "Remove previos directory"
        shutil.rmtree(resultdir)
        print "Make a new dir", resultdir
        os.makedirs(resultdir)
    else:
        print "Make a new dir", resultdir
        os.makedirs(resultdir)

    # Run simulation
    init_simulation(id_asteroid, initdate=initdate, tend=tend, resultdir=resultdir)

    # Make some plots
    list_ca = plot_simulation(asteroidname+".bin", initdate=initdate, resultdir=resultdir)
    
    # if CA is not found in JPL, use our result (accuracy is a bit worse, 1-12 minutes in CA time.. we only show the date.)
    # if pars[-1] == '':
    #     for ca in list_ca:
    # Only with the Earth (in case too many CA)


    timenow = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    print "Write a report..."
    
    report_tex = r"""\documentclass[final]{beamer}

\usepackage[scale=1.24]{beamerposter}
\usetheme{confposter}
\setbeamercolor{block title}{fg=ngreen,bg=white}
\setbeamercolor{block body}{fg=black,bg=white}
\setbeamercolor{block alerted title}{fg=white,bg=dblue!70}
\setbeamercolor{block alerted body}{fg=black,bg=dblue!10}

\newlength{\sepwid}
\newlength{\onecolwid}
\newlength{\twocolwid}
\newlength{\threecolwid}
\setlength{\paperwidth}{48in} % A0 width: 46.8in
\setlength{\paperheight}{34in} % A0 height: 33.1in
\setlength{\sepwid}{0.024\paperwidth} % Separation width (white space) between columns
\setlength{\onecolwid}{0.22\paperwidth} % Width of one column
\setlength{\twocolwid}{0.464\paperwidth} % Width of two columns
\setlength{\threecolwid}{0.708\paperwidth} % Width of three columns
\setlength{\topmargin}{-0.5in} % Reduce the top margin size
%-----------------------------------------------------------

\usepackage{graphicx}  % Required for including images

\usepackage{booktabs} % Top and bottom rules for tables

\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=blue,
}

\usepackage{amsmath}

\title{Close Encounter: """ + pars[0] + r""" } % Poster title

\author{Bosscha Observatory $\vert$ Astronomy Research Division, ITB}

\institute{This information is generated on """ + timenow + r""".} 

\begin{document}

\addtobeamertemplate{block end}{}{\vspace*{2ex}}
\addtobeamertemplate{block alerted end}{}{\vspace*{2ex}}

\setlength{\belowcaptionskip}{2ex}
\setlength\belowdisplayshortskip{2ex}

\begin{frame}[t]

\begin{columns}[t] 

% First Column

\begin{column}{\sepwid}\end{column}

\begin{column}{\onecolwid}

\begin{alertblock}{Basic Properties}
\begin{itemize}
\item Name: """ + pars[0] + pars[1] + r""" 
\item Estimated diameter: """ + pars[2] + r"""
\item Classification: """ + pars[3] + r"""
\item H: """ + str(pars[4]) + r"""
\item Period: """ + str(pars[6]) + r""" days
\end{itemize}
\begin{table}
\caption{Orbital elements at epoch """+ pars[5] + r""" JD }
\begin{tabular}{l c r}
\toprule
\textbf{Parameter} & & \textbf{    Value    } \\
\midrule """ + pars[7] + r"""\bottomrule
\end{tabular}
\end{table}

\end{alertblock}


\begin{block}{Orbit}
\begin{figure}
\includegraphics[width=0.99\textwidth]{initial_orbit.pdf}
\caption{Orbit of the asteroid and the planets. This plot only shows the inner region of the Solar System.}
\end{figure}
\end{block}

\begin{alertblock}{Acknowledgements}
Our script retrieve basic informations and initial condition from \href{https://ssd.jpl.nasa.gov/}{JPL NASA}, re-integrate the Solar System using \href{https://github.com/hannorein/rebound}{\texttt{Rebound}} package, and write this report using \LaTeX.
\end{alertblock}

\end{column}

% Second column

\begin{column}{\sepwid}\end{column}

\begin{column}{\twocolwid}

\begin{block}{Distance to the Earth and Orbital Elements}
\begin{figure}
\includegraphics[width=0.99\linewidth]{daeiw.pdf}
\caption{Distance to the Earth and orbital elements of the asteroid for the next 100 years. Starting time of the integration is """+ initdate + r""" UTC.}
\end{figure}
\end{block}

\begin{columns}[t, totalwidth=\twocolwid] % Split up the two columns wide column

\begin{column}{\onecolwid}\vspace{-.6in}
\begin{block}{Geocentric}
\begin{figure}
\includegraphics[width=0.99\textwidth]{geocentric.pdf}
\caption{Movement of the asteroid relative to the Earth ($xy$-plane; only the first 12 years of integration).}
\end{figure}
\end{block}
\end{column}

\begin{column}{\onecolwid}\vspace{-.6in} 
\begin{block}{Rotating Frame}
\begin{figure}
\includegraphics[width=0.99\textwidth]{rotframe.pdf}
\caption{Movement of the asteroid relative to the Sun-Earth system ($xy$-plane; only the first 12 years of integration).}
\end{figure}
\end{block}
\end{column}

\end{columns} 

\end{column} 


%%%% 
\begin{column}{\sepwid}\end{column} 

\begin{column}{\onecolwid} 

\setbeamercolor{block alerted title}{fg=black,bg=norange}
\setbeamercolor{block alerted body}{fg=black,bg=norange!10}

\begin{alertblock}{Nearest close encounter}
\begin{itemize}
\item Time: 1 September 2017, 12.05 UT
\item Distance:  $ 7.066 \times 10^6$ km ($18.38$ Lunar Distance)
\item Relative velocity: 
\end{itemize}
\end{alertblock}


\begin{alertblock}{List of close encounter}
\begin{table}
\vspace{2ex}
\begin{tabular}{l c c c}
\toprule
\textbf{Time} & \textbf{   Body   } & \textbf{  $\boldsymbol{d}$ (au)  } & \textbf{$\boldsymbol{v_{rel}}$ (km/s)} \\
\midrule""" + pars[-1] + r"""\bottomrule
\end{tabular}
%\caption{Word Formation}
\end{table}
\end{alertblock}

\end{column} 


\end{columns} 
\end{frame}
\end{document}"""
    
    with open(resultdir+"information.tex", 'w') as ofile:
        ofile.write(report_tex)
    
    try:
        print "Copy beamerthemeconfposter.sty file to ", resultdir
        os.system("cp beamerthemeconfposter.sty "+resultdir)
        os.chdir(resultdir)
        print "Running pdflatex..."
        os.system("pdflatex information.tex")

        print "Removing some temporary files..."
        if os.path.exists("information.aux"): os.remove("information.aux")
        if os.path.exists("information.log"): os.remove("information.log")
        if os.path.exists("information.nav"): os.remove("information.nav")
        if os.path.exists("information.out"): os.remove("information.out")
        if os.path.exists("information.snm"): os.remove("information.snm")
        if os.path.exists("information.toc"): os.remove("information.toc")
        if os.path.exists("information.synctex.gz"): os.remove("information.synctex.gz")

        os.chdir("../")
    except:
        print "Error running pdflatex"
    



if __name__ == '__main__':
    initdate = "2017-08-17 00:00"
    
    asteroidname = raw_input("Name of the asteroid (e.g. apophis, 2014 MN4, 99942, or 2099942): ")
    
    idate = raw_input("Initial time for integration?[2017-08-17 00:00] : ")

    if idate != '':
        initdate = idate


    make_report(asteroidname, initdate=initdate)


    # asteroidname = "Florence"
    # if len(sys.argv) > 1:
    #     while len(sys.argv) > 1:
    #         option = sys.argv[1]; del sys.argv[1]
    #         print option
    #         if option == '-name':
    #             try:
    #                 if (len(sys.argv[2]) == 3):
    #                     asteroidname = sys.argv[1]+" "+sys.argv[2]; del sys.argv[1]; del sys.argv[2]
    #             except:
    #                 asteroidname = sys.argv[1]; del sys.argv[1]
    #         elif option == '-initdate':
    #             initdate = sys.argv[1]+" "+sys.argv[2]; del sys.argv[1]; del sys.argv[2]
    #         else:
    #             print sys.argv[0], ': invalid option', option
    #             print "Usage: python closenc.py -name apophis -initdate 2017-08-17 00:00"
    #             sys.exit(1)
    # else: