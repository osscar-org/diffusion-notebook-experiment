import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from scipy.stats import linregress
import traitlets
import datetime

# This replaces the %matplotlib widget in the notebook
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'widget')

class NotoLogger:
    def __init__(self, event = None):
        self._get_logger()
        self._get_kernel_id()
        if event is not None:
            self.logEvent(event)

    def _get_logger(self):
        try:
            from cedelogger import cedeLogger
            self.my_logger = cedeLogger()
        except ImportError:
            self.my_logger = None

    def _get_kernel_id(self):
        import os
        from ipykernel import zmqshell
        try:
            connection_file = os.path.basename(zmqshell.get_connection_file())
            self.kernel_id = connection_file.split('-', 1)[1].split('.')[0]
        except Exception:
            self.kernel_id = 'n/a'

    def logEvent(self, event):
        data_to_log = {}
        if event['data']['type'] != 'click':
            data_to_log = {
                'from_value': event['data']['old'],
                'to_value': event['data']['new'],
            }     
        l = {
                'raw_event': str(event),
                'kid': self.kernel_id,
                'where': event['where'],
                'what': event['data']['owner'].__class__.__name__,
                'which': event['data']['owner'].description,             
                **data_to_log
            }
        #with open('test.log', 'a') as fhandle:
        #    fhandle.write(f"[{datetime.datetime.now()}] {l}\n")
        if self.my_logger:
            self.my_logger.log(l)
        else:
            print(f"Warning, cedeLogger not available, just printing: {l}")


class LoggingPlay(ipw.Play):
    @traitlets.observe("playing")
    def _log_playing(self, change):
         NotoLogger({'where': 'logging_play', 'data': change})

def show_diffusion():
    # Global variables in sub-function need to be declared global also here
    global play, trajectory, px_slider, r_std_sq, slope, intercept, dots_art, traj_art, circle, ax1, ax2, ax3, nsteps_slider, frame_slider
    
    eventLogger = NotoLogger()

    box_xrange = (-10, 10)
    box_yrange = (-10, 10)
    starting_radius = 0.1
    r = np.linspace(0,10,100)

    layout = ipw.Layout(width='auto', height='30px')
    ndots_slider = ipw.IntSlider(value=1000, min=1, max=1000, step=10, description='Number of points $N$', style= {'description_width': 'initial'}, layout=layout, continuous_update=False) # number of points
    stepsize_slider = ipw.FloatSlider(value=0.05, min=0.01, max=0.1, step=0.01, description='Step size $l$', continuous_update=False, readout=True, readout_format='.2f', style= {'description_width': 'initial'}, layout=layout) # max step size
    nsteps_slider = ipw.IntSlider(value=5000, min=100, max=10000, step=100, description='Number of time steps $t$', continuous_update=False, disabled=False, style= {'description_width': 'initial'}, layout=layout)
    px_slider = ipw.FloatSlider(value=0.5, min=0.45, max=0.55, step=0.01, description='$p_x$', continuous_update=False, readout=True, readout_format='.2f', style= {'description_width': 'initial'}, layout=layout) # max step size
    frame_slider = ipw.IntSlider(value=0, min=0, max=nsteps_slider.value, step=100, description='Time step # $(t)$', continuous_update=False, readout=True, disabled=True, style= {'description_width': 'initial'}, layout=layout) # step index indicator and slider
    

    traj_chkbox = ipw.Checkbox(value=False,description='Show trajectory of one particle', disabled=False, indent=False)
    map_chkbox = ipw.Checkbox(value=False,description='Show density map', disabled=False, indent=False)

    run_btn = ipw.Button(description='Simulate')
    run_btn.style.button_color = 'green'
    play = LoggingPlay(value=0, min=0,
        max=nsteps_slider.value, step=100, disabled=True,
        interval=500, show_repeat=False) # iterate frame with 500ms interval

    trajectory = []         # trajectory of all dots
    r_std_sq = np.array([]) # square standard radius
    slope = 0.              # slope of linear fit in plot 3
    intercept = 0.          # intercept of the fit

    def plot_dots_circle(ax):
        show_traj = traj_chkbox.value
        show_map = map_chkbox.value
        frame_idx = frame_slider.value

        r_l = np.sqrt(frame_idx) * stepsize_slider.value * np.sqrt(2) # analytical radius = sqrt(N) * stepsize * sqrt(2), a factor of sqrt(2) since we are in 2D
        r_std = np.sqrt(r_std_sq[frame_idx, 1])           # standard radius from simulation
        frame_coords = trajectory[frame_idx]

        ax.clear()
        ax.set_xlim(box_xrange)
        ax.set_ylim(box_yrange)
        ticks_ax1 = [-10., -5., 0., 5., 10]
        ax.xaxis.set_ticks(ticks_ax1)
        ax.yaxis.set_ticks(ticks_ax1)
        ax.set_aspect(1.)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        ax.set_title('Position of the points (2D)\n\n', fontsize=9, loc='center', wrap=True)

        # draw dots  
        ax.plot(frame_coords[:,0], frame_coords[:,1], '.', alpha=0.1, zorder=11)
        # draw circles if p==0.5
        if px_slider.value == 0.5:
            circle_std = plt.Circle((0, 0), r_std, color='green', linewidth=2, fill=False,zorder=12, label='$r_{std}$')
            ax.add_patch(circle_std)
        # draw trajectory of first dots
        if show_traj:
            full_traj = trajectory[:frame_idx:20,0,:]
            
            step_displacement = np.sqrt(((full_traj[1:, :] - full_traj[:-1, :])**2).sum(axis=1))

            # Get the size (both x and y) of the box
            box_size = min(
                abs(box_xrange[1] - box_xrange[0]),
                abs(box_yrange[1] - box_yrange[0]))
            # Find points where the displacement is larger than 1/6 of the box
            # This is not perfect, e.g. it depends on the step size, and might not
            # work near a corner of the box. It would be better to get directly
            # the information when generating the data, before applying PBC.
            # Nevertheless, this is good enough for now (it's just for visualization
            # purposes).
            # If any of the two x,y coordinates jumps, I want to mark it as a
            # point to break the trajectory
            # I get the index of the points where the jump is too large either
            # on x or on y. Note that the step displacement has length reduced 
            # by 1 w.r.t. to the full_trajectory.
            # We therefore add 1 (we want to break when the jump happens,
            # not at the step before)
            breaking_points = np.arange(len(step_displacement))[step_displacement > box_size / 6.] + 1
            # I want now to define segments. I always want to have 0 as the first
            # point and len(full_traj) as the last point
            breaking_points = np.append(np.insert(breaking_points, 0, 0), len(full_traj))

            for segment_start, segment_end in zip(breaking_points[:-1], breaking_points[1:]): 
                ax.plot(full_traj[segment_start:segment_end,0], full_traj[segment_start:segment_end,1], linewidth=2, color='purple', zorder=13)
         # analytical density map for the diffusion plot as a comparison for the actual simulation pattern
        if show_map:
            x = np.linspace(-10, 10, 30)
            y = np.linspace(-10, 10, 30)
            N = frame_idx
            l = stepsize_slider.value
            gx = expected_1d(x, N, l)
            gy = expected_1d(y, N, l)
            H = np.ma.outerproduct(gx, gy).data
            ax.imshow(H, origin='lower', interpolation='none', extent=[box_xrange[0], box_xrange[1], box_yrange[0], box_yrange[1]],aspect='equal', alpha=1, cmap='Reds')
        if px_slider.value == 0.5:            
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), prop={'size': 9})

    def expected_1d(x, N, l):
        """A helper function for plot 2.
        x: range
        N: number of steps
        l: stepsize
        Return expected distribution on 1D
        """
        if N == 0:
            return np.zeros(len(x)) # for simplicity of visualization, zeros is returned instead of a Dirac distribution
        var = N * l**2
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-x**2/ (2 * var)) 

    def plot_1d_hist(ax):
        """ draw plot 2
        Histogram is obtained consider only x direction, which should fits under
        1D expected distribution. Note that histogram may deviates from the expected one
        after prolonged time due to PBC.
        """
        frame_idx = frame_slider.value
        N = ndots_slider.value
        stepsize = stepsize_slider.value
        x_coords = trajectory[frame_idx][:,0]
        nbins = 30
        bin_width = (box_xrange[1] - box_xrange[0]) / nbins
        hist, bins= np.histogram(x_coords, bins=30, range=box_xrange, density=True)
    #     hist = hist / (bin_width * N) # normalized count by count/ (N * width) to get f(r)
        h_offset =  0.5 * bin_width # horizontal offset for histogram plot so the first column starts at 0
        r = np.linspace(box_xrange[0], box_xrange[1], 100)
        gr = expected_1d(r, frame_idx, stepsize)

        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("frequency", fontsize=9)
        ax.set_title('1D histogram of the position of the points\nalong the x axis\n\n', fontsize=9, loc='center', wrap=True)
        ax.bar(bins[:-1]+h_offset, hist, ec='k', width=bin_width)
        if px_slider.value == 0.5:
            ax.plot(r, gr, 'r--',label='Expected distribution')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), prop={'size': 9})

    def plot_radii(ax):
        """draw Plot 3

        """
        frame_idx = frame_slider.value
        nsteps = nsteps_slider.value
        ax.clear()

        if px_slider.value == 0.5:
            # plot r_std^2 (MSD) vs t
            interval = 500
            ax.plot(r_std_sq[::interval,0], r_std_sq[::interval,1], '.') # plot every few steps
            ax.plot(frame_idx, r_std_sq[frame_idx, 1], 'o', color='green', label='current step')

            # plot linear fitting line
            lx = np.linspace(0,nsteps,10)
            ly = lx * slope + intercept
            ax.plot(lx, ly, 'r--', lw=1, label='fit')
            #, label='fit: {:.2e} t + {:.2f}'.format(slope, intercept))

            ax.set_xlabel('time step # $(t)$', fontsize=9)
            ax.set_ylabel('$r_{std}^2$', fontsize=9)
            ax.set_title('$r_{std}^2$ as a function of the time step $t$\n\n', fontsize=9, loc='center')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, prop={'size': 9})

    def plot_frame(change):
        ''' plot current frame for all axis'''
        # check if trajectory is already stored
        if len(trajectory) == 0:
            return
        # plot 1
        plot_dots_circle(ax1)
        # plot 2
        plot_1d_hist(ax2)       # in x direction
    #     plot_circle_hist(ax2) # in spherical coords, along radius
        # plot 3
        plot_radii(ax3)

    def run(button):
        '''Main function for simulation
        - generate initial particle coords
        - run diffusion simulation and store trajectory of all dots in trajectory
        - do linear fitting on r_std and t for plot 3
        '''
        global trajectory, r_std_sq, slope, intercept, px_slider

        # I mage the 'data' closer to a callaback observed for other
        # widgets (buttons instead return only the button
        NotoLogger({'where': 'run', 'data': {
            'owner': button,
            'old': None,
            'new': None,
            'type': 'click'
            }
        })
        run_btn.style.button_color = 'red'
        N = ndots_slider.value
        # Initial coords with a random radial distribution generated by creating normal
        # random coords and take first N points in the initial circle. Arguably, we can
        # start with all particles at origin but that is less realistic. A demo
        # is attached as commented out code at the end of the notebook.
        stepsize = stepsize_slider.value # mean stepsize
        coords = (np.random.random((10*N, 2)) - 0.5)*2 * stepsize
        coords = coords[(coords**2).sum(axis=1) < starting_radius**2][:N] 

        assert len(coords) == N # check if enough points are in the circle

        # run simulation and store trajectory 
        trajectory = [coords]
        num_steps = nsteps_slider.value
        for i in range(num_steps):
            # two different ways of displacement with same distribution
            # random_displacement = (np.random.random((N, 2)) - 0.5) * 2 * stepsize # continuous
    
            # Probability to move to the right or to go up
            p_right = px_slider.value
            p_up = 0.5
            
            random_displacement_x = (np.random.choice([-1,1],N, p=[1-p_right, p_right]))  * stepsize # discrete
            random_displacement_y = (np.random.choice([-1,1],N, p=[1-p_up, p_up]))  * stepsize # discrete
            random_displacement = np.array([random_displacement_x, random_displacement_y]).T
            
            new_positions = trajectory[-1] + random_displacement
            # Some points might have gone beyond the box.
            # I could either reflect them back as a hard wall, or just use PBC. For simplicity, I use PBC
            new_positions[:,0] = (new_positions[:,0] - box_xrange[0]) % (box_xrange[1] - box_xrange[0]) + box_xrange[0]
            new_positions[:,1] = (new_positions[:,1] - box_yrange[0]) % (box_yrange[1] - box_yrange[0]) + box_yrange[0]    
            trajectory.append(new_positions)
        trajectory = np.array(trajectory)

        # calculate r_std by sqrt(mean**2 + std**2) and do the fitting
        radii = np.sqrt((trajectory**2).sum(axis=2))
        r_std_sq = (radii**2).sum(axis=1) / radii.shape[1] # radii.mean(axis=1)**2 + radii.std(axis=1)**2
        r_std_sq = np.c_[np.arange(len(r_std_sq)), r_std_sq]
        res = linregress(r_std_sq)
        slope = res.slope
        intercept = res.intercept

        # enable play and frame slider after the simulation run
        play.disabled = False
        frame_slider.disabled = False
        plot_frame('init')
        run_btn.style.button_color = 'green'

    def stop(change):
        ''' disable play widget and reset frame slider'''
        global dots_art, traj_art, circle

        NotoLogger({'where': 'plot_frame', 'data': change})
        play.disabled = True
        play.playing = False
        frame_slider.value = 0
        # reset all the axes

        for ax in [ax1, ax2, ax3]:
            ax.clear()
        initialize_plot()

    def initialize_plot():
        """Initialized plot to specify ranges, ticks or labels on x, y axis
        Called when first run the notebook or the simulation parameters change."""
        global ax1, ax2, ax3
        ax = ax1
        ax.set_xlim(box_xrange)
        ax.set_ylim(box_yrange)
        ticks_ax1 = [-10., -5., 0., 5., 10]
        ax.xaxis.set_ticks(ticks_ax1)
        ax.yaxis.set_ticks(ticks_ax1)
        ax.set_aspect(1.)
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        ax.set_title('Position of the points (2D)\n\n', fontsize=9, loc='center', wrap=True)

        ax = ax2
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("frequency", fontsize=9)
        ax.set_title('1D histogram of the position of the points\nalong the x axis\n\n', fontsize=9, loc='center', wrap=True)

        ax = ax3
        ax.set_xlabel('time step # $(t)$', fontsize=9)
        ax.set_ylabel('$r_{std}^2$', fontsize=9)
        ax.set_title('$r_{std}^2$ as a function of the time step $t$\n\n', fontsize=9, loc='center')

    def traj_callback(change):
        NotoLogger({'where': 'traj_callback', 'data': change})
        plot_frame(change)
        
    def frame_slider_callback(change):
        global play

        # Log only if not playing.
        # Note: we lose messages if the user clicks to change frame while playing,
        # but this is still better than logging a line at every played frame
        if not play.trait_values()['playing']:
            NotoLogger({'where': 'frame_slider_callback', 'data': change})
        plot_frame(change)

    # link widgets
    ipw.jslink((play, 'value'), (frame_slider, 'value'))
    ipw.jslink((nsteps_slider, 'value'), (frame_slider,'max'))
    # The value of both needs to be set, otherwise the frame_slider will set the value
    # of the play widget that will bring it back to its old 'max' value
    ipw.jslink((nsteps_slider, 'value'), (play,'max'))
    frame_slider.observe(frame_slider_callback, names='value', type='change')
    traj_chkbox.observe(traj_callback, names='value', type='change')

    # click run for simulation and collect trajectory
    run_btn.on_click(run)

    # change simulation parameters will disable play and frame slider until finish run
    ndots_slider.observe(stop, names='value', type='change')
    stepsize_slider.observe(stop, names='value', type='change')
    nsteps_slider.observe(stop, names='value', type='change')
    px_slider.observe(stop, names='value', type='change')

    # group widgets
    play_wdgt = ipw.HBox([run_btn, play])
    ctrl_widgets = ipw.VBox([ndots_slider, stepsize_slider, nsteps_slider, px_slider, play_wdgt,  traj_chkbox, frame_slider])

    # frame_idx = 0
    # use Output to wrap the plot for better layout
    plotup_out = ipw.Output()
    with plotup_out:
        fig_up, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True, figsize=(6,3))
        plt.show()

    plotdwn_out = ipw.Output()
    with plotdwn_out:
        fig_dwn, ax3 = plt.subplots(constrained_layout=True, figsize=(3,2.5))
        plt.show()

    # This gives more space to the plots by hidding the default interactive toolbar (caused by %matplotlib widget)
    fig_up.canvas.toolbar_visible = False
    fig_up.canvas.header_visible = False
    fig_up.canvas.footer_visible = False
    fig_dwn.canvas.toolbar_visible = False
    fig_dwn.canvas.header_visible = False
    fig_dwn.canvas.footer_visible = False

    initialize_plot()
    display(ipw.VBox([ipw.HBox([plotup_out]), ipw.HBox([ctrl_widgets, plotdwn_out])]))


