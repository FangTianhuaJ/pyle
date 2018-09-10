from Queue import Empty
from multiprocessing import Process, Queue
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import widgets
from scipy import interpolate

from labrad.units import Unit
V, mV, us, GHz, rad = [Unit(s) for s in ('V', 'mV', 'us', 'GHz', 'rad')]
from pyle.dataking import utilMultilevels as ml
from pyle.fitting import fitting


def adjust_s_scanning(qubit, data, qnd=False):
    f, phase = data.T
    traces = [{'x':f, 'y': phase, 'args':('b.',)}]
    if qnd:
        params = [{'name': 'qnd_readout frequency', 'val': qubit['qnd_readout frequency'][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    else:
        params = [{'name': 'readout frequency', 'val': qubit['readout frequency'][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        if qnd:
            qubit['qnd_readout frequency'] = result['qnd_readout frequency']*GHz
        else:
            qubit['readout frequency'] = result['readout frequency']*GHz


def adjust_phase(qubit, data):
    fb, left, right = data.T
    traces = [{'x': fb, 'y': left, 'args': ('b.',)},
              {'x': fb, 'y': right, 'args': ('r.',)}]
    params = [{'name':'adc adjusted phase', 'val': qubit['adc adjusted phase'][rad], 'range':(-np.pi, np.pi), 'axis':'y', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['adc adjusted phase'] = (-2 - result['adc adjusted phase'])*rad + qubit['adc adjusted phase']
        if qubit['adc adjusted phase']>np.pi:
            qubit['adc adjusted phase'] = qubit['adc adjusted phase']-2*np.pi
        elif qubit['adc adjusted phase']<-np.pi:
            qubit['adc adjusted phase'] = qubit['adc adjusted phase']+2*np.pi

def adjust_phase_arc(qubit, data):
    fb, left, right = data.T
    traces = [{'x': fb, 'y': left, 'args': ('b.',)},
              {'x': fb, 'y': right, 'args': ('r.',)}]
    params = [{'name': 'operate', 'val': qubit['biasOperate'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'b'},
              {'name': 'readout', 'val': qubit['biasReadout'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'reset0', 'val': qubit['biasReset'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'},
              {'name': 'reset1', 'val': qubit['biasReset'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'm'},
              {'name': 'Phase', 'val': qubit['critical phase'][rad], 'range': (-np.pi,np.pi), 'axis': 'y', 'color': 'k'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['operate']*V
        qubit['biasReadout'] = result['readout']*V
        qubit['biasReset'] = [result['reset0']*V, result['reset1']*V] * 2
        qubit['critical phase'] = result['Phase']*rad

def adjust_squid_steps(qubit, data):
    fb, low, high = data.T
    traces = [{'x': fb, 'y': low, 'args': ('b.',)},
              {'x': fb, 'y': high, 'args': ('r.',)}]
    params = [{'name': 'operate', 'val': qubit['biasOperate'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'b'},
              {'name': 'readout', 'val': qubit['biasReadout'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'reset0', 'val': qubit['biasReset'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'},
              {'name': 'reset1', 'val': qubit['biasReset'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'm'},
              {'name': 'timing0', 'val': qubit['squidSwitchIntervals'][0][0][us], 'range': (0,60), 'axis': 'y', 'color': 'k'},
              {'name': 'timing1', 'val': qubit['squidSwitchIntervals'][0][1][us], 'range': (0,60), 'axis': 'y', 'color': 'gray'},
              {'name': 'Edge_left', 'val': qubit['squidEdges'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'Edge_right', 'val': qubit['squidEdges'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['operate']*V
        qubit['biasReadout'] = result['readout']*V
        qubit['biasReset'] = [result['reset0']*V, result['reset1']*V] * 2
        qubit['squidSwitchIntervals'] = [(result['timing0']*us, result['timing1']*us)]
        qubit['squidEdges'] = [result['Edge_left']*V,result['Edge_right']*V] #mark the edge of two branches of the same color. Converts voltage-to-Phi_not


def adjust_time(data):
    t, probs = data[:,0], data[:,1:].T
    traces = [{'x': t, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 't', 'val': (min(t)+max(t))/2, 'range': (min(t), max(t)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        return result['t']


def adjust_operate_bias(qubit, data):
    fb, prob = data.T
    traces = [{'x': fb, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': 'fb', 'val': qubit['biasOperate'][mV], 'range': (min(fb),max(fb)), 'axis': 'x', 'color': 'b'},
              {'name': 'step', 'val': qubit['biasStepEdge'][mV], 'range': (min(fb),max(fb)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['fb']*mV
        qubit['biasStepEdge'] = result['step']*mV


def adjust_scurve(qubit, data, states):
    colors = ['b','g','r','c','m','y','k']
    keynames=['measureAmp']+['measureAmp'+str(i) for i in list(np.arange(2,max(max(states)+2,2)))]
    mpa, probs = data.T[0], data.T[1:]
    traces = [{'x': mpa, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'mpa'+str(state+1), 'val': float(qubit[keynames[state]]), 'range': (min(mpa),max(mpa)), 'axis': 'x', 'color': colors[state]} for state in states]
    result = adjust(params, traces)
    if result is not None:
        for state in states:
            qubit[keynames[state]] = result['mpa'+str(state+1)]


def adjust_visibility(qubit, data, states):
    numstates=len(states)
    mpas, probs, visis = data.T[0], data.T[1:numstates], data.T[numstates:]
    colors = ['b','g','r','c','m','y','k']
    keynames=['measureAmp']+['measureAmp'+str(i) for i in list(np.arange(2,max(max(states)+1,2)))]
    #We have to make sure that the mpa axis is monotonically increasing for scipy.interpolation.interp1d to work properly
    if mpas[0]>mpas[-1]:        #If mpas runs negatively
        mpas = mpas[::-1]       #Reverse it's order
        probs = probs[:,::-1]   #and also reverse the order of the probabilities.
        visis = visis[:,::-1]   #and also reverse the order of the visibilities.
    traces = [{'x':mpas, 'y':vis, 'args': ('.-',)} for vis in visis]+[{'x':mpas, 'y':prob, 'args': ('.-',)} for prob in probs]
    params = [{'name':'mpa'+str(state), 'val': float(qubit[keynames[state-1]]), 'range': (min(mpas),max(mpas)), 'axis': 'x', 'color': colors[state-1]} for state in states[1:]]
    result = adjust(params, traces)
    if result is not None:
        for state in states[1:]:
            qubit[keynames[state-1]] = result['mpa'+str(state)]


def adjust_frequency(qubit, data, paramName=None):
    if paramName is None:
        paramName = 'f10'
    f, prob = data.T
    traces = [{'x': f, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': paramName, 'val': qubit[paramName][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit[paramName] = result[paramName]*GHz


def adjust_frequency_02(qubit, data):
    f10 = qubit['f10'][GHz]
    f21 = qubit['f21'][GHz]
    f20_2ph = (f21 + f10) / 2
    f, probs = data.T[0], data.T[1:]
    traces = [{'x': f, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'f10', 'val': f10, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'},
              {'name': 'f20_2ph', 'val': f20_2ph, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        f10 = result['f10']
        f20_2ph = result['f20_2ph']
        f21 = 2*f20_2ph - f10
        qubit['f10'] = f10*GHz
        qubit['f21'] = f21*GHz


def adjust_fc(qubit, data):
    f10 = qubit['f10'][GHz]
    fc = qubit['fc'][GHz]
    f, probs = data.T[0], data.T[1:]
    traces = [{'x': f, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'f10', 'val': qubit['f10'][GHz], 'range': (min(f)-.2,max(f)+.2), 'axis': 'x', 'color': 'b'},
              {'name': 'fc', 'val': qubit['fc'][GHz], 'range': (min(f)-.2,max(f)+.2), 'axis': 'x', 'color': 'g'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['f10'] = result['f10']*GHz
        qubit['fc'] = result['fc']*GHz


def adjust_rabihigh(qubit, data, state=1):
    rabiheight, probs = data.T
    traces = [{'x': rabiheight, 'y': probs, 'args': ('.-',)}]
    params = [{'name': 'maxRabi', 'val': float(qubit[ml.saveKeyNumber('piAmp',state)]), 'range': (min(rabiheight),max(rabiheight)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        return result['maxRabi']
    else:
        return None

def adjust_piZ(qubit, data, state=1):
    zAmp, probs = data.T
    key = ml.saveKeyNumber('piAmpZ',state)
    traces = [{'x': zAmp, 'y': probs, 'args': ('.-',)}]
    params = [{'name': 'bestAmp', 'val': float(qubit[key]), 'range': (min(zAmp),max(zAmp)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        print 'Old %s: %f' %(key, qubit[key])
        selectedAmp = result['bestAmp']
        qubit['piAmpZ'] = selectedAmp
        print 'New %s: %f' %(key, selectedAmp)


def adjust_cZControlPhaseCorrAmp(qubit, data):
    cZCorrAmpMax = qubit['cZControlPhaseCorrAmpMax']
    cZCorrAmpMin = qubit['cZControlPhaseCorrAmpMin']
    controlAmp, probs = data.T
    traces = [{'x': controlAmp, 'y': probs, 'args': ('b.-',)}]
    params = [{'name': 'MAX', 'val': float(qubit['cZControlPhaseCorrAmpMax']), 'range': (min(controlAmp),max(controlAmp)), 'axis': 'x', 'color': 'b'},
              {'name': 'MIN', 'val': float(qubit['cZControlPhaseCorrAmpMin']), 'range': (min(controlAmp),max(controlAmp)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        cZCorrAmpMax = result['MAX']
        cZCorrAmpMin = result['MIN']
        qubit['cZControlPhaseCorrAmp'] = cZCorrAmpMax
        qubit['cZControlPhaseCorrAmpMax'] = cZCorrAmpMax
        qubit['cZControlPhaseCorrAmpMin'] = cZCorrAmpMin

def adjust_cZTargetPhaseCorrAmp(qubit, data):
    cZCorrAmpMax = qubit['cZTargetPhaseCorrAmpMax']
    cZCorrAmpMin = qubit['cZTargetPhaseCorrAmpMin']
    targetAmp, probs = data.T
    traces = [{'x': targetAmp, 'y': probs, 'args': ('b.-',)}]
    params = [{'name': 'MAX', 'val': float(qubit['cZTargetPhaseCorrAmpMax']), 'range': (min(targetAmp),max(targetAmp)), 'axis': 'x', 'color': 'b'},
              {'name': 'MIN', 'val': float(qubit['cZTargetPhaseCorrAmpMin']), 'range': (min(targetAmp),max(targetAmp)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        cZCorrAmpMax = result['MAX']
        cZCorrAmpMin = result['MIN']
        qubit['cZTargetPhaseCorrAmp'] = cZCorrAmpMax
        qubit['cZTargetPhaseCorrAmpMax'] = cZCorrAmpMax
        qubit['cZTargetPhaseCorrAmpMin'] = cZCorrAmpMin

def adjust(*a, **kw):
    """
    params: 
        describe the x (or y) slides 
        a list of dict, every dict should have the key 'name', 'val', 'range', 'axis', 'color' 
    traces:
        the data to be plot
        a list of dict, every dict should have the key 'x', 'y'
    
    example:
    amp = np.linspace(0, 1, 101)
    probs = np.sin(amp*np.pi*4)**2
    traces = [{'x': amp, 'y': prob, 'args': ('b-',), 'kw': {'linewidth': 2.0}}]
    params = [{'name': 'MAX', 'val': 0.5, 'range': (min(amp), max(amp)), "axis": "x", "color": 'r'},
              {'name': 'Prob', 'val: 0.5, 'range': (min(probs), max(probs)), "axis": "y", "color": "g"}
            ]
    result = adjust(params, traces)
    """
    # return runInSubprocess(_adjustGeneric, *a, **kw)
    return _adjustGeneric(*a, **kw)

class Adjust(object):
    """
    Object oriented interface to the adjustment tool.  Simple to use:

    adj = Adjust()
    adj.plot(x1, y1, 'b')
    adj.plot(x2, y2, 'r')
    adj.x_param('xslide_name', 0, 0, 50, 'r')
    adj.y_param('yslide_name')
    result = adj.run()
    """
    def __init__(self):
        self.plots = []
        self.x_sliders = []
        self.y_sliders = []

    def plot(self, x, y, *args, **kw):
        self.plots.append((x,y,args, kw))

    def x_param(self, name, initial_val=None, min_val=None, max_val=None, color=None):
        self.x_sliders.append((name, initial_val, min_val, max_val, color))

    def y_param(self, name, initial_val=None, min_val=None, max_val=None, color=None):
        self.y_sliders.append((name, initial_val, min_val, max_val, color))

    def run(self):
        traces = []
        parameters = []
        x_range = [float('inf'), -float('inf')]
        y_range = [float('inf'), -float('inf')]
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        color_idx = 0
        for p in self.plots:
            traces.append({'x': p[0], 'y': p[1], 'args': p[2], 'kw': p[3]})
            x_range[0] = min(x_range[0], min(p[0]))
            x_range[1] = max(x_range[1], max(p[0]))
            y_range[0] = min(y_range[0], min(p[1]))
            y_range[1] = min(y_range[1], max(p[1]))
        for s in self.x_sliders:
            range_min = s[2] if s[2] is not None else x_range[0]
            range_max = s[3] if s[3] is not None else x_range[1]
            initial_value = s[1] if s[1] else range_min
            if s[4] is None:
                color = colors[color_idx % len(colors)]
                color_idx += 1
            else:
                color = s[4]
            parameters.append({'name': s[0],
                               'axis': 'x',
                               'val': initial_value,
                               'range': (range_min, range_max),
                               'color': color})

        for s in self.y_sliders:
            range_min = s[2] if s[2] is not None else y_range[0]
            range_max = s[3] if s[3] is not None else y_range[1]
            initial_value = s[1] if s[1] else range_min
            if s[4] is None:
                color = colors[color_idx % len(colors)]
                color_idx += 1
            else:
                color = s[4]
            parameters.append({'name': s[0],
                               'axis': 'y',
                               'val': initial_value,
                               'range': (range_min, range_max),
                               'color': color})
        return adjust(parameters, traces)

def _adjustGeneric(params, traces):
    xlines = []
    ylines = []
    xsliders = []
    ysliders = []

    xparams = [p for p in params if p['axis'] == 'x']
    yparams = [p for p in params if p['axis'] == 'y']

    result = [None]
    finished = [False]

    sh = 0.03
    dy = 0.04

    sw = 0.025
    dx = 0.03

    top = 0.95
    left = 0.08
    bottom = sh + dy*(len(xparams)+2)
    right = 1 - sw - dx*(len(yparams)+1)

    bgap = 0.1
    bw = (right - left - bgap)/2.0

    # plot the original data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top=top, left=left, bottom=bottom, right=right)
    plottedTraces = []
    for trace in traces:
        args = trace.get('args', ())
        kw = trace.get('kw', {})
        plottedTrace = ax.plot(trace['x'], trace['y'], *args, **kw)
        plottedTraces.append(plottedTrace)
    ax.grid(True)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # draw lines and create sliders for adjustable params
    for i, p in enumerate(xparams):
        val = p['val']
        x = [val, val] if p['axis'] == 'x' else xlim
        y = [val, val] if p['axis'] == 'y' else ylim
        line, = ax.plot(x, y, c=p['color'])
        xlines.append(line)
        min, max = p['range']
        slider_ax = fig.add_axes([left, sh+dy*(len(xparams)-i), right-left, sh])
        s = widgets.Slider(slider_ax, p['name'], min, max, p['val'], valfmt='%0.3f', color=p['color'])
        xsliders.append(s)

    for i, p in enumerate(yparams):
        val = p['val']
        x = [val, val] if p['axis'] == 'x' else xlim
        y = [val, val] if p['axis'] == 'y' else ylim
        line, = ax.plot(x, y, c=p['color'])
        ylines.append(line)
        min, max = p['range']
        slider_ax = fig.add_axes([1-sw-dx*(len(yparams)-i), bottom, sw, top-bottom])
        s = VerticalSlider(slider_ax, p['name'], min, max, p['val'], valfmt='%0.3f', color=p['color'])
        ysliders.append(s)

    # create save and cancel buttons
    btn_ax = fig.add_axes([left, sh, bw, sh])
    save_btn = widgets.Button(btn_ax, 'Save')

    btn_ax = fig.add_axes([right-bw, sh, bw, sh])
    cancel_btn = widgets.Button(btn_ax, 'Cancel')

    # event callbacks
    def update(val):
        for p, line, slider in zip(xparams, xlines, xsliders):
            val = p['val'] = slider.val
            x = [val, val] if p['axis'] == 'x' else xlim
            y = [val, val] if p['axis'] == 'y' else ylim
            line.set_xdata(x)
            line.set_ydata(y)
            # This is a little hacky and only works for x-axis as of right now -JK
            if p.has_key('update'):
                tracesToUpdate = p['update']
                for k,plottedTrace,trace in zip(range(len(plottedTraces)),plottedTraces,traces):
                    if k in tracesToUpdate:
                        plottedTrace[0].set_xdata(trace['x']-val)
        for p, line, slider in zip(yparams, ylines, ysliders):
            val = p['val'] = slider.val
            x = [val, val] if p['axis'] == 'x' else xlim
            y = [val, val] if p['axis'] == 'y' else ylim
            line.set_xdata(x)
            line.set_ydata(y)
        plt.draw() # redraw the figure

    def save(e):
        result[0] = dict((p['name'], p['val']) for p in params)
        finished[0] = True
        # plt.close(fig)

    def cancel(e):
        finished[0] = True
        # plt.close(fig)

    # hook up events
    for slider in xsliders + ysliders:
        slider.on_changed(update)
    cancel_btn.on_clicked(cancel)
    save_btn.on_clicked(save)

    # initial update
    update(None)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()
    try:
        while not finished[0]:
            rv = fig.waitforbuttonpress(timeout=0.1)
    except Exception: # Closed figure raises TclError
        pass
    plt.close(fig)
    return result[0]


class VerticalSlider(widgets.Widget):
    """
    A vertical slider representing a floating point range

    The following attributes are defined
      ax     : the slider axes.Axes instance
      val    : the current slider value
      vline  : a Line2D instance representing the initial value
      poly   : A patch.Polygon instance which is the slider
      valfmt : the format string for formatting the slider text
      label  : a text.Text instance, the slider label
      closedmin : whether the slider is closed on the minimum
      closedmax : whether the slider is closed on the maximum
      slidermin : another slider - if not None, this slider must be > slidermin
      slidermax : another slider - if not None, this slider must be < slidermax
      dragging : allow for mouse dragging on slider

    Call on_changed to connect to the slider event
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None, slidermax=None,
                 dragging=True, **kwargs):
        """
        Create a slider from valmin to valmax in axes ax;

        valinit -  the slider initial position

        label - the slider label

        valfmt - used to format the slider value

        closedmin and closedmax - indicate whether the slider interval is closed

        slidermin and slidermax - be used to contrain the value of
          this slider to the values of other sliders.

        additional kwargs are passed on to self.poly which is the
        matplotlib.patches.Rectangle which draws the slider.  See the
        matplotlib.patches.Rectangle documentation for legal property
        names (eg facecolor, edgecolor, alpha, ...)
          """
        self.ax = ax

        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axhspan(valmin,valinit,0,1, **kwargs)

        self.hline = ax.axhline(valinit,0,1, color='r', lw=1)


        self.valfmt = valfmt
        ax.set_yticks([])
        ax.set_ylim((valmin, valmax))
        ax.set_xticks([])
        ax.set_navigate(False)

        ax.figure.canvas.mpl_connect('button_press_event', self._update)
        if dragging:
            ax.figure.canvas.mpl_connect('motion_notify_event', self._update)
        # TODO fix text
        self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                             verticalalignment='center',
                             horizontalalignment='right')

        self.valtext = ax.text(1.02, 0.5, valfmt%valinit,
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax

    def _update(self, event):
        'update the slider position'
        if event.button !=1: return
        if event.inaxes != self.ax: return
        val = event.ydata
        if not self.closedmin and val <= self.valmin: return
        if not self.closedmax and val >= self.valmax: return

        if self.slidermin is not None:
            if val <= self.slidermin.val: return

        if self.slidermax is not None:
            if val >= self.slidermax.val: return

        self.set_val(val)

    def set_val(self, val):
        xy = self.poly.xy
        xy[1] = 0, val
        xy[2] = 1, val
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt%val)
        if self.drawon: self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: return
        for _cid, func in self.observers.items():
            func(val)

    def on_changed(self, func):
        """
        When the slider valud is changed, call this func with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        'remove the observer with connection id cid'
        try: del self.observers[cid]
        except KeyError: pass

    def reset(self):
        "reset the slider to the initial value if needed"
        if (self.val != self.valinit):
            self.set_val(self.valinit)


def runInSubprocess(f, *a, **kw):
    q = Queue()
    p = Process(target=_run, args=(q, f, a, kw))
    p.start()
    while True:
        try:
            result = q.get(timeout=0.3)
            break
        except Empty:
            if not p.is_alive():
                raise Exception('Child process died!')
    p.join()
    return result


def _run(q, f, a, kw):
    q.put(f(*a, **kw))

