import numpy as np
import numpy as sp
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
from pyle import math as pym
import itertools
import warnings

def __bar3d(self, x, y, z, dx, dy, dz, color='b', zsort='average', shade=True, *args, **kwargs):
    '''
    Generate a 3D bar, or multiple bars.

    When generating multiple bars, x, y, z have to be arrays.
    dx, dy, dz can be arrays or scalars.

    *color* can be:

     - A single color value, to color all bars the same color.

     - An array of colors of length N bars, to color each bar
       independently.

     - An array of colors of length 6, to color the faces of the
       bars similarly.

     - An array of colors of length 6 * N bars, to color each face
       independently.

     When coloring the faces of the boxes specifically, this is
     the order of the coloring:

      1. -Z (bottom of box)
      2. +Z (top of box)
      3. -Y
      4. +Y
      5. -X
      6. +X

    Keyword arguments are passed onto
    :func:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    '''
    had_data = self.has_data()

    if not np.iterable(x):
        x = [x]
    if not np.iterable(y):
        y = [y]
    if not np.iterable(z):
        z = [z]

    if not np.iterable(dx):
        dx = [dx]
    if not np.iterable(dy):
        dy = [dy]
    if not np.iterable(dz):
        dz = [dz]

    if len(dx) == 1:
        dx = dx * len(x)
    if len(dy) == 1:
        dy = dy * len(y)
    if len(dz) == 1:
        dz = dz * len(z)

    if len(x) != len(y) or len(x) != len(z):
        warnings.warn('x, y, and z must be the same length.')

    minx, miny, minz = 1e20, 1e20, 1e20
    maxx, maxy, maxz = -1e20, -1e20, -1e20

    polys = []
    for xi, yi, zi, dxi, dyi, dzi in zip(x, y, z, dx, dy, dz):
        minx = min(xi, minx)
        maxx = max(xi + dxi, maxx)
        miny = min(yi, miny)
        maxy = max(yi + dyi, maxy)
        minz = min(zi, minz)
        maxz = max(zi + dzi, maxz)

        polys.extend([
            ((xi, yi, zi), (xi + dxi, yi, zi),
                (xi + dxi, yi + dyi, zi), (xi, yi + dyi, zi)),
            ((xi, yi, zi + dzi), (xi + dxi, yi, zi + dzi),
                (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

            ((xi, yi, zi), (xi + dxi, yi, zi),
                (xi + dxi, yi, zi + dzi), (xi, yi, zi + dzi)),
            ((xi, yi + dyi, zi), (xi + dxi, yi + dyi, zi),
                (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

            ((xi, yi, zi), (xi, yi + dyi, zi),
                (xi, yi + dyi, zi + dzi), (xi, yi, zi + dzi)),
            ((xi + dxi, yi, zi), (xi + dxi, yi + dyi, zi),
                (xi + dxi, yi + dyi, zi + dzi), (xi + dxi, yi, zi + dzi)),
        ])

    facecolors = []
    if color is None:
        # no color specified
        facecolors = [None] * len(x)
    elif len(color) == len(x):
        # bar colors specified, need to expand to number of faces
        for c in color:
            facecolors.extend([c] * 6)
    else:
        # a single color specified, or face colors specified explicitly
        facecolors = list(mpl.colors.colorConverter.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= (6 * len(x))

    normals = self._generate_normals(polys)
    if shade:
        sfacecolors = self._shade_colors(facecolors, normals)
    else:
        sfacecolors = facecolors
    col = art3d.Poly3DCollection(polys,
                                 zsort=zsort,
                                 facecolor=sfacecolors,
                                 *args, **kwargs)
    self.add_collection(col)

    self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)

    return col

# monkey patch
Axes3D.bar3d = __bar3d

def arrow(rho, scale=1.0, color=None, width=0.05, headwidth=0.1, headlength=0.1, chopN=None, amp=1.0, collect=False):
    plt.figure()
    rho=rho.copy()
    s=np.shape(rho)
    if chopN!=None:
        rho = rho[:chopN,:chopN]
    s=np.shape(rho)
    rho = rho*amp
    ax = plt.gca()
    ax.set_aspect(1.0)
    pos = ax.get_position()
    r = np.real(rho)
    i = np.imag(rho)
    x = np.arange(s[0])[None,:] + 0*r
    y = np.arange(s[1])[:,None] + 0*i
    plt.quiver(x,y,r,i,units='x',scale=1.0/scale, width=width, headwidth=headwidth, headlength=headlength, color=color)
    plt.xticks(np.arange(s[1]))
    plt.yticks(np.arange(s[0]))
    plt.xlim(-0.9999,s[1]-0.0001)
    plt.ylim(-0.9999,s[0]-0.0001)
    if collect:
        return rho


def manhattan3d(rho, axesLabels=None, d=0.1, cmap=None, vmin=-1, vmax=1, zmin=-1, zmax=1):
    if cmap is None:
        cmap = cm.RdYlBu_r
    norm = mpl.colors.Normalize(vmin, vmax)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_proj_type('ortho')

    w, h = rho.shape
    if axesLabels is None:
        axesLabels = [bin(i)[2:].rjust(int(np.log2(w)),'0') for i in range(w)]
    xrange = range(w)
    yrange = range(h)
    yrange.reverse()
    for x in xrange:
        for y in yrange:
            rhoVal = rho[x,y]
            ax.bar3d([x+d/2], [y+d/2], [0], 1-d, 1-d, np.real(rhoVal), cmap(norm(rhoVal.real)),
                     alpha=1.0, shade=False, edgecolor='k', linewidth=0.5)

    ax.set_zlim3d(zmin, zmax)

    ax.w_xaxis.set_major_locator(ticker.FixedLocator(np.arange(w)+0.5))
    ax.w_xaxis.set_ticklabels(axesLabels)

    ax.w_yaxis.set_major_locator(ticker.FixedLocator(np.arange(w)+0.5))
    ax.w_yaxis.set_ticklabels(axesLabels)

    # remove background and the grid
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))
    ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))
    ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))

    return fig


def manhattan2d(rho, theory, cmap=None):
    fig = plt.figure(figsize=(11,6))
    axabs_ex = fig.add_subplot(2,3,1)
    axre_ex = fig.add_subplot(2,3,2)
    axim_ex = fig.add_subplot(2,3,3)
    axabs_th = fig.add_subplot(2,3,4)
    axre_th = fig.add_subplot(2,3,5)
    axim_th = fig.add_subplot(2,3,6)
    plt.subplots_adjust(right=0.8)

    def set_ticklabels(ax, n=3):
        ticks = [i+0.5 for i in range(2**n)]
        labels = [bin(i)[2:].rjust(n,'0') for i in range(2**n)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    for ax in [axabs_ex, axre_ex, axim_ex, axabs_th, axre_th, axim_th]:
        set_ticklabels(ax)

    absmax = lambda a: max(np.max(np.max(np.abs(a.real))), np.max(np.max(np.abs(a.imag))))

    vmax_ex = absmax(rho)
    vmax_th = absmax(theory)

    if cmap is None:
        cmap = cm.get_cmap('RdYlBu')
    opts_ex = {'vmin': -vmax_ex, 'vmax': vmax_ex, 'cmap': cmap}
    opts_th = {'vmin': -vmax_th, 'vmax': vmax_th, 'cmap': cmap}

    axabs_ex.pcolor(np.abs(rho), **opts_ex)
    axre_ex.pcolor(rho.real, **opts_ex)
    axim_ex.pcolor(rho.imag, **opts_ex)

    axabs_th.pcolor(np.abs(theory), **opts_th)
    axre_th.pcolor(theory.real, **opts_th)
    axim_th.pcolor(theory.imag, **opts_th)

    axabs_ex.set_title('abs')
    axre_ex.set_title('real')
    axim_ex.set_title('imag')

    axabs_ex.set_ylabel('experiment')
    axabs_th.set_ylabel('theory')

    norm_ex = mpl.colors.Normalize(vmin=-vmax_ex, vmax=vmax_ex)
    norm_th = mpl.colors.Normalize(vmin=-vmax_th, vmax=vmax_th)

    rect_ex = axim_ex.get_position().get_points()
    axcb_ex = fig.add_axes([0.85, rect_ex[0,1], 0.05, rect_ex[1,1]-rect_ex[0,1]])
    mpl.colorbar.ColorbarBase(axcb_ex, norm=norm_ex, cmap=cmap)

    rect_th = axim_th.get_position().get_points()
    axcb_th = fig.add_axes([0.85, rect_th[0,1], 0.05, rect_th[1,1]-rect_th[0,1]])
    mpl.colorbar.ColorbarBase(axcb_th, norm=norm_th, cmap=cmap)

    return fig


def matrix_hist3d(rho, cmap=None, vlimit=None, zlimit=None, space=0.2, plot_labels=True,
                  label_zpos=None, label_colors=None, colorbar=False):
    """
    plot histrogram of matrix (real part) especially for density matrix

    @param rho: density matrix
    @param cmap: colormap
    @param vlimit: (vmin, vmax), the min and max for colormap
    @param zlimit: (zmin, zmax), the min and max for z-axis
    @param space:  space between neighbour bar
    @param plot_labels: bool, show colorblob labels like Fig.4 in
           RamiBarends, Nature,2014, doi:10.1038/nature13171
    @param label_zpos: the z-val of the colorblob label
    @param label_colors: the colors represent each qubit, a list of colors
    @param colorbar: bool
    @return: figure
    """
    # colors in Barends Nature 2014 (doi:10.1038/nature13171)
    LABELCOLORS = ["#c22326", "#033e86", "#006d36", "#5a278b", "#a67f16"]
    # colors in default matplotlib
    # LABELCOLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    if cmap is None:
        cmap = mpl.cm.RdYlBu_r

    if vlimit is None:
        vmin, vmax = np.min(rho), np.max(rho)
    else:
        vmin, vmax = vlimit[0], vlimit[1]
    if zlimit is None:
        zmin, zmax = vmin-0.1, vmax+0.1
    else:
        zmin, zmax = zlimit[0], zlimit[1]

    norm = mpl.colors.Normalize(vmin, vmax)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho') # for matplotlib 2.1
    w, h = rho.shape

    x_range = range(w)
    y_range = range(h)
    d = space
    xpos, ypos = np.meshgrid(x_range, y_range)
    xpos = xpos.T.flatten() + d / 2.0
    ypos = ypos.T.flatten() + d / 2.0
    zpos = np.zeros_like(xpos)
    dx = dy = (1 - d) * np.ones_like(xpos)
    dz = np.real(rho.flatten())
    colors = cmap(norm(dz))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=1.0, shade=False,
             linewidth=0.5, edgecolor='k')

    ax.set_zlim3d(zmin, zmax)

    if plot_labels:
        assert w == h
        n_qubit = int(np.log2(w))
        assert 2**n_qubit == w

        # plot xy marker
        color0 = "#e6e6e6"  # color for |0>
        if label_colors is None:
            label_colors = LABELCOLORS[:n_qubit]
        if len(label_colors) == 1:
            label_colors = label_colors * n_qubit
        assert len(label_colors) == n_qubit
        colors = [(color0, c) for c in label_colors]
        if label_zpos is None:
            label_zpos = zmin

        # marker of X axis
        color_iter = itertools.product(*colors)
        for x, curr_colors in zip(range(w), color_iter):
            for c, y in zip(curr_colors, range(-n_qubit, 0, 1)):
                ax.bar3d(x + d / 2.0, y + d / 2.0 - space, label_zpos,
                         1 - d, 1 - d, 0, color=c, alpha=1.0, linewidth=0.0)
        # marker of Y axis
        color_iter = itertools.product(*colors)
        for y, curr_colors in zip(range(h), color_iter):
            for c, x in zip(curr_colors, range(w + n_qubit - 1, w - 1, -1)):
                ax.bar3d(x + d / 2.0 + space, y + d / 2.0, label_zpos,
                         1 - d, 1 - d, 0, color=c, alpha=1.0, linewidth=0.0)

    # x-axis, y-axis
    ax.plot([0, w + d / 2], [-d / 2, -d / 2], [0, 0], color='k', lw=0.7)
    ax.plot([w + d / 2, w + d / 2], [-d / 2, h], [0, 0], color='k', lw=0.7)
    # ax.plot([0, 0], [0, 0], [zmin, zmax], color='k', lw=1) # z-axis

    # get rid of the pane background
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))
    ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))
    ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.0))

    # Get rid of the spines
    # ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=0.0)
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    return fig

def circles(rho):

    #TODO: I can't seem to change the linewidth of the
    #      lines that denote the phase

    dim = len(rho)
    amps = abs(rho)
    phases = np.angle(rho)

    fig = plt.figure(figsize = (7,6))
    ax = plt.axes([0,0,1,1])
    ax.set_ylim(-2, 2*dim)
    ax.set_xlim(-2, 2*dim)

    pos = np.mgrid[2*(dim-1):-2:-2, 0:2*dim:2]

    circlePatches = []
    circleColors = []
    for rownum, row in enumerate(rho):
        for elementnum, element in enumerate(row):
            art = mpl.patches.Circle(
                             [pos[1][rownum,elementnum],
                              pos[0][rownum,elementnum]],
                              amps[rownum,elementnum])
            circleColors.append(amps[rownum, elementnum])
            circlePatches.append(art)

    # this next crap is for scaling the colors, it's a
    # dumb way to do it.
    circlePatches.append(mpl.patches.Circle([0,0],0))
    circlePatches.append(mpl.patches.Circle([-20,-20],1))
    circleColors.append(0)
    circleColors.append(1)

    for rownum, row in enumerate(rho):
        for elementnum, element in enumerate(row):
            dx = amps[rownum, elementnum]*np.cos(phases[rownum, elementnum])
            dy = amps[rownum, elementnum]*np.sin(phases[rownum, elementnum])
            x = pos[1][rownum,elementnum]
            y = pos[0][rownum,elementnum]
            art = mpl.lines.Line2D([x, x+dx],[y, y+dy], lw = 3, color='k')
            ax.add_line(art)

    circleCollection = mpl.collections.PatchCollection(circlePatches,
                                       cmap = mpl.cm.RdBu)

    circleCollection.set_array(np.array(circleColors))
    ax.add_collection(circleCollection)
    plt.colorbar(circleCollection)


def drawBlochSphere(bloch=True, labels=True, ax=None):
    # takes a sequence and projects the trajectory of the qubit
    # onto the bloch sphere, works for T1/T2!
    if ax is None:
        ax = plt3.Axes3D(plt.figure())
    ax.view_init(30, 30)
    color = '#FFDDDD'
    alpha = 0.2
    #draw a bloch sphere
    if bloch:
        u = np.linspace(0, 2*np.pi, 41)
        v = np.linspace(0, np.pi, 41)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.set_axis_off()
        lim = 0.85
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(-lim, lim)
        ax.plot_surface(x, y, z,  rstride=2, cstride=2, color = color, alpha = alpha, linewidth=0)#, cmap = cm.PuBu) #, alpha = 1.0)
        ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color = 'gray',
                          alpha = alpha, linewidth=1.0)
        ax.plot(1.0*np.cos(u), 1.0*np.sin(u), zs=0, zdir='z', lw=0.8, color='gray', alpha=0.7)
        ax.plot(1.0*np.cos(u), 1.0*np.sin(u), zs=0, zdir='y', lw=0.8, color='gray', alpha=0.7)
        ax.plot(1.0*np.cos(u), 1.0*np.sin(u), zs=0, zdir='x', lw=0.8, color='gray', alpha=0.7)
        ax.set_aspect('equal')

    #Axis label
    if labels:
        pos = 1.1
        opts = {'color': 'black', 'fontsize': 20}
        ax.text(0, 0, pos,  r'$\left| 0\right\rangle$', **opts)
        ax.text(0, 0, -pos, r'$\left| 1\right\rangle$', **opts)
        ax.text(pos, 0, 0,  r'$\left| 0\right\rangle + \left|1\right\rangle$', **opts)
        ax.text(-pos, 0, 0, r'$\left| 0\right\rangle - \left|1\right\rangle$', **opts)
        ax.text(0, pos, 0,  r'$\left| 0\right\rangle +i\left|1\right\rangle$', **opts)
        ax.text(0, -pos, 0, r'$\left| 0\right\rangle -i\left|1\right\rangle$', **opts)
        # ax.text(pos+0.2, 0, 0, r'$x$', **opts) # x
        # ax.text(0, pos+0.2, 0, r'$y$', **opts) # x

    #Axis
    opts = {'color': 'gray', 'linewidth': 1.0}
    lim = 1.0
    ax.plot([-lim,lim],[0,0],[0,0], **opts)
    ax.plot([0,0],[-lim,lim],[0,0], **opts)
    ax.plot([0,0],[0,0],[-lim,lim], **opts)
    ax.set_aspect('equal')

    return ax

def plotTrajectory(rhos, state=1, labels=True):
    blochs = np.array([pym.rho2bloch(rho) for rho in rhos])

    ax = drawBlochSphere(True, labels)

    if state is None:
        state = range(4)
    if isinstance(state,
    int):
        state = [state]

    colors = ['r', 'b', 'g', '#CC6600']
    markers = ['o', 's', 'd', '^']
    for s, c, m in zip(state, colors, markers):
        ax.plot(blochs[:,0].real, blochs[:,1].real, blochs[:,2].real, m,
                color=c, markersize=7, markeredgecolor='none')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plotVector(rhos, ax=None):
    if ax is None:
        ax = drawBlochSphere(True, True)
    if isinstance(rhos, list):
        rhos = np.array(rhos)

    if rhos.ndim == 3:
        blochs = np.array([pym.rho2bloch(rho) for rho in rhos])
        xs, ys, zs = blochs.T
    elif rhos.ndim == 2:
        blochs = pym.rho2bloch(rhos)[:,None]
        xs, ys, zs = blochs
    else:
        raise Exception("dimension of rhos should be 2 or 3")
    colors = ['g', '#CC6600', 'r', 'b']
    width = 3
    style = '-|>'
    mutation = 20
    opts={'linewidth': width, 'arrowstyle': style, 'mutation_scale': mutation}
    for x,y,z, c in zip(xs, ys, zs, colors):
        x = x.real*np.array([0,1])
        y = y.real*np.array([0,1])
        z = z.real*np.array([0,1])
        print x
        a = Arrow3D(x, y, z, color=c, **opts)
        ax.add_artist(a)

# set up operator bases
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
dot3 = lambda a, b, c: np.dot(np.dot(a, b), c)

basis1 = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

labels2 = ['XI', 'YI', 'ZI',
           'IX', 'IY', 'IZ',
           'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
basis2 = dict((label, reduce(np.kron, [basis1[c] for c in label])) for label in labels2)

labels3 = ['XII', 'YII', 'ZII',
           'IXI', 'IYI', 'IZI',
           'IIX', 'IIY', 'IIZ',

           'XXI', 'XYI', 'XZI', 'YXI', 'YYI', 'YZI', 'ZXI', 'ZYI', 'ZZI',
           'XIX', 'XIY', 'XIZ', 'YIX', 'YIY', 'YIZ', 'ZIX', 'ZIY', 'ZIZ',
           'IXX', 'IXY', 'IXZ', 'IYX', 'IYY', 'IYZ', 'IZX', 'IZY', 'IZZ',

           'XXX', 'XXY', 'XXZ', 'XYX', 'XYY', 'XYZ', 'XZX', 'XZY', 'XZZ',
           'YXX', 'YXY', 'YXZ', 'YYX', 'YYY', 'YYZ', 'YZX', 'YZY', 'YZZ',
           'ZXX', 'ZXY', 'ZXZ', 'ZYX', 'ZYY', 'ZYZ', 'ZZX', 'ZZY', 'ZZZ']
basis3 = dict((label, reduce(np.kron, [basis1[c] for c in label])) for label in labels3)

labels4 = ['XIII', 'YIII', 'ZIII',
           'IXII', 'IYII', 'IZII',
           'IIXI', 'IIYI', 'IIZI',
           'IIIX', 'IIIY', 'IIIZ',

           'XXII', 'XYII', 'XZII', 'YXII', 'YYII', 'YZII', 'ZXII', 'ZYII', 'ZZII',
           'XIXI', 'XIYI', 'XIZI', 'YIXI', 'YIYI', 'YIZI', 'ZIXI', 'ZIYI', 'ZIZI',
           'XIIX', 'XIIY', 'XIIZ', 'YIIX', 'YIIY', 'YIIZ', 'ZIIX', 'ZIIY', 'ZIIZ',
           'IXXI', 'IXYI', 'IXZI', 'IYXI', 'IYYI', 'IYZI', 'IZXI', 'IZYI', 'IZZI',
           'IXIX', 'IXIY', 'IXIZ', 'IYIX', 'IYIY', 'IYIZ', 'IZIX', 'IZIY', 'IZIZ',
           'IIXX', 'IIXY', 'IIXZ', 'IIYX', 'IIYY', 'IIYZ', 'IIZX', 'IIZY', 'IIZZ',

           'XXXI', 'XXYI', 'XXZI', 'XYXI', 'XYYI', 'XYZI', 'XZXI', 'XZYI', 'XZZI',
           'YXXI', 'YXYI', 'YXZI', 'YYXI', 'YYYI', 'YYZI', 'YZXI', 'YZYI', 'YZZI',
           'ZXXI', 'ZXYI', 'ZXZI', 'ZYXI', 'ZYYI', 'ZYZI', 'ZZXI', 'ZZYI', 'ZZZI',
           'XXIX', 'XXIY', 'XXIZ', 'XYIX', 'XYIY', 'XYIZ', 'XZIX', 'XZIY', 'XZIZ',
           'YXIX', 'YXIY', 'YXIZ', 'YYIX', 'YYIY', 'YYIZ', 'YZIX', 'YZIY', 'YZIZ',
           'ZXIX', 'ZXIY', 'ZXIZ', 'ZYIX', 'ZYIY', 'ZYIZ', 'ZZIX', 'ZZIY', 'ZZIZ',
           'XIXX', 'XIXY', 'XIXZ', 'XIYX', 'XIYY', 'XIYZ', 'XIZX', 'XIZY', 'XIZZ',
           'YIXX', 'YIXY', 'YIXZ', 'YIYX', 'YIYY', 'YIYZ', 'YIZX', 'YIZY', 'YIZZ',
           'ZIXX', 'ZIXY', 'ZIXZ', 'ZIYX', 'ZIYY', 'ZIYZ', 'ZIZX', 'ZIZY', 'ZIZZ',
           'IXXX', 'IXXY', 'IXXZ', 'IXYX', 'IXYY', 'IXYZ', 'IXZX', 'IXZY', 'IXZZ',
           'IYXX', 'IYXY', 'IYXZ', 'IYYX', 'IYYY', 'IYYZ', 'IYZX', 'IYZY', 'IYZZ',
           'IZXX', 'IZXY', 'IZXZ', 'IZYX', 'IZYY', 'IZYZ', 'IZZX', 'IZZY', 'IZZZ'

           'XXXX', 'XXXY', 'XXXZ', 'XXYX', 'XXYY', 'XXYZ', 'XXZX', 'XXZY', 'XXZZ',
           'XYXX', 'XYXY', 'XYXZ', 'XYYX', 'XYYY', 'XYYZ', 'XYZX', 'XYZY', 'XYZZ',
           'XZXX', 'XZXY', 'XZXZ', 'XZYX', 'XZYY', 'XZYZ', 'XZZX', 'XZZY', 'XZZZ',
           'YXXX', 'YXXY', 'YXXZ', 'YXYX', 'YXYY', 'YXYZ', 'YXZX', 'YXZY', 'YXZZ',
           'YYXX', 'YYXY', 'YYXZ', 'YYYX', 'YYYY', 'YYYZ', 'YYZX', 'YYZY', 'YYZZ',
           'YZXX', 'YZXY', 'YZXZ', 'YZYX', 'YZYY', 'YZYZ', 'YZZX', 'YZZY', 'YZZZ',
           'ZXXX', 'ZXXY', 'ZXXZ', 'ZXYX', 'ZXYY', 'ZXYZ', 'ZXZX', 'ZXZY', 'ZXZZ',
           'ZYXX', 'ZYXY', 'ZYXZ', 'ZYYX', 'ZYYY', 'ZYYZ', 'ZYZX', 'ZYZY', 'ZYZZ',
           'ZZXX', 'ZZXY', 'ZZXZ', 'ZZYX', 'ZZYY', 'ZZYZ', 'ZZZX', 'ZZZY', 'ZZZZ']
basis4 = dict((label, reduce(np.kron, [basis1[c] for c in label])) for label in labels4)


def E(op, rho):
    """Find the expectation value of an operator for a given density matrix."""
    return np.trace(np.dot(op, rho))


def pauli(rho, rho_th=None, d=0.1, phases=None, dpi=200, map=True):
    num_qubits = int(np.log2(rho.shape[0]))
    N = 4**num_qubits-1
    #basis = {2: basis2, 3: basis3, 4: basis4}[num_qubits]
    basis, labels = {2: (basis2, labels2),
                     3: (basis3, labels3)}[num_qubits]

    if phases is not None:
        z = pym.tensor(sp.linalg.expm(-1j*phase*Z) for phase in phases)
        rho = dot3(z, rho, z.conj().T)

    w = 1 - 2*d
    x = np.array(range(N)) + d
    xticks = np.array(range(N)) + 0.5
    #labels = sorted(basis.keys())
    if rho_th is not None:
        vals_th = [E(basis[op], rho_th) for op in labels]
    vals = [E(basis[op], rho) for op in labels]

    cmap = cm.get_cmap('RdYlBu')
    norm = mpl.colors.Normalize(-1, 1)

    fig = plt.figure(figsize=(10,3), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.add_artist(mpl.patches.Rectangle((-0.5, -1.1), 3.5, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(mpl.patches.Rectangle((6, -1.1), 3, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(mpl.patches.Rectangle((18, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    ax.add_artist(mpl.patches.Rectangle((36, -1.1), 27.5, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))

    #ax.add_artist(matplotlib.patches.Rectangle((3, -1.1), 3, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    #ax.add_artist(matplotlib.patches.Rectangle((9, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))
    #ax.add_artist(matplotlib.patches.Rectangle((27, -1.1), 9, 2.2, color=(0.85, 0.85, 0.85), zorder=-10))

    if rho_th is not None:
        if map:
            for xi, val in zip(x, vals_th):
                c = cmap(norm(val))
                c = (0.75, 0.75, 0.75)
                ax.bar([xi], [val], width=w, fc=c, ec=(0.3, 0.3, 0.3))
        else:
            ax.bar(x, vals_th, width=w, fc=(0.7, 0.7, 1), ec=(0.4, 0.4, 0.4))
    if map:
        for xi, val in zip(x, vals):
            c = cmap(norm(val))
            ax.bar([xi], [val], width=w, fc=c, ec=(0, 0, 0))
    else:
        ax.bar(x, vals, width=w, fc=(0, 0, 1), ec=(0, 0, 0))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, N+0.5)
    ax.set_ylim(-1.1, 1.1)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.12)

    rect = ax.get_position().get_points()
    axcb = fig.add_axes([0.93, rect[0,1], 0.02, rect[1,1]-rect[0,1]])
    mpl.colorbar.ColorbarBase(axcb, norm=norm, cmap=cmap)

    return fig
