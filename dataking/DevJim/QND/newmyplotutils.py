from pylab import subplot, axes, subplots_adjust, setp, imshow, plot, draw_if_interactive, gcf, xticks, yticks, gca, cm, fill
from numpy import shape, log, exp, arange, linspace, min, max, zeros, uint32, sqrt, array, dot
#from PIL import Image
from matplotlib.colors import LinearSegmentedColormap


def arrowhead(x,y,dx,dy,size=0.5,color='k'):
    s=sqrt(dx**2+dy**2)
    dx=-dx*size/s
    dy=-dy*size/s
    fill([x,x+dx-0.25*dy,x+0.75*dx,x+dx+0.25*dy],[y,y+dy+0.25*dx,y+0.75*dy,y+dy-0.25*dx],edgecolor=color,facecolor=color)
    

def panelplot(plots,xlabel=None,ylabel=None):
    n = len(plots)
    ax0=subplot(n,1,n,)
    hsp=0.05
    for i in range(n):
        if i>0:
            ax = subplot(n,1,n-i,sharex=ax0,sharey=ax0)
            setp(ax.get_xticklabels(), visible=False)
        plot(*plots[i])
    subplots_adjust(hspace=hsp)
    if xlabel is not None:
        ax0.set_xlabel(xlabel)
    if ylabel is not None:
        ax0.set_ylabel(ylabel,position=(0,0.5*(n+hsp*(n-1))))
    draw_if_interactive()
    return ax0


def customticks(ax=None,aboveticks=None,aboveticklabels=[],rightticks=None,rightticklabels=[]):
    """
    Make a second axes overlay ax (or the current axes if ax is None)
    sharing the xaxis.  The ticks for ax2 will be placed on the right,
    and the ax2 instance is returned.  See examples/two_scales.py
    """
    if ax is None:
        ax=gca()
    ax2 = gcf().add_axes(ax.get_position(), frameon=False)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())

    if aboveticks is None:
        aboveticks = []
    else:
        ax.xaxis.tick_bottom()

    if rightticks is None:
        rightticks = []
    else:
        ax.yaxis.tick_left()
        

    ax2.yaxis.tick_right()
    ax2.xaxis.tick_top()
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.set_label_position('top')
    xticks(aboveticks,aboveticklabels)
    yticks(rightticks,rightticklabels)

    draw_if_interactive()
    return ax2


            

def plot2d(data,extent=None,vmin=None,vmax=None,cmap=None,aspect='auto'):
    s = shape(data)
    if isinstance(extent,dict):
        Xmin = extent['Xmin']
        Xmax = extent['Xmax']
        Ymin = extent['Ymin']
        Ymax = extent['Ymax']
    elif isinstance(extent,tuple):
        Xmin = extent[0]
        Xmax = extent[1]
        Ymin = extent[2]
        Ymax = extent[3]
    else:
        Xmin = 0
        Xmax = s[1]-1
        Ymin = 0
        Ymax = s[0]-1
            
   
    if s[0]:
        hstepX = 0.5*(Xmax-Xmin)/(s[1]-1)
    elif Xmax > Xmin:
        hstepX = 0
    else:
        hstepX = 0.5
    if s[1]:
        hstepY = 0.5*(Ymax-Ymin)/(s[0]-1)
    elif Ymax > Ymin:
        hstepY = 0
    else:
        hstepY = 0.5

    imshow(data, aspect=aspect, origin='lower', cmap=cmap,
           interpolation = 'nearest',
           extent = (Xmin - hstepX,
                     Xmax + hstepX,
                     Ymin - hstepY,
                     Ymax + hstepY),
           vmin=vmin,vmax=vmax)
    

def saveImage(data,filename,cmap=None,vmin=None,vmax=None):
    
    if vmin==None:
        vmin=min(data)
    if vmax==None:
        vmax=max(data)
    if not vmax > vmin:
        vmax += 0.5
        vmin -= 0.5
    s = shape(data)
    if len(s) <= 2:
        if not isinstance(cmap,LinearSegmentedColormap):
            cmap = cm.get_cmap(cmap)
        data = cmap(1.0*(data-vmin)/(vmax-vmin))
    #be sure data is between 0 and 1
    tobig = (data >=1) 
    data = data * (data > 0) * (1-tobig) + tobig        
    img = zeros(s,dtype=uint32)
    data = (data*256).astype(uint32)
    data -= (data == 256)
    #& unit32(0xFF)
    for i in arange(4):
        img |= (data[:,:,i]) << (8*i)
    #Image.frombuffer('RGBA',(s[1],s[0]),img,'raw','RGBA',0,0).save(filename)
    
    
    
    
    
        
lowGamut = array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.15,0.25,0.6]])
        
    

    
def mycolorscheme(x,colormatrix=None):
    sp = 90.0/240
    co = 140.0/240
    ws = sp**2/log(2.)
    wc = (1-co)**2/log(2.)
    r = (2*exp(-(x+sp)**2/ws))*(x>0) + (2*exp(-(x+co)**2/wc)-1)*(x<-co) + \
        1 * (x<=0) * (x>=-co)
    g = 2 * exp(-0.25*x**2/ws) - 1 
    g = g * (g>0)
    b = 2 * exp(-(x-sp)**2/ws) * (x<0) + (2*exp(-(x-co)**2/wc)-1) * (x>co) + \
        1 * (x>=0) * (x<=co)
    if colormatrix is not None:
        r,g,b = dot(colormatrix,[r,g,b])
    return {'red':r,'green':g, 'blue':b}

def mycolormap(colors,colormatrix=None):
    n=256
  
    if colors == 'wbk':
        x=linspace(0,1,n)
    elif colors == 'kbw':
        x=linspace(1,0,n)
    elif colors == 'wrk':
        x=linspace(0,-1,n)
    elif colors == 'krw':
        x=linspace(-1,0,n)
    elif colors == 'rwb':
        x=linspace(-0.9,0.9,n)
    elif colors == 'krwbk':
        x=linspace(-1,1,n)
    elif colors == 'rkb':
        x=linspace(-0.3,-1.7,n)
        x += 2 * (x<-1)
    elif colors == 'bkr':
        x=linspace(-1.7,-0.3,n)
        x += 2 * (x<-1)
    elif colors == 'wrkbw':
        x=linspace(0,-2,n)
        x += 2 * (x<-1)
    else:
        raise Exception('unknown color sequence')
        
    c = mycolorscheme(x,colormatrix)
    x = linspace(0,1,n)
    segmentdata = {
        'red':[(e,c['red'][i],c['red'][i]) for i,e in enumerate(x)],
        'green':[(e,c['green'][i],c['green'][i]) for i,e in enumerate(x)],
        'blue':[(e,c['blue'][i],c['blue'][i]) for i,e in enumerate(x)]}
             
    return LinearSegmentedColormap(colors,segmentdata,N=n)

    
    
