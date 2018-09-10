import numpy
import pylab
import newtomoplot
import newmyplotutils

def plotProbas(probas):
    pylab.clf()
    n = numpy.shape(probas)[1]
    w = int(numpy.round(numpy.sqrt(n*1.5)))
    h = (n+w-1)/w
    for i in arange(n):
        pylab.subplot(h,w,i+1)
        pylab.title('n=%d' % (i+1))
        pylab.bar(arange(n+1)-0.4, probas[:,i])
        pylab.xlim(-0.5,n+0.5)
        pylab.ylim(-10,100)


def plotPn(ds, coherentVis, coherent, rabiVis, rabis=None, session=None, n=5,minContrast=2.0,chop=None):
    p0,p1 = getVis(ds,coherentVis,session)
    if isinstance(rabis, ndarray):
        data = rabis
    else:
        data = getDataset(ds, dataset=rabis, session=session)
    
    timestep = data[1,0]-data[0,0]
    

    rabifreq,maxima,amplscale,visibilities = \
        coherentanalysis(ds, dataset=coherent, session=session,
                         chopForVis=data[-1,0], minContrast=minContrast,
                         doPlot=False, p0=p0, p1=p1)
    if n>len(maxima):
        print 'Not enough Rabi frequencies found, using fitted vacuum Rabi frequency'
        maxima = rabifreq[0]

    data = data[:,1]
    if not chop is None:
        data = data[:int(round(chop/timestep))]

    p0,p1=getVis(ds,rabiVis,session)
    Pk = photonnumbers1(data, maxima, timestep, p0, p1, n=n,decayrate=2e-3,
                        visibilities=visibilities,normalize=True)
    bar(arange(n)-0.4, Pk)
    pylab.xlim(-0.5,n-0.5)
    pylab.ylim(-0.1,1)
    pylab.xlabel('photon number')
    pylab.ylabel('probability')



def plotDensity(rho,legend=None,scale=1.0,color=None, width=0.05, ax=None):
    s=numpy.shape(rho)
    if ax is None:
        ax = pylab.gca()
    ax.set_aspect(1.0)
    pos = ax.get_position()
    arrlabelsize = ax.xaxis.get_ticklabels()[0].get_fontsize()
    r = numpy.real(rho)
    i = numpy.imag(rho)
    x = numpy.arange(s[1])[None,:] + 0*r
    y = numpy.arange(s[0])[:,None] + 0*i
    pylab.quiver(x,y,r,i,units='x',scale=1.0/scale, width=width, color=color)
    pylab.xlabel('$n$')
    pylab.ylabel('$m$')
    if legend is not None:
        if numpy.shape(legend)== (2,2):
            pass
        elif numpy.shape(legend) == (2,):
            legend = [[legend[0],legend[0]], [legend[1],legend[1]+1]]
        else:
            legend = [[s[0]+0.5]*2,[-1,0]]
        x = pylab.quiver(legend[0], legend[1], [1,0],[0,1],
                         units='x',scale=1.0/scale, color=color,
                         width=width)
        x.set_clip_on(False)
        pylab.text(legend[0][0]+scale*0.5,
                   legend[1][0]+width*(-1.5+3.0*(legend[1][0]>0))*scale, r'$1$',
                   ha='center',va=['top','bottom'][int(legend[1][0]>0)],
                   fontsize=arrlabelsize)
        pylab.text(legend[0][1]+width*(-1.5+3.0*(legend[0][1]>0))*scale,
                   legend[1][1]+0.5*scale,r'$i$',
                   ha=['right','left'][int(legend[0][1]>0)],
                   va='center',fontsize=arrlabelsize)
        pylab.text(legend[0][0]+0.5*scale,legend[1][1]+scale+0.1,r'$\rho_{mn}$',va='bottom',ha='center')
    pylab.xlim(-0.9999,s[1]-0.0001)
    pylab.ylim(-0.9999,s[0]-0.0001)
    

def plotState(phi, action='', oldstate=None,filename=None):
    s=numpy.shape(phi)        


    assert len(s)==2
    assert s[0]==2
    rf = 4.0
    if action == 'swap':
        qf = 4.0
    else:
        qf = 3.0
    pylab.figure(figsize=(3,(rf*(s[1]-1)+4)/4.0))
    ax = pylab.axes((0,0,1,1,))
    qstates = ['g','e']
#    pylab.ioff()
    ax.set_frame_on(False)
    q=numpy.arange(s[0])
    r=numpy.arange(s[1])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    x=numpy.array([-1.0,1.0])
    pylab.plot(x[:,None]+0.0*r[None,:],0.0*x[:,None]+rf*r[None,:],'k-')
    pylab.plot(4+x[:,None]+0.0*r[None,:-1],qf+0.0*x[:,None]+rf*r[None,:-1],'k-')
        
    ql = q[:,None] + 0*r[None,:]
    rl = 0*q[:,None] + r[None,:]
    ql = numpy.reshape(ql, numpy.size(ql))
    rl = numpy.reshape(rl, numpy.size(rl))
    g = numpy.argwhere(ql+rl<s[1])[:,0]
    ql = ql[g]
    rl = rl[g]
    phi = numpy.reshape(phi, numpy.size(phi))[g]
    

    if oldstate is not None:
        print oldstate
        oldstate = numpy.reshape(oldstate, numpy.size(oldstate))[g]
        pylab.quiver(4*ql, qf*ql+rf*rl,
                     numpy.real(oldstate), numpy.imag(oldstate),
                     color=(0.5,0.5,1,1),units='x',scale=0.7,width=0.1)
        
    for i in r:
        pylab.text(-1.5,rf*i,r'$|'+str(i)+r'\sf g\rangle$', ha='right',va='center')
    for i in r[:-1]:
        pylab.text(5.5, rf*i+qf, r'$|'+str(i)+r'\sf e\rangle$', ha='left',va='center')
        
        
    if action=='swap':
        t = numpy.linspace(0,1,21)
        for i in r[1:]:
#            pylab.plot(4*t,rf*i+1*numpy.sin(t*numpy.pi),'-r')
#            pylab.plot(4*t,rf*i-1*numpy.sin(t*numpy.pi),'-r')
#            myplotutils.arrowhead(4,rf*i,4/numpy.pi,1,color='r')
#            myplotutils.arrowhead(0,rf*i,-4/numpy.pi,-1,color='r')
            pylab.plot([1,3],[rf*i]*2,'-r')
            newmyplotutils.arrowhead(3,rf*i,1,0,color='r')
            newmyplotutils.arrowhead(1,rf*i,-1,0,color='r')
            
            
    elif action == 'drive':
        t = numpy.linspace(0,1,200)
        u = 0.3*numpy.sin(t*20*numpy.pi)*numpy.exp(-20*(t-0.5)**2)
        for i in r[:-1]:
            pylab.plot(4*t,rf*i+qf*t+u,'-r')
            newmyplotutils.arrowhead(0,rf*i+u[0],-4,-qf,color='r')
            newmyplotutils.arrowhead(4,rf*i+qf+u[-1],4,qf,color='r')
    elif action == 'phase':
        circx=numpy.linspace(1,2*numpy.pi,100)
        circy=numpy.cos(circx)
        circx=-numpy.sin(circx)
        for x in r[:-1]:
            pylab.plot(4+1.3*circx,rf*x+qf+1.3*circy,'-r')
            newmyplotutils.arrowhead(4,rf*x+qf+1.3,-1,0,color='r')
    pylab.quiver(4*ql, qf*ql+rf*rl,
                     numpy.real(phi), numpy.imag(phi),
                     color=(0,0,1,1),units='x',scale=0.7,width=0.1)
    print numpy.shape(q[:,None] + 0*r[None,:])
    print numpy.shape(0*q[:,None] + r[None,:])
    print numpy.shape(phi)
    print numpy.shape(oldstate)
    pylab.xlim(-4,8)
    pylab.ylim(-2,rf*(s[1]-1)+2)
    if filename is not None:
        pylab.savefig(filename)

def printstate(state,action='',value=None,oldstate=None,filename=None):
    if action != '':
        print action,':',value
        return
    rtr = numpy.sum(state,axis=1)
    rho = numpy.conjugate(state)[:,:,None,None]*state[None,None,:,:]
    nr =numpy.arange(numpy.shape(state)[1])
    rhoq = numpy.sum(rho[:,nr,:,nr],axis=0)
    x = numpy.real(rhoq[0,1] + rhoq[1,0])
    y = numpy.imag(-rhoq[0,1]+rhoq[1,0])
    z = numpy.real(rhoq[0,0]-rhoq[1,1])
    r,t,p = newtomoplot.xyz2rtp(x,y,z)
    print '      |      |g>      |      |e> '
    print '------+---------------+--------------'
    for n in nr:
        print '%5s | % 6.3f%+ 6.3fi | % 6.3f%+ 6.3fi' % ('|%d>'%n,numpy.real(state[0,n]),numpy.imag(state[0,n]),numpy.real(state[1,n]),numpy.imag(state[1,n]))
    print   'trace over resonator: r = %g, theta = %g pi, phi = %g pi' % (r,t/numpy.pi,p/numpy.pi)
    print 
        

def printlatex(state, action='',value=None, oldstate=None, filename=None):
    if action != '':
        print action, ':', value
        return
    sg=''
    se=''
    for i,a in enumerate(state[0,:]):
        a = 0.001*numpy.round(numpy.real(a)*1000)+0.001j*numpy.round(numpy.imag(a)*1000)

        if numpy.imag(a) == 0:
            a = numpy.real(a)
        if a:
            sg += '+'+str(a) + r'\ket{' + str(i) + '}'
    if sg != '':
        if sg[0] == '+':
            sg = sg[1:]
        sg = '(' + sg + r')\ket{g}'
    for i,a in enumerate(state[1,:]):
        a = 0.001*numpy.round(numpy.real(a)*1000)+0.001j*numpy.round(numpy.imag(a)*1000)
        if numpy.imag(a) == 0:
            a = numpy.real(a)

        if a:
            se += '+'+str(a) + r'\ket{' + str(i) + '}'
    if se != '':
        if se[0] == '+':
            se = se[1:]
        se = '(' + se + r')\ket{e}'
    if sg != '' and se != '':
        print sg + '+' + se
    else:
        print sg + se
        
