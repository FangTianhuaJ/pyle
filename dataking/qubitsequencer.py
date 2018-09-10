from pyle.util.labradtools import ContextCycler


class QubitSequencer(ContextCycler):
    """Create a context cycler for sending pipelined requests to the Qubit Sequencer."""
    def __init__(self, *a, **kw):
        ContextCycler.__init__(self, 'Qubit Sequencer', *a, **kw)


def prettyDump(pkt):
    """Pretty-print packets sent from the Qubit Sequencer to the GHz DACs.
    
    You can get access to the packet sent from the Qubit Sequencer to the
    GHz DACs server by calling the 
    """
    # TODO move this sort of pretty-printing into the qubit sequencer itself
    devs = []
    mems = {}
    srams = {}
    cmds = []
    for cmd, data in pkt:
        if cmd == 'Select Device':
            if len(data.split('ADC'))>1:
                pass
            else:
                dev = data
                devs.append(dev)
        elif cmd == 'Memory':
            mems[dev] = data
        elif cmd == 'SRAM':
            srams[dev] = data
        elif cmd == 'SRAM Address':
            pass # ignore these
        else:
            cmds.append((cmd, data))
    devs.sort()
    
    lines = []
    lines.append(', '.join(devs))
    lines.append('')
        
    lines.append('Memory')
    #Old style pretty printing
#    for row in zip(*[mems[dev] for dev in devs]):
#        lines.append('  '.join('%06X' % c for c in row))
#    lines.append('')
    #New style pretty printing, 2012 August. Daniel Sank
    for dev in devs:
        lines.append(dev)
        lines.append(str(['%06X' %cmd for cmd in mems[dev]]))
            
    lines.append('SRAM')
    for row in zip(*[srams[dev] for dev in devs]):
        lines.append('  '.join('%08X' % c for c in row))
    lines.append('')
            
    for cmd, data in cmds:
        lines.append(cmd)
        lines.append(str(data))
        lines.append('')
    
    return '\n'.join(lines)

