from mininet.topo import Topo
from mininet.net import Mininet
#from mininet.node import CPULimitedHost
#from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI

def mycmd( self, line ):
    print "Testing bandwidth between h1 and h2"
    h1, h2 = self.net.get( 'h1', 'h2' )
    self.net.iperf( (h1, h2),seconds=5 )
#    "mycmd is an example command to extend the Mininet CLI"
#    net = self.mn
#    print( 'mycmd invoked for', net, 'with line', line, '\n'  )
CLI.do_mycmd = mycmd
