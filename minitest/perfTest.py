from mininet.topo import Topo
from mininet.net import Mininet
#from mininet.node import CPULimitedHost
#from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI

def perfTest():
    print "Testing bandwidth between h1 and h2"
    h1, h2 = net.get( 'h1', 'h2' )
    net.iperf( (h1, h2),seconds=5 )
CLI.do_perfTest = perfTest

