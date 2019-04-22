#!/usr/bin/python

from mininet.topo import Topo
from mininet.net import Mininet
#from mininet.node import CPULimitedHost
#from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel

class KalmanSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    def build( self ):
        # Add hosts and switches
        leftHost = self.addHost( 'h1' )
        rightHost = self.addHost( 'h2' )
        leftHost2 = self.addHost( 'h3' )
        rightHost2 = self.addHost( 'h4' )
        leftSwitch = self.addSwitch( 's3' )
        rightSwitch = self.addSwitch( 's4' )

        # Add links
        self.addLink( leftHost, leftSwitch)
        self.addLink( leftHost2, leftSwitch )
        self.addLink( leftSwitch, rightSwitch )
        self.addLink( rightSwitch, rightHost )
        self.addLink( rightSwitch, rightHost2 )

def perfTest():
    "Create network and run simple performance test"
    topo = KalmanSwitchTopo()
    net = Mininet( topo=topo)
    net.start()
    # print "Dumping host connections"
    dumpNodeConnections( net.hosts )
    #print "Testing network connectivity"
    # net.pingAll()
    print "Testing bandwidth between h1 and h2"
    h1, h2 = net.get( 'h1', 'h2' )
    net.iperf( (h1, h2),seconds=5 )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    perfTest()
