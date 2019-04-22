#!/usr/bin/python

from mininet.topo import Topo
from mininet.net import Mininet
#from mininet.node import CPULimitedHost
#from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import OVSSwitch, Controller, RemoteController
from time import sleep

class KalmanSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    # This sets up the topology of the network
    def build( self ):
        # Add hosts
        leftEndHost1 = self.addHost( 'h1' )
        leftEndHost2 = self.addHost( 'h2' )

        # Add switches
        leftEndSwitch = self.addSwitch( 's1' )

        # Add links from host to switches
        self.addLink( leftEndHost1, leftEndSwitch )
        self.addLink( leftEndHost2, leftEndSwitch )

def perfTest():
    "Create network and run simple performance test"
    topo = KalmanSwitchTopo()
#    net = Mininet( topo=topo)
    net = Mininet( topo=topo, controller=None)
    net.addController( 'c0', controller=RemoteController, ip='127.0.0.1', port=6633 )

    net.start()
    print "Dumping host connections"
    dumpNodeConnections( net.hosts )
    #print "Testing network connectivity"
    #net.pingAll()
    print "Testing bandwidth between h1 and h2"
    h1, h2 = net.get( 'h1', 'h2' )
    net.iperf( (h1, h2),seconds=10,udpBw='10K',l4Type='UDP')
    sleep(10)  
    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    perfTest()
