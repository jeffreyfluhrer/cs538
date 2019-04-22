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
from Test1_Plot import *

class KalmanSwitchTopo( Topo ):
    "Single switch connected to n hosts."
    # This sets up the topology of the network
    def build( self ):
        # Add hosts
        leftEndHost1 = self.addHost( 'h1' )
        leftEndHost2 = self.addHost( 'h2' )
        leftMidHost1 = self.addHost( 'h3' )
        leftMidHost2 = self.addHost( 'h4' )
        rightMidHost1 = self.addHost( 'h5' )
        rightMidHost2 = self.addHost( 'h6' )
        rightEndHost1 = self.addHost( 'h7' )
        rightEndHost2 = self.addHost( 'h8' )

        # Add switches
        leftEndSwitch = self.addSwitch( 's1' )
        leftMidSwitch = self.addSwitch( 's2' )
        rightMidSwitch = self.addSwitch( 's3' )
        rightEndSwitch = self.addSwitch( 's4' )

        # Add links from host to switches
        self.addLink( leftEndHost1, leftEndSwitch )
        self.addLink( leftEndHost2, leftEndSwitch )
        self.addLink( leftMidHost1, leftMidSwitch )
        self.addLink( leftMidHost2, leftMidSwitch )
        self.addLink( rightMidHost1, rightMidSwitch )
        self.addLink( rightMidHost2, rightMidSwitch )
        self.addLink( rightEndHost1, rightEndSwitch )
        self.addLink( rightEndHost2, rightEndSwitch )

        # Add links from switch to switch
        self.addLink( leftEndSwitch, leftMidSwitch )
        self.addLink( rightMidSwitch, leftMidSwitch )
        self.addLink( rightMidSwitch, rightEndSwitch )

def perfTest():
    "Create network and run simple performance test"
    topo = KalmanSwitchTopo()
#    net = Mininet( topo=topo)
    net = Mininet( topo=topo, controller=None)
    net.addController( 'c0', controller=RemoteController, ip='127.0.0.1', port=6633 )

    net.start()
    sleep(10)
    print "Testing bandwidth between h1 and h2"
    h1, h2, h3, h5, h6, h7 = net.get( 'h1','h2','h3','h5','h6','h7' )
    h1.sendCmd('iperf -s -u')
    h2.sendCmd('iperf -c 10.0.0.1 -u -b 10K')
    sleep(12)
    print "Testing bandwidth between h3 and h6"
    h3.sendCmd('iperf -s -u')
    h6.sendCmd('iperf -c 10.0.0.3 -u -b 10K')
    sleep(12)
    print "Testing bandwidth between h5 and h7"
    h5.sendCmd('iperf -s -u')
    h7.sendCmd('iperf -c 10.0.0.5 -u -b 10K')
    sleep(10)
    plotTest1()    
    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    perfTest()
