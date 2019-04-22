#!/usr/bin/python

from pox.core import core
from pox.lib.util import dpidToStr
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr, EthAddr
from StateMachine import StateMachine
from pox.openflow.of_json import *
import numpy as np
import StateMachine as st

# I think it's normal to leave the logger global
log = core.getLogger()
row = [0, 0, 0, 0, 0, 0, 0, 0]


class Monitor (object): 

  def __init__ (self, num_switches=4, num_hosts = 8, output="/home/jeffrey/minitest/output",flows=[],ss=True):
    # attach handlers to listeners
    core.openflow.addListenerByName("FlowStatsReceived", self.handle_flowstats_received)
    # JRF:  Not dealing with ports right now
    #core.openflow.addListenerByName("PortStatsReceived",  self.handle_portstats_received)
    self.web_bytes = 0
    self.web_flows = 0
    self.num_switches = num_switches
    self.num_hosts = num_hosts
    self.num_flows = num_hosts * (num_hosts - 1)
    self.TM = np.array([row, row, row, row, row, row, row, row])
    self.f = open(output, 'w')
    self.filter = st.StateMachine(num_switches=num_switches, num_hosts=num_hosts)
    self.measurement_counter = 0
    self.filter.DefineRoutingMatrix(self.InitRoutingMatrix())
    self.flows = flows
    self.ss = ss
    if self.ss:
      print("Performing Switch Selection")
    else:
      print("Not performing switch selection")
    #print("The flows output is " + str(flows))
    print("The number of hosts is " + str(self.num_hosts) + " and number of switches is " + str(self.num_switches))
    

  def InitRoutingMatrix(self):
    flow_mat = np.eye(self.num_flows)
    temp = flow_mat
    for i in range(self.num_switches - 1):
      temp = np.concatenate((temp,flow_mat),axis=0)
    return temp

  # handler to display flow statistics received in JSON format
  # structure of event.stats is defined by ofp_flow_stats()
  def handle_flowstats_received (self,event):
    stats = flow_stats_to_list(event.stats)
    log.debug("FlowStatsReceived from %s: %s",dpidToStr(event.connection.dpid), stats)
    # Turn off write to file now 
    #switch = self.write_TM(event)
    # Turn on write to State Machine
    switch = self.issue_switch_meas(event) 
    # Currently, and update measured value each flow stat
    # and then count the number of switches and once I get to the total, perform update
    self.measurement_counter += 1
    # if all switches have been heard from, perform update and print state vector
    if self.measurement_counter == self.num_hosts:
      #print "Performing update now"
      saveMeasVec = self.filter.z
      #print(saveMeasVec)
      # Perform Switch Selection Here ... I guess
      if self.ss == True:
        self.filter.CalculateHbyEntropy() 
      # Here is where the state is updated
      self.filter.Update()
      #print "Here is the new state"
      #print self.filter.ReturnStateResult()
      # Write to output filter here ----------------------------------
      self.f.write(str(self.filter.curr_time - self.filter.init_time) + '\t')
      for i in range(self.num_flows):
        if i in self.flows:
          # write the flow estimate if this is a selected flow
          self.f.write(str(self.filter.s[i,0]) + '\t')
          # write all switch measurements (Note:  Plotted in rate form)
          for j in range(self.num_switches):
            index = i + j * self.num_flows
            #self.f.write(str(self.filter.MeasVector[index,0]) + '\t')
            #print("index = " + str(index) + " gives " + str(saveMeasVec[index,0]))
            if abs(self.filter.curr_time - self.filter.prev_time) > 0.01:
            	self.f.write(str(saveMeasVec[index,0] / (self.filter.curr_time - self.filter.prev_time)) + '\t')
      self.f.write('\n')
      self.f.flush()
      # Finished output section here ---------------------------------
      self.measurement_counter = 0
  
  def issue_switch_meas(self,event):
    # TODO:  Maybe consider entire vector to the StateMachine object?
    # Extract the switch id from the event info
    switchId = dpidToStr(event.connection.dpid)
    switchNo = int(switchId[-1])
    # iterate over the event info and generate the flow vector
    # initialize the measurement vector
    #measurement = np.zeros((self.num_flows,1))
    # loop over the flow numbers and write the value to the measurement vector
    for f in event.stats:
      for i in range(self.num_hosts+1):
        addressFrom = "10.0.0." + str(i)
        #print(addressFrom)
        for j in range(self.num_hosts+1):
          addressTo = "10.0.0." + str(j)
          #  Should I check for looping back?
          if f.match.nw_dst == IPAddr(addressFrom) and f.match.nw_src == IPAddr(addressTo):
            if i != j:
              #print 'The index = ' + str(self.filter.ComputeFlowNum(i, j))
              #print 'The switch number = ' + str(switchNo)
              self.filter.UpdateMeasurement(self.filter.ComputeFlowNum(i, j), switchNo, f.byte_count)
              #measurement[self.filter.ComputeFlowNum(i, j),1] = f.byte_count
    return switchNo 


  def write_TM(self,event):
    switchId = dpidToStr(event.connection.dpid)
    switchNo = int(switchId[-1])
    for f in event.stats:
      if switchNo == 1:
        for i in range(8):
          addressFrom = "10.0.0." + str(i)
          for j in range(8):
            addressTo = "10.0.0." + str(j)
            if f.match.nw_dst == IPAddr(addressFrom) and f.match.nw_src == IPAddr(addressTo):
	      self.TM[i][j] += f.byte_count
        print self.TM
        self.f.write(str(self.TM))
    return switchNo   

  # handler to display port statistics received in JSON format
  def handle_portstats_received (self,event):
    stats = flow_stats_to_list(event.stats)
    log.debug("PortStatsReceived from %s: %s", dpidToStr(event.connection.dpid), stats) 

# handler for timer function that sends the requests to all the
# switches connected to the controller.
def _timer_func ():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
# JRF:  Not requesting port stats now    
#    connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))
    
# main function to launch the module
def launch ():
  from pox.lib.recoco import Timer
  
  core.registerNew(Monitor)

  # timer set to execute every five seconds
  Timer(5, _timer_func, recurring=True)

