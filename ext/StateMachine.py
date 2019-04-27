import numpy as np
from numpy.linalg import inv
import time
import random

class StateMachine(object):

  def __init__(self,num_switches=2,num_hosts=4):

    # The number of hosts and switches -- this needs to change with the example case
    # This is metadata
    self.num_switches = num_switches
    self.num_hosts = num_hosts
    self.updateThres = 5000.0
    self.BurstState = False
    self.BurstDuration = 40
    self.prev_rate = 0.0
    # Note:  The switch selection is hardcoded to use only half the total number of switches
    # TODO:  Make this more general
    if self.num_switches > 1:
      self.switch_limit = self.num_switches / 2
    self.stateCovarMult = 0.001
    self.measCovarMult = 0.1
    self.processNoiseMult = 0.001
    self.measNoiseCrossMult = 100.0
    self.measNoiseDiagMult = 0.1
    # for the process noise matrix (see below)
    # |d .l.|m .n.m .n.m .n.m..n.m .n. 
    # |.d...|.m....m....m....m....m...
    # |..d..|..m....m....m....m....m..
    # |.l.d.|.n.m..n.m..n.m..n.m..n.m.
    # |....d|....m....m....m....m....m
    # |-------------------------------
    # |m .n.|j.....k..................
    # |.m...|.j.......................
    # |..m..|.j......................
    # |.n.m.|...j.....................
    # |....m|....j....................
    # |m .n.|.....j...................
    # |.m...|......j..................
    # |..m..|..k....j.................
    # |.n.m.|........j................
    # |....m|..........j..............

    # upper left matrix size -> self.num_flow by self.num_flow
    # d = (1 - scale_down1) * flow_pnoise_mult
    # l = flow_pnoise_mult
    # upper right matrix size -> self.num_flows by self.num_meas
    # m = (1 - scale_down1) * self.cross_pnoise_mult
    # n = self.cross_pnoise_mult
    # lower left matrix size -> self.num_meas by num_flow
    # j = (1 - scale_down1) * self.cross_pnoise_mult
    # k = self.cross_pnoise_mult
    # upper left matrix size -> self.num_meas by self.num_meas
    # d = (1 - scale_down2) * self.meas_pnoise_mult
    # l = self.meas_pnoise_mult
    self.flow_pnoise_mult = 100
    self.cross_pnoise_mult = 0.1
    self.flow_meas_pnoise_mult = 2
    self.meas_pnoise_mult = 1.0
    self.pnoise_diag_scale_down1 = 0.995
    self.pnoise_diag_scale_down2 = 0.5
    # This truly initializes the routing matrix
    self.all_zeros = True

    self.gainScaleVal = 2.0

    # The time counter which should get updated each timestep
    # Just be sure to get current time before taking any time difference
    self.prev_time = time.time()
    self.curr_time = self.prev_time + 1
    self.init_time = self.prev_time
    
    # The state vector in the order of flows and then measured switch stats
    # This state vector counts from src = host 1 -> dst = host 2 <-> n then so forth
    self.num_flows = self.num_hosts * (self.num_hosts - 1)
    self.num_meas = self.num_switches * self.num_flows
    
    # The switch stats is the measured flow stats for each switch
    # Indexing this will be something to be cautious of
    # The size is the number of flows and the number of measurements which is a flow at each switch
    self.size_state_vec = self.num_flows * (self.num_switches + 1)

    # This below is the state vector for the system
    # I think this is enough initialization
    # Equation 5b
    # This is the s vector
    self.s = np.zeros((self.size_state_vec, 1))

    # This is the Q term or known as process noise
    # This is the Q matrix
    #self.Q = np.ones((self.size_state_vec,self.size_state_vec))
    #self.Q *= self.processNoiseMult
    self.InitQ()

    # This below is the state covariance matrix
    # I think this is sufficient for now -- can revisit later
    # This is the P matrix
    #self.P = np.ones((self.size_state_vec,self.size_state_vec))
    #self.P *= self.stateCovarMult
    # Initialize state covariance to process noise
    self.P = self.Q.copy()


    # This below is the state transition matrix
    # This allows the computation of the predicted measurements using the estimated flow rates
    # Need to build A array to reflect the routing of the network
    #self.F = np.zeros((self.size_state_vec, self.size_state_vec))
    # This is the F matrix -- Initialize to all zeros to begin with -- calculate when needed later
    self.F = np.zeros((self.size_state_vec, self.size_state_vec))
    # Also init routing matrix just in case
    self.route_matrix = self.InitRoutingMatrix(all_zeros=self.all_zeros)
    #print("The routing matrix is:")
    #print(self.route_matrix) 

    # This below is the Measurement Noise Covar matrix
    # Initializes the R matrix
    # |d ........| 
    # |.d........|
    # |..d...l...|
    # |...d......|
    # |....d.....| 
    # |.....d....|
    # |......d...|
    # |..l....d..|
    # |........d.|
    # |.........d|
    # matrix size -> self.num_meas by self.num_meas
    # d = self.measNoiseDiagMult
    # l = self.measNoiseCrossMult
    temp = np.ones((self.num_meas,self.num_meas))
    temp -= np.eye(self.num_meas,self.num_meas)
    self.R = np.eye(self.num_meas,self.num_meas) * self.measNoiseDiagMult + temp * self.measNoiseCrossMult

    # This below is the Measurement matrix
    #self.H = zeros((self.num_meas,self.size_state_vec))
    # Initializes the H matrix
    self.InitH()

    # This below is the Gain Matrix
    # This is the G matrix
    #self.G = np.zeros((self.size_state_vec,self.size_state_vec))
    self.G = np.zeros((self.size_state_vec,self.num_meas))
    
    # This is the measurement vector -- the measurements received from the query
    # This is the z vector
    self.z = np.zeros((self.num_meas,1))
    self.prev_z = np.zeros((self.num_meas,1))

    # Initialize this but don't likely need this since it will get updated the first iteration
    # This is the output of Equation 11
    self.s_pred = np.zeros((self.size_state_vec,1))

 # --------------------------------------------------------------------------------------------
 # --------------------------------------------------------------------------------------------
 # Start State initialization functions here --------------------------------------------------
 # --------------------------------------------------------------------------------------------
 # --------------------------------------------------------------------------------------------

  # This routine initializes the state transition function
  # This array is in the form of [[C 0],[A I]] where C => num_flow by num_flow
  # The upper right zero array should size num_flow by num_meas
  # Equation 5a
  def CalcF(self, time_diff):
     # initialize upper left to an eye matrix of size c
     Carray = np.eye(self.num_flows)
     ZeroArray = np.zeros((self.num_flows,self.num_meas))
     # TODO:  The A array needs to know the routing map of the network
     # This needs to work with the routing code so when a flow is first installed in a switch
     # to add a term to this array
     # Ideally, set this to all zeros initially and then get this from the switch object
     #print(time_diff)
     #print(self.route_matrix) 
     Aarray = self.route_matrix * time_diff
     IdenArray = np.eye(self.num_meas)
     comp1 = np.concatenate((Carray, ZeroArray), axis=1)
     comp2 = np.concatenate((Aarray, IdenArray), axis=1)
     self.F = np.concatenate((comp1,comp2),axis=0)

  # This routine needs to be able to receive a routing matrix since it needs to be defined
  # externally from the routing service
  # The route matrix is inserted into the transition matrix and the transition matrix is updated
  # Basically, it is the same as above with route matrix info
  def DefineRoutingMatrix(self, route_matrix):
     self.route_matrix = route_matrix

  # This routine modifies the route matrix based on flow detections at the switch
  # The purpose of the route matrix is to ensure that an flow rate prediction is applied to a flow stat
  # Note: The route matrix is the lower left matrix in the transition matrix
  def AddRouteForRouteMatrix(self, switchNo, HostSrc, HostDst):
     counter = 0
     for i in range(self.num_hosts):
       for j in range(self.num_hosts):
         if i != j:
           #print("The counter number is " + str(counter))
           if i == (HostSrc - 1) and j == (HostDst - 1):
             self.route_matrix[(switchNo - 1) * self.num_flows + counter, counter] = 1
             return
           counter += 1

  # This routine initializes the measurement matrix
  # Equation 7 -- I believe this is unchanging
  def InitH(self):
     # initialize upper left to an eye matrix of size c
     ZeroArray = np.zeros((self.num_meas,self.num_flows))
     IdenArray = np.eye(self.num_meas)
     self.H = np.concatenate((ZeroArray,IdenArray),axis=1)

  def CalculateHbyEntropy(self):
    # Create empty matrix
    Hstar = np.zeros((self.num_meas,self.num_meas + self.num_flows))
    # Limit the number of switches to include in the measurement matrix to self.switch_limit
    for i in range(self.switch_limit):
       #print("Adding switch number " + str(i))
       curr_h = self.EstimateEntropy(Hstar)
       Hhat = Hstar
       for j in range(self.num_switches):
         Htilde = self.AddSwitch2Measurement(Hstar.copy(), j)
         #print("Hstar is here")
         #print(Hstar)
         new_h = self.EstimateEntropy(Htilde)
         if new_h < curr_h:
           Hhat = Htilde.copy()
           curr_h = new_h
       Hstar = Hhat
    self.H = Hstar  
           
  # Compute the estimated process covariance using input Htilde and then estimate entropy (just determinant)
  def EstimateEntropy(self, Htilde):
    # calculate new gain here
    Gnew = self.CalculateG(Htilde)
    Pnew = self.CalculateP(Gnew, Htilde)
    # return the determinant of the matrix
    return np.linalg.det(Pnew)   

  def AddSwitch2Measurement(self, Hstar, switchNo):
    #print("The switch number = " + str(switchNo))
    #print(Hstar.shape)
    for i in range(self.num_flows):
      offset = i + (self.num_flows * switchNo)
      Hstar[offset][offset + self.num_flows] = 1.0 
    return Hstar

  def InitRoutingMatrix(self, all_zeros=False):
    if all_zeros:
      temp = np.zeros((self.num_meas,self.num_flows))
    else:
      flow_mat = np.eye(self.num_flows)
      temp = flow_mat
      for i in range(self.num_switches - 1):
        temp = np.concatenate((temp,flow_mat),axis=0)
    return temp
    
  def InitQ(self):
    # Init upper left corner -- num_flow x num_flow
    upper_left1 = np.eye(self.num_flows) * self.pnoise_diag_scale_down1
    upper_left2 = np.ones((self.num_flows, self.num_flows)) - upper_left1
    upper_left2 = upper_left2 * self.flow_pnoise_mult
    # Init upper right corner -- num_flow x num_meas  
    #upper_right = np.ones((self.num_flows, self.num_meas)) * self.flow_meas_pnoise_mult
    upper_right1 = np.eye(self.num_flows) * self.pnoise_diag_scale_down1
    upper_right2 = np.ones((self.num_flows, self.num_flows)) - upper_right1
    upper_right2 = upper_right2 * self.cross_pnoise_mult
    upper_right = upper_right2
    for i in range(self.num_switches - 1):
      upper_right = np.concatenate((upper_right, upper_right2), axis=1)
    lower_left = upper_right.T
    lower_right1 = np.eye(self.num_meas) * self.pnoise_diag_scale_down2
    lower_right2 = np.ones((self.num_meas, self.num_meas)) - lower_right1
    lower_right2 = lower_right2 * self.meas_pnoise_mult  
    comp1 = np.concatenate((upper_left2, upper_right), axis=1)
    comp2 = np.concatenate((lower_left, lower_right2), axis=1)
    self.Q = np.concatenate((comp1,comp2),axis=0)


 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # End State initialization functions here --------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------

 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # Start State equations here ---------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------

  # Equation 11 -- Predict State Vector
  def PredictState(self):
    time_diff = self.curr_time - self.prev_time
    # Calculate the state trans matrix accounting for time difference here
    self.CalcF(time_diff)
    # Calculate the predicted state
    self.s_pred = np.dot(self.F,self.s)

  # Equation 12 -- Update State Covariance
  # 4/20/19 -- added scaling for time delta to process noise
  def PredictCovar(self):
    time_scale = self.curr_time - self.prev_time
    #F = self.F * time_scale
    temp = np.dot(self.P,self.F.T)
    temp2 = np.dot(self.F,temp)
    self.P = temp2 + self.Q * time_scale
    
  # Equation 13 -- Update G
  def UpdateG(self):
     # This is also the numerator
     temp = np.dot(self.P, self.H.T)
     numer = temp
     # This calculates the denominator
     temp1 = np.dot(self.H, temp)
     #self.R = np.zeros((self.num_meas,self.num_meas))
     temp2 = self.R + temp1
     denom = inv(temp2)
     self.G = np.dot(numer,denom)
     # TEMP:  This scales the gains for debugging
     #self.GainScale()
     # TODO:  Loop over all gain terms and limit them between 0 and 1
     #self.ClampGains()
     #self.ZeroRateGains()

  def ZeroRateGains(self):
    for i in range(self.num_flows):
      for j in range(self.num_meas):
        self.G[i,j] = 0.0

  def GainScale(self):
  # This function scales all gains by the argument value
     self.G = self.G * self.gainScaleVal 

  # This function fixes the gain terms between 0.0 and 1.0
  def ClampGains(self):
     for i in range(self.size_state_vec):
       for j in range(self.num_meas):
         if self.G[i,j] > 1.0:
           self.G[i,j] = 1.0
         if self.G[i,j] < 0.0:
           self.G[i,j] = 0.0

  # The idea of this is the gain terms that feed the rates terms need compensated for time
  # The size of the gains matrix is (num_flows + num_meas) by (num_meas)
  def CompensateGain(self):
     topArray = np.ones((self.num_flows,self.num_meas)) / (self.curr_time - self.prev_time)
     #topArray = self.gainScaleVal * np.ones((self.num_flows,self.num_meas)) / (self.curr_time - self.prev_time)
     bottomArray = np.ones((self.num_meas,self.num_meas))
     fullArray = np.concatenate((topArray,bottomArray),axis=0)
     self.G *= fullArray

  # Equation 14 -- Update State Vector
  def Updatestor(self):
     temp = np.dot(self.H, self.s_pred)
     self.residual = self.z - temp
     temp3 = np.dot(self.G, self.residual)
     self.s = self.s_pred + temp3
     self.prev_z = self.z.copy()

  def Updatestor2(self):
     for i in range(self.num_meas):
       self.s[self.num_flows + i] = self.z[i]
     # Estimate rates here   
     self.UpdateRate()
     #print('The state vector is here')
     #print(self.s)
     self.prev_z = self.z.copy()

  # This routine dithers between Updatestor and Updatestor2 based on sense of burstiness
  def Updatestor3(self):
     time_diff = self.curr_time - self.prev_time
     estRate = (self.z - self.prev_z)//time_diff
     change = estRate - self.prev_rate
     print("change vector =")
     print(change)
     print(self.z)
     print(self.prev_z)
     changeLevel = sum(abs(change))
     print("The changeLevel is " + str(changeLevel))
     if changeLevel > self.updateThres or self.BurstState:
       print("TRIGGERED BURST DETECTION!!")
       if not self.BurstState:
         self.BurstIter = self.BurstDuration
         self.BurstState = True
       self.Updatestor2()
       self.BurstIter = self.BurstIter - 1
       if self.BurstIter == 0:
         self.BurstState = False
     else:
       self.Updatestor()
     self.prev_rate = estRate

  # Estimation for Rate Value
  def UpdateRate(self):
    time_diff = self.curr_time - self.prev_time
    rate = np.zeros((self.num_flows,1))
    for i in range(self.num_flows):
      for j in range(self.num_switches):
        diff = self.z[i + j*self.num_flows] - self.prev_z[i + j*self.num_flows]
        if diff > rate[i]:
          rate[i] = diff
      self.s[i] = rate[i] / time_diff
    

  # Equation 15 -- Update State Covariance
  def UpdateP(self):  
     Ivec = np.eye(self.size_state_vec,self.size_state_vec)
     temp = np.dot(self.G, self.H)
     temp2 = Ivec - temp
     self.P = np.dot(temp2, self.P)

  # Equation 13 (modified) -- This returns new gain based on input measurement matrix
  def CalculateG(self, Htilde):
     # This is also the numerator
     temp = np.dot(self.P, Htilde.T)
     numer = temp
     # This calculates the denominator
     temp1 = np.dot(Htilde, temp)
     temp2 = self.R + temp1
     denom = inv(temp2)
     return np.dot(numer,denom)

  # Equation 15 (modified)-- Returns State Covariance based on input measurement matrix and Gain
  def CalculateP(self, Gmod, Htilde):  
     Ivec = np.eye(self.size_state_vec,self.size_state_vec)
     temp = np.dot(Gmod, Htilde)
     temp2 = Ivec - temp
     return np.dot(temp2, self.P)

 # -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------
 # End State equations here ----------------------------------------------------
 # -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------

 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # Start Main Methods here ---------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
 # ------------------------------------------------------------------------------------------
  
  # This routine updates the time based on "clock" time
  def UpdateTime(self):
    self.prev_time = self.curr_time
    self.curr_time = time.time()

  # This routine updates the time based on sim time
  def FixTime(self, man_time):
    self.prev_time = self.curr_time
    self.curr_time = man_time

  # This routine init the time based on sim time
  def InitFixTime(self, man_time):
    self.prev_time = man_time
    self.curr_time = man_time

  # This routine receives the all measurements from each switch to update the measurement vector
  # The ordering of the measured Values will be a list from 1 -> 2,n, 2 -> 1,n ... n-1 -> 1,n
  def UpdateMeasurementBySwitch(self, measVals, switchNo = 1, numHosts = 8):
    # Calculate starting position of measured values vector
    startIndex = (switchNo - 1) * self.num_flows
    # Loop over measVals and insert into measured values
    for i in range(self.num_flows):
      #print("The index is " + str(i) + " and the startIndex = " + str(startIndex))
      self.z[i + startIndex,0] = measVals[i,0] + self.z[i + startIndex,0]

  def Update(self, set_time=False, fixed_time=0.0):
    # first update the time for the correct time difference amount
    if set_time:
      self.FixTime(fixed_time)
    else:
      self.UpdateTime()

    # Calculate Eq 11
    self.PredictState()

    # Calculate Eq 12
    self.PredictCovar()

    # Calculate Eq 13
    self.UpdateG()

    # Hey, I think this made this much better!!  Not good but better.
    self.CompensateGain()

    # Calculate Eq 14
    self.Updatestor()

    # Calculate Eq 15
    self.UpdateP()

    # Clear the measurement vector to start measurement update from scratch
    #self.z = np.zeros((self.num_meas,1))

  # This outputs the state vector to a file output for later post-processing
  # TODO:  Finish this method up
  def OutputStateResult(self, file):
    pass

  # This outputs the state vector to a file output for later post-processing
  def ReturnStateResult(self):
    return self.s

  # This outputs the predicted state vector
  def ReturnStatePredict(self):
    return self.s_pred

  # This outputs the state covar vector to a file output for later post-processing
  def ReturnPResult(self):
    return self.P

  # This outputs the state covar vector to a file output for later post-processing
  def ReturnGainResult(self):
    return self.G

  # This outputs the state covar vector to a file output for later post-processing
  def ReturnH(self):
    return self.H

  # This outputs the Measurement Noise Matrix
  def ReturnMeasNoise(self):
    return self.R

  # This outputs the Measurement Noise Matrix
  def ReturnQ(self):
    return self.Q

 # -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------
 # End Main Methods here ----------------------------------------------------
 # -----------------------------------------------------------------------------
 # ----------------------------------------------------------------------------- 

 # -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------
 # helper functions below ---------------------------------------------
 # -----------------------------------------------------------------------------
 # -----------------------------------------------------------------------------

  # This computes the index into the measurement vector
  def ComputeMeasIndex(self,flow_num, switch_num):
    return flow_num + switch_num * self.num_flows 

  # This updates the measurement vector for a single flow for a single switch
  # 4/20/19 -- Changing measurement from time step change to flow statistic  
  def UpdateMeasurement(self,flow_num, switch_num, measurement):
#    if switch_num < 1:
#      print "invalid switch num %i " %switch_num
#    print("The flow num = " + str(flow_num) + " and switch num = " + str(switch_num))
    # TODO:  Make sure that this indexing is correct
    self.z[(switch_num - 1) * self.num_flows + flow_num - 1] = measurement + self.z[(switch_num - 1) * self.num_flows + flow_num - 1]

  def ComputeFlowNum(self,host_src, host_dst):
    temp = 1
    count = self.num_hosts + 1
    for i in range(1,count):
        for j in range(1,count):
            if i == j:
                pass
            elif i == host_src and j == host_dst:
                return temp
            else:
                temp += 1

# This is the case of a simple one switch and two cost topology
# The routing matrix should be a simple 2x2 identity matrix
def CalculateSimpleTopo():
  return np.eye(2)
  

def TwoSwitchFourHost(s):
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=3, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=1, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=3, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=4)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=4, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=2, HostDst=4)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=4, HostDst=2)

def OneSwitchTwoHost(s):
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=1)

def TwoSwitchThreeHost(s):
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=3, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=3, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=1, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=3, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=2, HostDst=3)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=3, HostDst=2)

def TwoSwitchTwoHost(s):
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=1, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=1, HostSrc=2, HostDst=1)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=1, HostDst=2)
    s.AddRouteForRouteMatrix(switchNo=2, HostSrc=2, HostDst=1)

def GetMeasurementsOneSwitchTwoHost(iteration, type):
   if type == 1:
     if iteration == 0:
       return np.array([[3.6],[0]])
     else:
       return np.array([[0.0],[0]])
   elif type == 2:
       return np.array([[2.5],[0]])
   # Generate ramps here
   elif type == 3:
     if iteration > 0 and iteration < 10:
        return np.array([[0.0],[0]])   
     elif iteration >= 10 and iteration < 15:
        value = ((iteration - 10.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 15 and iteration < 30:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 30 and iteration < 35:
        value = 10000.0 - ((iteration - 30.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])
     elif iteration >= 35 and iteration < 50:
       return np.array([[0.0],[0]])
     elif iteration >= 50 and iteration < 55:
        value = ((iteration - 50.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 55 and iteration < 70:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 70 and iteration < 75:
        value = 10000.0 - ((iteration - 70.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])
     else:
        return np.array([[0.0],[0]])           
   # Generate bursts here
   elif type == 4:
     if iteration > 0 and iteration < 10:
        return np.array([[0.0],[0]])   
     elif iteration >= 10 and iteration < 12:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 12 and iteration < 20:
        return np.array([[0.0],[0]])  
     elif iteration >= 20 and iteration < 22:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 22 and iteration < 30:
       return np.array([[0.0],[0]])
     elif iteration >= 30 and iteration < 32:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])    
     elif iteration >= 32 and iteration < 40:
       return np.array([[0.0],[0]])  
     elif iteration >= 40 and iteration < 42:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])    
     elif iteration >= 42 and iteration < 50:
       return np.array([[0.0],[0]])  
     else:
        return np.array([[0.0],[0]]) 
   # Generate mixed inputs here
   elif type == 5:
     if iteration > 0 and iteration < 10:
        return np.array([[0.0],[0]])   
     elif iteration >= 10 and iteration < 15:
        value = ((iteration - 10.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 15 and iteration < 30:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 30 and iteration < 35:
        value = 10000.0 - ((iteration - 30.0)/5.0) * 10000.0 + 100 * random.random()
        return np.array([[value],[0]])
     elif iteration >= 35 and iteration < 50:
       return np.array([[0.0],[0]])
     elif iteration >= 50 and iteration < 52:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])   
     elif iteration >= 52 and iteration < 60:
       return np.array([[0.0],[0]])
     elif iteration >= 60 and iteration < 62:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])    
     elif iteration >= 62 and iteration < 70:
       return np.array([[0.0],[0]])  
     elif iteration >= 70 and iteration < 72:
        value = 10000.0 + 100 * random.random()
        return np.array([[value],[0]])    
     elif iteration >= 72 and iteration < 80:
       return np.array([[0.0],[0]])  
     else:
        return np.array([[0.0],[0]]) 

# This writes the F matrix to file
def writeFMatrix(s,f):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('----------- Start F matrix here -----------------------\n')
  # Loop over F matrix 
  for i in range(s.size_state_vec):
    for j in range(s.size_state_vec):
      f.write(str(s.F[i,j]) + ' ')
    f.write('\n')
  f.write('----------- End F matrix here -----------------------\n')

# This writes the H matrix to file
def writeHMatrix(s,h,f):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('----------- Start H matrix here -----------------------\n')
  # Loop over H matrix 
  for i in range(s.num_meas):
    for j in range(s.size_state_vec):
      f.write(str(h[i,j]) + ' ')
    f.write('\n')
  f.write('----------- End H matrix here -----------------------\n')


def writeCovarMatrix(s,covarMat,f,title):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('----------- Start ' + title + ' Covar matrix here -----------------------\n')
  # Loop over covar matrix 
  for i in range(s.size_state_vec):
    for j in range(s.size_state_vec):
#      f.write(str(covarMat[i,j]) + '\t')
      f.write("%.2f" %covarMat[i,j] + '\t')
    f.write('\n')
  f.write('----------- End ' + title + ' Covar matrix here -----------------------\n')

def writeGainMatrix(s,gain,f):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('----------- Start Gain matrix here -----------------------\n')
  # Loop over covar matrix 
  for i in range(s.size_state_vec):
    for j in range(s.num_meas):
      f.write("%.2f" %gain[i,j] + '\t')
    f.write('\n')
  f.write('----------- End Gain matrix here -----------------------\n')

def writeStateVector(s,predSV,updSV,f):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('------- Predict left ----- Update Right ----------------\n')
  # Loop over SVs 
  for i in range(s.size_state_vec):
    f.write('\t' + "%.2f" %predSV[i,0] + '\t\t\t' + "%.2f" %updSV[i,0])   
    f.write('\n')
  f.write('---------------- End SVs here -----------------------\n')

def writeResidualVector(s,f):
  f.write('Timestep = ' + str(s.curr_time) + '\n')
  f.write('------- Residual Vector ----------------\n')
  # Loop over vector 
  for i in range(s.num_meas):
    f.write('\t' + "%.2f" %s.residual[i,0])   
    f.write('\n')
  f.write('---------------- End Residual here -----------------------\n')

def writeStateResult(f,sv,time, comps):
  f.write('\t' + "%.2f" %time)
  for i in comps:
    f.write('\t' + "%.2f" %sv[i])
  f.write('\n')

# The purpose of this is to debug the Kalman filtering
# TODO:  Place a loop in here and store all vectors and matrices in separate files 
if __name__ == '__main__':
  # This initializes the simpliest meaningful case
  noSwitches = 1
  noHosts = 2 
  noFlows = noHosts * (noHosts - 1)
  noMeas = noFlows * noSwitches
  initTime = 0.0
  nextTime = 1.0
  timeStep = 1.0
  numLoops = 80
  updateMode = 3  # 1 = Updatestor, 2 = Updatestor2 and 3 = Updatestor3
  useEntropy = False # Determines whether to select partial switches or not
  inputType = 5 # 1 = step, 2 = ramp, 3 = rounded boxcars, 4 = bursts, 5 = mixed boxcar and bursts
  s = StateMachine(num_switches=noSwitches,num_hosts=noHosts)
  print("State machine created with " + str(noSwitches) + " switches and " + str(noHosts) + " hosts")
  s.InitFixTime(initTime)
  print("Time initialized")

  # Open file descriptors here
  fMatrix=open('/home/jeffrey/minitest/internal/F_matrix', 'w')
  sVector=open('/home/jeffrey/minitest/internal/s_vector', 'w')
  predCovarMatrix=open('/home/jeffrey/minitest/internal/pred_covar_matrix', 'w')
  updateCovarMatrix=open('/home/jeffrey/minitest/internal/upd_covar_matrix', 'w')
  hMatrix=open('/home/jeffrey/minitest/internal/h_matrix', 'w')
  gMatrix=open('/home/jeffrey/minitest/internal/g_matrix', 'w')
  resVector=open('/home/jeffrey/minitest/internal/r_vector', 'w')
  sResult=open('/home/jeffrey/minitest/internal/s_result', 'w')


  # Check Transition Matrix Definition
  #s.DefineRoutingMatrix(CalculateSimpleTopo())
  # Check for adding states to routing matrix properly
  # Note:  This is specific to the topology and needs to be tweaked based on example
  if noSwitches == 2 and noHosts == 4:
    TwoSwitchFourHost(s)
  elif noSwitches == 1 and noHosts == 2:
    OneSwitchTwoHost(s)
  elif noSwitches == 2 and noHosts == 3:
    TwoSwitchThreeHost(s)
  elif noSwitches == 2 and noHosts == 2:  
    TwoSwitchTwoHost(s)
    
  if noSwitches == 1 and noHosts == 2:
    s.s = np.array([[1],[0],[0.5],[0]])
    print('Initialize the state vector!!')

  for iteration in range(numLoops):
    # This is where the loop should start
    # Set current time
    s.FixTime(nextTime)
    # Increment nextTime here in case looping should occur
    nextTime = nextTime + timeStep
    print("The time diff is " + str(s.curr_time - s.prev_time))

    # Step 1:  Calculate State Trans Matrix
    s.CalcF(s.curr_time - s.prev_time)
    writeFMatrix(s,fMatrix)


    # Step 2:  Update measurement code here to handle varying rate input  
    print("Update the measurements here")
    if noSwitches == 1 and noHosts == 2:
      measurements = GetMeasurementsOneSwitchTwoHost(iteration,inputType)
      print('The measured flow stats are')
      print(measurements)
      s.UpdateMeasurementBySwitch(measurements, switchNo=1, numHosts=noHosts)
    elif noSwitches == 2 and noHosts == 3:
      measurements1 = np.array([[1],[0],[1],[0],[1],[0]])
      measurements2 = np.array([[0],[0],[0],[0],[1],[0]])
      s.UpdateMeasurementBySwitch(measurements1, switchNo=1, numHosts=noHosts)
      s.UpdateMeasurementBySwitch(measurements2, switchNo=2, numHosts=noHosts)
    if noSwitches == 2 and noHosts == 2:
      measurements = GetMeasurementsOneSwitchTwoHost(iteration,inputType)
      print('The measured flow stats are')
      print(measurements)
      s.UpdateMeasurementBySwitch(measurements, switchNo=1, numHosts=noHosts)
      s.UpdateMeasurementBySwitch(measurements, switchNo=2, numHosts=noHosts)
    else:
      measurements = np.zeros((noMeas,1))
      s.UpdateMeasurementBySwitch(measurements, switchNo=noSwitches, numHosts=noHosts)

    # Step 1:  State vector prediction
    s.PredictState()
    predictSV = s.s_pred

    # Step 2:  State covariance prediction
    s.PredictCovar()

    # The Process Noise matrix
    #print("The process noise matrix")
    #print(s.ReturnQ())

    # The Measurement Noise matrix
    #print("The measurement noise matrix")
    #print(s.ReturnMeasNoise())

    # Step 3:  Write the Measurement Matrix
    h = s.ReturnH()
    writeHMatrix(s,h,hMatrix)

    # Step 4:  Calculate and Write the gain matrix
    s.UpdateG()
    g = s.ReturnGainResult()
    writeGainMatrix(s,g,gMatrix)

    # Check state vector update
    # TODO:  Save the predicted state vector, compute update and store both to file
    if updateMode == 1:
      s.Updatestor()
      # write residual vector to file
      writeResidualVector(s,resVector)
    elif updateMode == 2:
      s.Updatestor2()
    else:
      s.Updatestor3()    
    updateSV = s.ReturnStateResult()
    writeStateVector(s,predictSV,updateSV,sVector)
    # output state vector result here
    writeStateResult(sResult,updateSV,s.curr_time,[0])
    #print(s.ReturnStateResult())



    # Check state covariance update
    #print("The State Covar before update")
    #print(s.ReturnPResult())
    #print("Update State Covar here")
    # TODO:  Output both predicted and update state covar to file
    writeCovarMatrix(s,s.ReturnPResult(),predCovarMatrix,'Predicted')
    s.UpdateP()
    writeCovarMatrix(s,s.ReturnPResult(),updateCovarMatrix,'Updated')
    #print(s.ReturnPResult()) 

    if useEntropy:
      s.CalculateHbyEntropy()
    #print("The New Measurement Matrix")
    #print(s.ReturnH())
  
  # close files
  fMatrix.close()
  sVector.close()
  predCovarMatrix.close()
  updateCovarMatrix.close()
  hMatrix.close()
  gMatrix.close()   
  resVector.close()
  sResult.close()
