import matplotlib.pyplot as plt

def plotTest1():
  # Create empty lists for time and first data column
  time = []
  flow1_rate = []
  flow1_meas1 = []
  flow1_meas2 = []
  flow1_meas3 = []
  flow1_meas4 = []
  flow2_rate = []
  flow2_meas1 = []
  flow2_meas2 = []
  flow2_meas3 = []
  flow2_meas4 = []
  # open up the results file and read it line by line
  f=open('/home/jeffrey/minitest/Test1_Result1', 'r')
  output = f.readlines()
  for line in output:
    splitline = line.split('\t')
    time.append(splitline[0])
    flow1_rate.append(splitline[1])
    flow1_meas1.append(splitline[2])
    flow1_meas2.append(splitline[3])
    flow1_meas3.append(splitline[4])
    flow1_meas4.append(splitline[5])
    flow2_rate.append(splitline[6])
    flow2_meas1.append(splitline[7])
    flow2_meas2.append(splitline[8])
    flow2_meas3.append(splitline[9])
    flow2_meas4.append(splitline[10])
  # Convert strings to floats
  time = [float(x) for x in time]
  flow1_rate = [float(x) for x in flow1_rate]
  flow1_meas1 = [float(x) for x in flow1_meas1]
  flow1_meas2 = [float(x) for x in flow1_meas2]
  flow1_meas3 = [float(x) for x in flow1_meas3]
  flow1_meas4 = [float(x) for x in flow1_meas4]
  flow2_rate = [float(x) for x in flow2_rate]
  flow2_meas1 = [float(x) for x in flow2_meas1]
  flow2_meas2 = [float(x) for x in flow2_meas2]
  flow2_meas3 = [float(x) for x in flow2_meas3]
  flow2_meas4 = [float(x) for x in flow2_meas4]
  # plot the results
  plt.figure(1)
  plt.subplot(211)  
  plt.plot(time,flow1_rate,'r',time,flow1_meas1,'g',time,flow1_meas2,'b')
  plt.subplot(212)
  plt.plot(time,flow2_rate,'r',time,flow2_meas1,'g',time,flow2_meas2,'b')
  plt.show()

# The purpose of this is to plot the results of Test 1
if __name__ == '__main__':
  plotTest1()

