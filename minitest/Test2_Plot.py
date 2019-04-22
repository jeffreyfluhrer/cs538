import matplotlib.pyplot as plt

# The purpose of this is to plot the results of Test 1
if __name__ == '__main__':
  # Create empty lists for time and first data column
  time = []
  flow1_rate = []
  # open up the results file and read it line by line
  f=open('/home/jeffrey/minitest/Test2_Result1', 'r')
  output = f.readlines()
  for line in output:
    splitline = line.split('\t')
    time.append(splitline[0])
    flow1_rate.append(splitline[1])
  # Convert strings to floats
  time = [float(x) for x in time]
  flow1_rate = [float(x) for x in flow1_rate]
  # plot the results  
  plt.plot(time,flow1_rate)
  plt.show()
