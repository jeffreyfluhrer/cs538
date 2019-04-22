def ComputeFlowNum(host_src, host_dst):
    temp = 1
    num_hosts = 4 + 1
    for i in range(1,num_hosts):
        for j in range(1,num_hosts):
            #print(i,j)
            if i == j:
                pass
            elif i == host_src and j == host_dst:
                return temp
            else:
                temp += 1
