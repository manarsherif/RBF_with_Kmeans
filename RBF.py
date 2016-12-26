import numpy as np
import random
# Usage: data = dbmoon(N, d, r, w)
# doublemoon.m - genereate the double moon data set in Haykin's book titled
# "neural networks and learning machine" third edition 2009 Pearson
# Figure 1.8 pp.61
# The data set contains two regions A and B representing 2 classes
# each region is a half ring with radius r = 10, width = 6, one is upper
# half and the other is lower half
# d: distance between the two regions
# will generate region A centered at (0, 0) and region B is a mirror image
# of region A (w.r.t.x axis) with a (r, d) shift of origin
# N:  # of samples each class, default = 1000
# d: seperation of two class, negative value means overlapping (default=1)
# r: radius (default=10), w: width of ring (default=6)
def dbmoon(N=1000, d=1, r=10, w=6):
    # clear all; close all;
    # generate region A:
    # first generate a uniformly random distributed data points from (-r-w / 2, 0)
    # to (r+w / 2, r+w / 2)
    N1 = 10 * N  # generate more points and select those meet criteria
    w2 = w / 2
    done = 0
    tmp1 = np.array([[]])
    while not done:
        tmp = np.array(2 * (r + w2) * (np.random.uniform(0, 1, (N1, 1)) - 0.5))
        tmp = np.column_stack((tmp, (r + w2) * np.random.uniform(0, 1, (N1, 1))))
        # 3rd column of tmp is the magnitude of each data point
        thirdCol = np.sqrt(np.multiply(tmp[:, 0], tmp[:, 0]) + np.multiply(tmp[:, 1], tmp[:, 1]))
        tmp = np.column_stack((tmp, thirdCol.T))
        idx = np.nonzero(np.logical_and((tmp[:, 2] > (r - w2)), (tmp[:, 2] < (r + w2))))
        if tmp1.shape == (1, 0):
            tmp1 = tmp[:, 0:2][idx]
        else:
            tmp1 = np.row_stack((tmp1, tmp[:, 0:2][idx]))
        if len(idx[0]) >= N:
            done = 1
            # if not enough data point, generate more and test

    # region A data and class label 0
    # region B data is region A data flip y coordinate - d, and x coordinate +r
    data = np.column_stack((tmp1[0:N, :], np.zeros((N, 1))))
    data1 = np.column_stack(((tmp1[0:N, 0] + r), (-tmp1[0:N, 1] - d), np.ones((N, 1))))
    data = np.row_stack((data, data1))
    # plot(data(1:N, 1), data(1:N, 2), '.r', data(N+1:end, 1), data(N+1:end, 2), '.b');
    # title(['Fig. 1.8 Double moon data set, d = ' num2str(d)]),
    # axis([-r-w2 2 * r+w2 -r-w2-d r+w2])
    return data

	
data = dbmoon()
tmp =data[:,0:2]


#kmeans 
def kmeans(k):
    mu = tmp[np.random.choice(tmp.shape[0], k)]
    for j in xrange(100):
        count=np.zeros((len(mu[:]),1))
        sum_distances=np.zeros((len(mu),2))
        for x in xrange(2000):#for all points
            distance = sum(abs(tmp[x,:]-mu).T)
            closest=np.argmin(distance)
            sum_distances[closest,:]=sum_distances[closest,:]+tmp[x,:]
            count[closest]+=1
        for i in range(len(mu[:])):
            mu[i,:]=sum_distances[i,:]/count[i]
    return mu
	
#training
for k in (2,4,6,8,10):
    print "\nK=",k
    mu=kmeans(k)
    print "\ninitial centers:"
    print mu
    sigma=20.
    eita =1.
    eita =eita/2000
    weights=np.random.randn(len(mu[:])+1,1)
    for m in xrange(100):
        for i in xrange(2000):
            #forwarding
            net=np.zeros((1,1))
            for j in range(len(mu[:])):
                dist=(tmp[i,0]-mu[j,0])**2+(tmp[i,1]-mu[j,1])**2
                net=np.column_stack((net,dist))
            net=net[:,1:]
            act=np.exp((-net/(2*sigma**2)))
            out_1=np.c_[1,act]
            #out_1=act
            y=np.dot(out_1,weights)
            #w.r.t weights
            delta_weights=eita*(data[i,2]-y)*out_1
            #w.r.t mu
            #winner = np.argmin(net)
            d=tmp[i,:]-mu
            tmp1=np.multiply(data[i,2]-y,act)*weights[1:,:]/(sigma**2)
            delta_mu=eita*np.dot(tmp1,d)
            #w.r.t sigma
            delta_sigma=eita*(data[i,2]-y)*y*(sum(sum(net))/sigma**3)
            weights+=delta_weights.T
            mu+=delta_mu
            sigma+=delta_sigma
    error=0.0       
    for i in xrange(2000):
    #forwarding
        net=np.zeros((1,1))
        for j in range(len(mu[:])):
            dist=(tmp[i,0]-mu[j,0])**2+(tmp[i,1]-mu[j,1])**2
            net=np.column_stack((net,dist))
        net=net[:,1:]
        act=np.exp((-1/(2*sigma**2))*net)
        out_1=np.c_[1,act]
        #out_1=act
        y=np.dot(out_1,weights)
        error+=(data[i,2:]-y)**2
    error=(error/len(data))*100
    print "\nHidden nodes centers:\n",mu
    print "\nWeights:"
    print weights
    print "\nError = ", error
	
print "all number of hidden nodes starting from 2 provide an acceptable error"
