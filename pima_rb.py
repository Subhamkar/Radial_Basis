
import numpy as np
import rbf
def rBasis():

    pima = np.loadtxt('C:\Users\subha\PycharmProjects\Radial_Basis/pima-indians-diabetes.data',delimiter=',')
    pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
    imax = np.concatenate((pima.max(axis=0)*np.ones((1,9)),pima.min(axis=0)*np.ones((1,9))),axis=0).max(axis=0)
    pima[:,:8] = pima[:,:8]/imax[:8]


    target = np.zeros((np.shape(pima)[0],3));
    indices = np.where(pima[:,8]==0)
    target[indices,0] = 1
    indices = np.where(pima[:,8]==1)
    target[indices,1] = 1
    indices = np.where(pima[:,8]==2)
    target[indices,2] = 1


    order = range(np.shape(pima)[0])
    np.random.shuffle(order)
    pima = pima[order,:]
    target = target[order,:]


    train = pima[::2,:8]
    traint = target[::2]
    valid = pima[3::4,0:8]
    validt = target[3::4]
    test = pima[1::4,0:8]
    testt = target[1::4]


    net = rbf.rbf(train,traint,3)

    net.rbftrain(train,traint,0.25,2000)
    cm_train, accuracy_train = net.confmat(train,traint)
    cm_test, accuracy_test = net.confmat(test,testt)
    return cm_train, accuracy_train, cm_test, accuracy_test

if __name__ == "__main__":
    acTrain_list = []
    acTest_list = []
    for i in range(10):
        cm_train, accuracy_train, cm_test, accuracy_test = rBasis()
        acTrain_list.append(accuracy_train)
        acTest_list.append(accuracy_test)
        sum1 = sum(acTrain_list)
        sum2 = sum(acTest_list)

    avg_train = np.mean(acTrain_list)
    avg_test = np.mean(acTest_list)

    print "testing data-->"
    print cm_test, "\n", accuracy_test

    print"\n"

    print "training data-->"
    print cm_train, "\n", accuracy_train

    print "average of training data:", avg_train
    print "average of testing data:", avg_test
