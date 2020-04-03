import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lamda = 3
learning=0.2

def sigmoid(z):
    return Activation(z)*(1-Activation(z))

def Activation(z):
    return 1 / (1 + np.exp(-z))

def Loss(X,Y,W1,W2,ForwardY):
    
    global learning,lamda
    m=len(X)
    temp = np.sum(-1*Y*np.log(ForwardY)-(1-Y)*np.log(1-ForwardY))/m
    temp1 = lamda*(np.sum(W1[:,1:]**2) + np.sum(W2[:,1:]**2))/(2*m)
    return temp+temp1


def NeuralNet(X, Y, W1, W2):
    global learning, lamda
    Z1 = X @ W1.T
    H = Activation(Z1)

    ones = np.ones([H.shape[0], 1])
    H = np.concatenate((ones, H), axis=1)

    Z2 = H @ W2.T
    ForwardY = Activation(Z2)

    return H, ForwardY, Z1


def Gradient1(X,Y,H,ForwardY,Z1,W2):
    global learning, lamda
    B2 = (ForwardY - Y)
    B1 = (B2@W2[:,1:])*sigmoid(Z1)
    GW1J = B1.T @ X
    return GW1J

def Gradient2(X,Y,H,ForwardY,Z1):
    global learning, lamda
    B2 = (ForwardY - Y)
    GW2J = B2.T @ H
    return GW2J


def W1_Gradient(X,Y,W1,H,ForwardY,Z1,W2):
    global learning, lamda
    m=len(X)
    tempW = np.array(W1)
    term1 = (1/m)*Gradient1(X,Y,H,ForwardY,Z1,W2)
    tempW[:,0] = 0
    term2 = (lamda/m)*W1
    return term1+term2


def W2_Gradient(X,Y,W2,H,ForwardY,Z1):
    global learning, lamda
    m=len(X)
    tempW = np.array(W2)
    term1 = (1/m)*Gradient2(X,Y,H,ForwardY,Z1)
    tempW[:,0] = 0
    term2 = (lamda/m)*W2
    return term1+term2


def Gradient_Descent(X,Y,W1,W2):
    global learning, lamda
    k=0
    cost = []
    while(k<500):
        
        H,ForwardY,Z1 = NeuralNet(X,Y,W1,W2)
        W1 = W1 - learning*W1_Gradient(X,Y,W1,H,ForwardY,Z1,W2)
        W2 = W2 - learning*W2_Gradient(X,Y,W2,H,ForwardY,Z1)
        cost.append(Loss(X,Y,W1,W2,ForwardY)) 
        k += 1

    return W1,W2,cost,k

def loaddata():
    Yold = pd.read_csv('Y.csv', header=None)
    Yold = Yold.values
    Y = np.zeros((5000, 10))
    for i in range(5000):
        if (Yold[i] == 10):
            Y[i, 9] = 1
        else:
            Y[i, (Yold[i] - 1)] = 1
    X = pd.read_csv('X.csv', header=None)
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    W1 = (pd.read_csv('initial_W1.csv', header=None)).values
    W2 = (pd.read_csv('initial_W2.csv', header=None)).values

    return X,Y,W1,W2

def Prediction(Y, ForwardY):

    Y_Actual = np.array([(np.where(r == 1)[0][0]) + 1 for r in Y]).reshape(5000, 1)
    Y_Predicted = np.array([(np.where(r == max(r))[0][0]) + 1 for r in ForwardY]).reshape(5000, 1)

    Difference = Y_Actual - Y_Predicted

    ones = np.ones([5000, 1])
    zeros = np.zeros([5000, 1])

    Correct = np.where(Difference == 0, ones, zeros)

    print("\nAccuracy: ", 100 * (np.sum(Correct) / 5000),"%")

    return Y_Actual,Y_Predicted



def Testcases(Actual, Predicted):

    print("\n")

    print("\nTEST CASES:\n")
    print("Index 2171: Predicted-->", Predicted[2170, 0], "\tActual-->", Actual[2170, 0])
    print("Index 144: Predicted-->", Predicted[144, 0], "\tActual-->", Actual[144, 0])
    print("Index 1581: Predicted-->", Predicted[1581, 0], "\tActual-->", Actual[1581, 0])
    print("Index 2445: Predicted-->", Predicted[2445, 0], "\tActual-->", Actual[2445, 0])
    print("Index 3392: Predicted-->", Predicted[3392, 0], "\tActual-->", Actual[3392, 0])
    print("Index 814: Predicted-->", Predicted[814, 0], "\tActual-->", Actual[814, 0])
    print("Index 1377: Predicted-->", Predicted[1377, 0], "\tActual-->", Actual[1377, 0])
    print("Index 528: Predicted-->", Predicted[528, 0], "\tActual-->", Actual[528, 0])
    print("Index 3944: Predicted-->", Predicted[3944, 0], "\tActual-->", Actual[3944, 0])
    print("Index 4627: Predicted-->", Predicted[4627, 0], "\tActual-->", Actual[4627, 0])
    print("\n\n")


def Plot(count,cost):
    print("\n\n")

    fig, ax = plt.subplots()
    ax.plot(np.arange(count), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('loss')
    ax.set_title('Loss function over iterations')
    plt.show()



def main():
    global learning, lamda
    X,Y,W1,W2=loaddata()
    t2, t3, t1 = NeuralNet(X,Y, W1, W2)
    H, ForwardY, Z1 = NeuralNet(X,Y, W1, W2)

    print("\n\n")
    print("Cost before Learning:", Loss(X, Y, W1, W2, ForwardY))
    print("Number of iterations to be run:",500)
    print("Learning rate: ",learning,"\tLambda: ", lamda)


    W1_, W2_, cost, count = Gradient_Descent(X, Y, W1, W2)

    H, ForwardY, Z1 = NeuralNet(X,Y, W1_, W2_)
    print("Cost after Learning:", Loss(X, Y, W1_, W2_, ForwardY))

    Actual, Predicted =  Prediction(Y, ForwardY)

    Testcases(Predicted,Actual)

    Plot(count,cost)


if __name__ == '__main__':
    main()

