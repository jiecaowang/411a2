from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0, randConst=1):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  print 'Random const: %.10f ' % (randConst)
  return p, mu, vary, logProbX

def mogEMInitByKmeans(x, K, iters, minVary=0, randConst=1):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  mu = KMeans(x, K, 5)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  print 'Random const: %.10f ' % (randConst)
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def q2Find():
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz', True, False)
  maxLogProxbX = -16000
  maxRandConst = 0
  for randConst in range (1, 40):
    p, mu, vary, logProbX = mogEM(inputs_train, 2, iters, minVary, randConst)
    if maxLogProxbX < logProbX[len(logProbX) - 1]:
      maxLogProxbX = logProbX[len(logProbX) - 1]
      maxRandConst = randConst
  print "maxLogProxbX: %.7f maxRandConst: %.2f" % (maxLogProxbX, maxRandConst)
  # for 2, 31 works the best, for 3, 27 works the best

  # raw_input('Press Enter to continue.')
  # ShowMeans(mu)
  # ShowMeans(vary)

def printPi(p):
  print '\n#############################################################################################\n' 
  print 'Mixing Proportion pi: ' + str(p)
  print '\n#############################################################################################\n' 

def q2Show():
  iters = 10
  minVary = 0.01
  # FOR 2
  # randConst = 31
  # inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz', True, False)
  # FOR 3
  randConst = 27
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz', False, True)
  p, mu, vary, logProbX = mogEM(inputs_train, 2, iters, minVary, randConst)
  printPi(p)
  raw_input('Press Enter to continue.')
  ShowMeans(mu)
  ShowMeans(vary)

def showMultiMeans(means):
  """Show the cluster centers as images."""
  for i in xrange(means.shape[1]):
    plt.figure(i)
    plt.clf()
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.draw()
    raw_input('Press Enter.')

def q3():
  iters = 10
  minVary = 0.01

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  p, mu, vary, logProbX = mogEMInitByKmeans(inputs_train, 20, iters, minVary)
  # printPi(p)
  raw_input('Press Enter to continue.')
  p, mu, vary, logProbX = mogEM(inputs_train, 20, iters, minVary)
  raw_input('Press Enter to continue.')
  # ShowMeans(mu)
  # ShowMeans(vary)

def calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, newInputX):
  probX2 = np.exp(logProbX2)
  probX3 = np.exp(logProbX3)
  probXGiven2 = np.exp(mogLogProb(p2, mu2, vary2, newInputX))
  probXGiven3 = np.exp(mogLogProb(p3, mu3, vary3, newInputX))
  probX = probX2 * probXGiven2 + probX3 * probXGiven3
  prob2GivenX = (probX2 * probXGiven2)/probX
  prob3GivenX = (probX3 * probXGiven3)/probX
  return prob2GivenX, prob3GivenX, findPrediction(prob2GivenX, prob3GivenX)

def findPrediction(prob2GivenX, prob3GivenX):
  diff = prob2GivenX - prob3GivenX
  return np.floor(diff + 1) # 0 is 2, 1 means 2, 50vs50 is 3 as well

def findFalsePredictedPercent(target, predictionByInt):
  return float(np.sum(np.absolute(target - predictionByInt)))/np.size(target)

def DisplayClassificationErrorPlot(numComponents, train_error, valid_error, test_error):
  plt.figure(1)
  plt.clf()
  plt.plot(numComponents, train_error, 'b', label='Train')
  plt.plot(numComponents, valid_error, 'g', label='Validation')
  plt.plot(numComponents, test_error, 'r', label='Test')
  plt.xlabel('numComponents')
  plt.ylabel('Classification Error Percent')
  plt.legend()
  plt.draw()
  raw_input('Press Enter to exit.')

def q4():
  iters = 10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  print(errorTrain)
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, logProbX2 = mogEMInitByKmeans(train2, K, iters, minVary)
    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, logProbX3 = mogEMInitByKmeans(train3, K, iters, minVary)
    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    # prob2GivenValid2, prob3GivenValid2, predictValid2 = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, valid2)
    # prob2GivenValid3, prob3GivenValid3, predictValid3 = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, valid3)

    # prob2GivenTest2, prob3GivenTest2, predictTest2 = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, test2)
    # prob2GivenTest3, prob3GivenTest3, predictTest3 = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, test3)
    
    prob2GivenValid, prob3GivenValid, predictTrain = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, inputs_train)
    prob2GivenValid, prob3GivenValid, predictValid = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, inputs_valid)
    prob2GivenTest, prob3GivenTest, predictTest = calculatePdGivenX(p2, mu2, vary2, logProbX2, p3, mu3, vary3, logProbX3, inputs_test)
    errorTrain[t] = findFalsePredictedPercent(target_train, predictTrain)
    errorValidation[t] = findFalsePredictedPercent(target_valid, predictValid)
    errorTest[t] = findFalsePredictedPercent(target_test, predictTest)

    
  # Plot the error rate
  
  #-------------------- Add your code here --------------------------------
  DisplayClassificationErrorPlot(numComponents, errorTrain, errorValidation, errorTest)
  
  

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------

  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  # q2Find()
  # q2Show()
  # q3()
  q4()
  # q5()

