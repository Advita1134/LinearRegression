"""
Simple Linear Regression

This is called "Simple" Linear Regression because there is only one input value, 
or a single feature to get an output

"""
import random
import matplotlib.pyplot as plt

## Step One: Data
X_train = [1, 3, 7]
y_train = [3, 5, 15]
X_test = [1.5, 4, 6, 4.5, 8]
y_test = [3, 8, 12, 9, 16]


## Step 2: Algorithm
# Equation: y = mx + b
m = random.randint(0,100) # random m
b = random.randint(0,100) # random b
print("random m: ", m, "random b: ", b)

## Step 3: Training
trainingerrorlist = []
lr = .005 # learning rate
for a in range (1000): # each loop of this is one epoch
  for i in range (len(X_train)):
    y_hat = m * X_train[i] + b
    trainingerror = ((y_hat-y_train[i])**2)*(.5)
    trainingerrorlist.append(trainingerror)
    mDerivative = (y_hat-y_train[i])*(X_train[i])
    bDerivative = y_hat-y_train[i]
    m = m - lr*(mDerivative)
    b = b - lr*(bDerivative)
print("final m: ", m, "final b:", b)

## Step 4: Testing/Prediction
testingerrorlist = []
y_predlist = []
for c in range (len(X_test)):
  y_pred = m*X_test[c] + b
  y_predlist.append(y_pred)
  print("Prediction for ", X_test[c], "is ", y_pred)
  testingerror =  ((y_pred-y_test[c])**2)*(.5)
  testingerrorlist.append(testingerror)
errorsum = sum(testingerrorlist)
print(errorsum)


# Analysis
plt.scatter(X_train, y_train, c = "purple", label = "Training Data") # Plotting the original data
maxX = max(X_train)
yofmaxX = m*maxX + b
plt.plot([0, maxX],[b, yofmaxX], label = "Line of Best Fit")
plt.scatter(X_test, y_predlist, marker = "x", s = 50, c = "red", label = "Testing Data")
plt.title("Linear Regression")
plt.xlabel("x-values")
plt.ylabel("y-values")
plt.legend()
plt.grid()

plt.figure()
plt.plot(trainingerrorlist)
plt.title("Training Error Plot")
plt.xlabel("Number of Runs") # plot of the error as m and b changes
plt.ylabel("Training Error of Each Run")




