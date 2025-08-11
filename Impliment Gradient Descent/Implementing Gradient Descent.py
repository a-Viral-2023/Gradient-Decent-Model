import matplotlib.pyplot as plt
# is used for plotting
import numpy as np
# is used for numerical arrays and other math packages
import pandas as pd 
# used for reading CSV files
# x = inputs
# y = targets

# the variables and what they are defined as
epochs = 1000
errors = []
# weights = np.random.rand(x.shape[1]) this line was moved behind x and y
    # gives us two random weights
    #np.random.rand(2) gives us two random numbers between 0 and 1
    # VVV  for inputs.shape[1]  VVV
     #You’re saying:
     # “Give me the number of columns (features) in the data.”
     # If each sample has 2 numbers (like x and y coordinates), 
     # inputs.shape[1] will be 2.
bias = np.random.rand()
# random bias's as random starting points helps the model explore different
# paths to find the best solution
# bias just shifts the output up or down
learning_rate = 0.1
#last_loss = None
# the loss is a number that twlls us how wrong the current iteration of 
# the model was


def plot_points(X,y):
# creates a method in java 
## or a fucntion in python with arguments
#                      x and y 
    admitted = X[np.argwhere(y==1)]
# y==1 creates a boolean array where the label is 1 from the
#notebook formula
#           np.argwhere( y == 1)
# returns indexes in the list where the values make this 
# condition true
#   x[np.argwhere(y==1)] means give me only the rows defined 
# in the overarching variable x where y = 1
# so for example:
# x = np.array([
  #  [0.3, 0.5],  # row 0
  #  [0.6, 0.8],  # row 1
  #  [0.1, 0.2],  # row 2
   # [0.9, 0.7],  # row 3
#])

#y = np.array([0, 1, 0, 1])  # Labels for each row

#so what it is essentially giving you is this:
# array([[1],[3]]) 
# and from that the computer deterimes what you want is
# from rows 1 and 3
# so the final result is array([[[0.6, 0.8]],  # row 1 of X
   #                           [[0.9, 0.7]],  # row 3 of X
#                                         ])
# So admitted is now a NumPy array containing 
# only the feature rows for y == 1.
#_________________________________________________________________
    rejected = X[np.argwhere(y==0)]
# same concepts exacpt this time it is looking for the array 
#rows of variable w which contains y equaling 0 for that row
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    # (from argwhere), extract the first coordinate (x-axis).
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'blue', edgecolor = 'k')
    # extract second coordinate (y-axis)
# plt.scatter plots the two sets of points 
# s[0][0] means: from each 3D-wrapped point 
# (from argwhere), extract the first coordinate (x-axis).

def display (m, b, color = 'g--'):
   # creates a dunciton named display with arguments m
   # as the slope, b as the y intercept, and the optional
   # color which g-- means a green color "dashed" line
   plt.xlim(-0.05,1.05)
   # shows the x axis limits on the plot graph viewer 
   # similar to a calculator it only shows the zoom range 
   # of the graph from -0.05 to 1.05
   plt.ylim(-0.05,1.05)
   # and vise versa for the y axis

   x = np.arange(-10, 10, 0.1)
   # creates a NumPy based array, of x values 
   # basically the array starts at -10 and ends at and increased by 0.1 each
  
   plt.plot(x, m*x+b, color)
# draws a 2d line on the graph using the array of x values


data = pd.read_csv(r"C:\Downloads\Gradient Descent\Impliment Gradient Descent\Gradient Data.csv", header = None)
# the r in front of the path makes it a raw string so that backslashes don;t need escaping.
# Reads the CSV file named data.csv using Pandas. 
# header=None tells Pandas not to treat the first row as
#  column names, and instead label columns numerically
# : 0, 1, 2, etc 
x = np.array(data[[0,1]])
# explained in notebook
y = np.array(data[2])
plot_points(x,y)
#This calls a custom function named plot_points,
#  which likely takes the features X and the labels y, 
# and creates a scatter plot.
plt.show()
#Tells Matplotlib to display the plot window with the 
# scatter plot.
#i moved this here from line 13 since x wasn't defined yet until this point
# VVV                           VVV
#weights = np.random.rand(x.shape[1])
#update: this line is not necessary because weights alreadight generates 
#new weights
def sigmoid_formula(x):
   return (1/(1 + np.exp(-x)))

def output_formula(inputs,weights, bias):
   return sigmoid_formula(np.dot(inputs,weights)+ bias)

def error_formula(y, output):
    return -1 * (y * np.log(output) + (1 - y) * np.log(1 - output))


def how_to_update_weights(x, y, weights, bias, learnrate):
   output = output_formula(x, weights, bias)
   d_error = y - output  # what is this? It is how far off your prediction is 
   #                                     from the actual label
   weights += learnrate * d_error * x
   bias += learnrate * d_error
   return weights, bias






# below this is the function used to train this alogorithm 

np.random.seed(44)
# produces the same random values every time for reproducibility 

def train(x, y, epochs, learnrate, graph_lines = False):
   # graph_lines is a parameter of the function, basically its telling not to 
   # graph lines from the points 
   errors = []
   #assigning the variable errors to an empty list
   n_records, n_weights = x.shape
   # creating the baseline for the shape of the input array
   #n_records is the number or rows, inputs or samples
   #n_weights is the number columns or the weight and its value per 
   # sample 
   last_loss = None
   bias = 0
   weights = np.random.normal(scale = 1 / n_weights** .5, size = n_weights)
   display(-weights[0]/ weights[1], -bias/weights[1])
   
# np.random.normal creates numbers drawn from a normal distribution of numbers which are
# closer to the average and are more likely to be drawn 
   for e in range(epochs):
      del_w = np.zeros(weights.shape)
      # what is np.random.normal and what is np.zero and what is weights.shape
      # np.zeros creates an array of zero's and weights.shape creates the dimentions of that array
      # based on the number of weights
      for x_i, y_i in zip(x,y):
         # what is zip?
    # zip pairs elements from two sequences zip([1,2,3] ,['a','b','c']
    # turning them  into this  
          weights,bias = how_to_update_weights(x_i , y_i , weights , bias, learnrate)


#printing log-loss errors in case something goes wrong
      out = output_formula(x, weights,bias)
      loss = np.mean(error_formula(y,out))
      errors.append(loss)
# Appends this current epoch’s loss to a list called errors, for later plotting or tracking

      if e % (epochs / 10) == 0: 
          print ("\n========== Epochs", e, "==========")
   #prints the current iteration of the the run cycle

      if last_loss and last_loss < loss:
         print("Train loss:" , loss, " Warning - Loss Increasing")
      else:
         print ("Train loss: ", loss)
         last_loss = loss

   # Converting the output (float) to boolean as it is a binary classification

   #e.g. 0.95 gets rounded up to -- > True(= 1), But if 0.35
   # gets rounded down to -- > False(= 0)
      predictions = out > 0.5

      accuracy = np.mean(predictions == y)
      print("Accuracy ", accuracy)

      if graph_lines and e % int (epochs / 100) == 0:
         display(-weights[0] / weights[1], -bias/weights[1])


   return weights, bias, errors



#things to note: data.csv is a CSV (Comma-Separated Values) file
# In a machine learning context, this file usually contains:
#Input features (e.g., feature1, feature2)
#Target values (e.g., target column for classification or regression)


# Now we have to train the model here
weights, bias, errors = train(x, y, epochs, learning_rate, graph_lines = False)

# plotting the (right/wrong) solution boundary line (the final part)
plt.title("Solution Boundary")
display(-weights[0]/ weights[1], -bias/weights[1])


# actually physically plotting the data part
plot_points(x, y)
plt.show()

#plotting the errors found in each run
plt.title("Error Plot")
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.plot(errors)
plt.show()

