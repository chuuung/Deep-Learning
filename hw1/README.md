# Handwritten Digits Classfication
## Introduction
we will be analyzed the normalized handwritten digits,
automatically scanned from envelopes by the U.S. Postal Service. The original
scanned digits are binary and of different sizes and orientations; the images here
have been deslanted and size-normalized, resulting in 16 × 16 grayscale
images (Le Cun et al., 1990).  
The goal is to predict the digit id (0-9) by these 16 × 16 grayscale images.

## Models
Fit the following 3 different neural network (NN) structures (Net-1 to Net-3):  
Net-1: One hidden layer with 12 nodes;  
Net-2: Two hidden layers with 64 and 16 nodes, respectively;  
Net-3: Four hidden layers with 64, 64, 16, and 16 nodes, respectively.  

For each NN structure, do  
- Randomly split the training dataset in a 80: 20 ratio for training and
validation, respectively
- Use the training part to train the model with different maximum steps for the training of the NN (i.e., numbers of epochs), and calculate their misclassification rates on the validation part. Plot numbers of epochs (xaxis: 10 ≤ x ≤ 30) versus misclassification rates (y-axis). Decide the “best” number of epochs based on the plot.

