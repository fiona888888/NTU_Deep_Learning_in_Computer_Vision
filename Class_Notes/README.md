# Deep Learning in Computer Vision Class Note

# Introduction (week: 02/20)

## AI Branches

![jpg](images/img1.jpg)

**AI (Artificial Intelligence):**
The broadest term, AI refers to the simulation of human intelligence in machines, enabling them to perform tasks that typically require human cognition, such as reasoning, problem-solving, and decision-making.

**ML (Machine Learning):**
A subset of AI, ML involves training algorithms to learn patterns from data and make predictions or decisions without explicit programming. ML models improve performance with more data over time.

**DL (Deep Learning):** 
A specialized subset of ML, DL uses neural networks with multiple layers (deep neural networks) to process large amounts of data and perform complex tasks such as image recognition, natural language processing, and autonomous driving.

## Categories of Machine Learning

**Supervised learning:**
- label/ground-truth of data is given
- A type of machine learning where the model is trained on labeled data, mapping inputs to known outputs.

**Unsupervised learning:**
- label/ground-truth of data is unknown.
- A type of machine learning where the model finds patterns and structures in unlabeled data without predefined outputs.
- transforming the data into other representations.
- ex: image processing, e.g., local binary pattern  
clustering: e.g., k-means clustering

**Reinforcement learning:**
determine the optimal policy (i.e., the best
set of actions) based on the `reward` learn from the `environment`.

## Computer Vision
Make the computers understand images and videos. Identify relationships between objects in the image through image processing.

**More Applications: Vision in Space**  
Vision systems (JPL) used for several tasks:  
- Panorama stitching:  
① detect feature points in image pairs.  
② Find matching features in adjacent images.  
③ Use a unified coordinate sys, to stitch the images together.
- 3D terrain modeling:  
① Detect feature points  
② obtain camera projection matrix.  
③ project those feature points back to their 3D coordinates
- Obstacle detection, position tracking
- For more, read “Computer Vision on Mars” by Matthies et al.

![jpg](images/img2.jpg)

# Data Representation and Normalization (week: 02/24)
- Conventional ML : Use engineered (human selected) features to train model
- DL : Train model with raw data, the model will automatically extract features for you

## Data Representation
prior to training a model, you must determine:  
➢ label ground-truth: eg. binary classification 0 or 1  
➢ representation of input:  
> - raw data : DL
> - transform to other features : ML

## Normalization (Feature scaling)
**When:** Input features have different order of magnitudes  
**Why:** poor performance due to:  
➢ Dominance of feature  <img src="images/img3.jpg" alt="Deeplearning AI" align="right" width="200">   
➢ Numerical stability  
➢ Convergence issues  

**Methods:**
-  Min-Max Normalization:  (normalize to 0 - 1)  
$x[:, j]_{normalization} = \frac{x[:, j] - min(x[:, j])}{max(x[:, j]) - min([:, j])}$    
    ➢ Feature-wise scaling, across all the samples  
    ➢ Most common scaling technique  
    ➢ cons: think about inherent constraints in physics

- Mean normalization:
![jpg](images/img4.jpg)
- Z-score:
![jpg](images/img5.jpg)

**How to apply:**  
- Classification  
➢ Usually no scaling in outputs.  
➢ Need scaling in inputs.  
- Regression  
➢ Usually need to scale in both inputs and outputs.  
➢ If you do scaling while training the model, need scaling    during testing  

    **🔹Testing Data:**  
    1. apply feature transformation  
    2. scaling input & output using the same scaling factors   employed during training
    3. scale the predictions back to original values. 

    ![jpg](images/img6.jpg)



# Model evaluation (week: 02/27)

## Assumption in Machine Learning
- How you determine your training dataset so that your model works
on the unseen (test) dataset?  
    1. Training dataset must be ``representative enough``
    2. In other words, training and testing datasets are  `on the same distribution`  

    <img src="images/img7.jpg" width="250"/>

## Robustness Evaluation
**Goal:** check how bad your model could be when you change your selection of training and testing data set.  
1.  Do **repeated trials**! (Change the selected training and testing)
2.  show **statistics** of the performance of your model on
the **testing dataset**  ex: boxplot, normal distribution of multiple trials.

## Overvitting 
 **Overfitting:** model works well on training dataset, but performs poor on testing dataset.  
 - Blue dots: 2D feature vectors of men images  
 - Red dots: 2D feature vectors of women images  
 - Green curve: an overfitted model
 - Black curve: a more general model

 <img src="images/img8.jpg" width="150"/>  

 > *Detect Outliers: Suppose small portion of outliers => How to detect outliers?  
Once the model is trained appropriately it should be a general model. Feed the training samples into the model to make predictions. the samples with relatively larger
**"prediction error"** could be potential outliers.

### Reasons and solutions to Overfitting
1. Training dataset is not representative 

    ➔ re-select the training samples  

2.  Model complexity is high  

    ➔ reduce the model complexity  

3.  Train too much   

    ➔ reduce the number of training epochs

### When to stop Training
<img src="images/img9.jpg" width=""/> 

1. keep testing dataset for final evaluation
2. Stop when error(loss) on **validation dataset start to increase**   

(Overfit: accuracy of training dataset high validation low. Underfit:accuracy of training dataset low validation low )

## Confusion matrix and Accurracy
