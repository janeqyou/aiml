
#### machine learning 300 questions 
1. Briefly introduce SVM. What does support vector mean ? 
    - SVM (support vector machine) is a linear classifier usually for binary classification. Intuitively it tries to define a hyper plane which separates the positive and negative examples. And this hyper plane is only defined by a few points near the hyper plane. Doing this can prevent the classifier to be skewed by a few outliers
    - Formally the classifier can be defined as $w=\underset{w}\arg\max \sum_{n=1}^{N} max(0,-y_i\cdot\hat{y_i}+C) = \underset{w}\arg\max \sum_{n=1}^{N} max(0,-y_i\cdot w^{T}x_i+C)$, sometimes adding regularization term $||w||^2$
    - $max(0,-y_i\cdot w^{T}x_i+C)$ is called hinge loss. Hinge loss only cares about the sign of ground truth $sign(y_i)$ and $sign(\hat{y_i}$. There are at least two disadvanges using $max(0,-y_i\cdot w^{T}x_i)$. First
    $max(0,-y_i\cdot w^{T}x_i)$ has a de generative solution of $w=0$, adding constant C will force a non zero solution of $w$. 

    - To solve this maximization, we first reformulate the objective function:
        + to minimize $max(0,-y_i\cdot w^{T}x_i+C)$ is equivalent of maxmize $C$ given $-y_i\cdot w^{T}x_i+C < 0$ or $y_i\cdot w^{T}x_i>C$. Under this objective function $y_i\cdot w^{T}x_i$ will be forced to become large. This loss prefers not only the sign consistency between ground truth label and predicted label $w^{T}x_i$, but also prefers large value $w^{T}x_i$. hence placing points at correct side of the plane and pushing them further away
        +  Re write the objective function as $\frac{1}{||w||^2}(y_i\cdot w^{T}x_i)>C$.$\frac{1}{||w||^2}(y_i\cdot w^{T}x_i)$ is normalized distance from point.The objective function becomes maximize $C \cdot||w||^2$ given $(y_i\cdot w^{T}x_i)>C\cdot||w||^2$. Let $C=\frac{1}{||w||^2}$,the objectvie function becomes:
        <p style="text-align: center;">minimize $\underset{w}\arg\min \frac{1}{2}||w||^2$,</p>
        <p style="text-align: center;">subject to $(y_i\cdot w^{T}x_i)>1$</p>
        + The above formulation $||w||^2$ is quadraic and convex, and is subject to a set of in equality constraints. Therefore can be solved using Lagrange Method. 
        The primal form is:
        <p style="text-align: center;">$L_p=\frac{1}{2}||w||^2-\sum_{i=1}^{N}\alpha_{i}[(y_i\cdot w^{T}x_i)-1]$</p>, 
        <p style="text-align: center;">given $\alpha_{i}>0, \forall i$ and subject to $\alpha_{i}[(y_i\cdot w^{T}x_i)-1]=0$ </p>. 
        To solve this equation, let $\frac{\partial{L_p}}{\partial w}=0$, i.e. $w=\sum_{i=1}^{N} \alpha_i\cdot y_i\cdot x_i$. Using this and substitute back to the primal form, we get the dual formulation:
        <p style="text-align: center;">$L_D=\sum_{i=1}^{N}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{N}\sum_{k=1}^{N}\alpha_{i}\alpha_{k}y_iy_kx_i^Tx_k$</p>
        <p style="text-align: center;">$\alpha_{i}>0, \forall i$ and subject to $\alpha_{i}[(y_i\cdot w^{T}x_i)-1]=0$ </p>. 

        + If the data point lies on the boundary of "slab" defined by $\frac{1}{2}||w||$, $(y_i\cdot w^{T}x_i)-1=0$ and $\alpha_{i}>0$. This data point will contributes to the calculation of $w=\sum_{i=1}^{N} \alpha_i\cdot y_i\cdot x_i$
        + If the data point is correctly classified, $(y_i\cdot w^{T}x_i)-1>0$, to meet the constraints $\alpha_{i}=0$
        + If the data point is wrongly classified, $(y_i\cdot w^{T}x_i)-1<0$, to meet the constraints $\alpha_{i}=0$. 
        
        Both the cases of $\alpha_{i}=0$ indicate those data points do not affect the calculation of $w$. Only those vectors $x_i$ with $\alpha_i>0$ support the calculations of $w$. More details of derivation of SVM solution can be found [here](http://people.csail.mit.edu/dsontag/courses/ml13/slides/lecture6.pdf)     
    
2. Whats the difference between Logistic Regression and SVM ?  
    - Both of them are popular classifer for binary classification 
    - In the setting of SVM, see above, the formulation of objective function learns parameter to maximize the margin to the hyper plane from the data points which are closest to the hyper plane . In the setting of Logistic regression, the objective function tries to maximize the likelihood $p(y|x)$ which is assumed to be a Bernoulli distribution. 
    - The parameter $w$ from the two classifiers are different. Because SVM only focuses on the points closest to hyper plane, it is robust to outliers. On the other hand, logistic regression does a non linear transforms the value of $w^Tx$ therefore will have large values for points further away from plane. LR prefers a solution to place data lies fruther from the separating hyperplane (on the correct side)
    
3. K-means clustering algorithm or K Nearest Neighbor Algorithm sometimes using Euclidean distance. 

5. Discuss similarities and differences between linear regression and logistic regression 
6. Discuss the similarities among decision trees, random forest, boosting, Adaboost, GBDT and XGBoost 
7. How to build classifiers for imbalanced data sets 


