## Probability, Machine Learning, Deep Learning Review (2020Q4-2021Q1)
### Machine Learning, A probability Perspective
#### Chapter 1,2,3
1.2.1.2 <code>The need for probablistic predictions</code>
`https://www.npmjs.com/get-npm`

#### Udemy - The complete neural network boot camp 
* Section 2 error functions
    * Mean Squared Error (MSE):$\frac{1}{n}\sum_{n=1}^{n}(\tilde(y_i)-y_i)^2$. more sensitive to outliers.
    * Root Mean Squared Error (RMSE): $\sqrt{\frac{1}{n}\sum_{n=1}^{n}(\tilde(y_i)-y_i)^2}$
    * Mean Absolute Error (MAE):$|\sum_{n=1}^{n}(\tilde(y_i)-y_i)|$.Less sensitive to outliers, error is linear.
    * Hubert Loss (Smoothed MAE): $\begin{cases}\frac{1}{2}(\tilde(y)-y)^2 & for |y-f(x)| <= \sigma \\\sigma[y-f(x)]-\frac{1}{2}\sigma^2 & otherwise\end{cases}$. When the error is big, MAE;otherwise MSE. Hyper-parameter $\sigma$ needs to be tuned
    * Binary Entropy Loss. $\sum_{i=1}^{n}(y_ilog(p_i)+(1-y_i)log(1-p_i)$. Used when there are two classes 0 or 1; or the output value is between 0 and 1. In fact the MLE version of Bernoulli distributed output.
    * Cross Entropy Loss $-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{C}(y_jlog(\tilde{y_j})$. $y_j$ is the actual label;loss term will only be non zero when $y_j$ is 1. Penalizes probabilities of correct class only.
    * Soft max function. $\sigma({z}_j)=\frac{e^(z_j)}{\sum_{k=1}{K}e^(z_k)}$. hand in hand with cross entropy loss; apply softmax function to output before calculating cross entropy.
    * KL divergence loss. $KL(y||\tilde{y})=\sum_{i}y_ilog\frac{1}{\tilde{y_i}}-\sum_{i}y_ilog\frac{1}{y_i}$. cross entropy - entropy.
    * Contrastive loss: $yd^2+(1-y)(max(margin-d,0))^2$. $d=\sqrt{O(x_i)-O(x_j)}$
        - distance-based loss function used to learn discriminative features for images
        - calculated on pairs and ensures that semantically similar examples are embedded close together. Positive pairs are images that are semantically similar; negative pairs are images that are semantically different
        - maximize the difference between points in negative pairs and minimize distance between points in positive pairs
    * Hinge loss: $max(0,1-y*\tilde{y})$
        - consider the following plot of $1-x$ and $max(0,1-x)$. Negative values of $1-x$ are suppressed. Value of $max(0,1-x)$ is linear/proportional to $1-x$ when $1-x > 0$ 
            ![Hinge-loss](/Users/qxy001/Documents/personal_src/aiml/notes/hinge-loss.png)

    * Triple Ranking Loss: $max(|f_a-f_p|^2-|f_a-f_n|^2+m)$. Minimize distance between positive samples and anchors and maximize distance between negative samples and anchor. m is margin, negative samples distance towards anchor needs to be larger than positive samples distance towards anchor. 
        - Easy triplets need to be avoid. 
        - Choose "hard negatives", false positive samples. And choose within online training mini batch. 

* Section 2 Activation function 
    * Why needs activation function?
    Many real world data is not linearly seperatable. Activation function provides powerful non linear transformation from intpu to output so neural network can learn complex functions
    * sigmoid activation: $\phi(z)=\frac{1}{1+e^{-z}}$
        * the derivatives of sigmoid is $\phi(z)(1-\phi(z))$ small and several multiplication will cause gradient vanishing problem. 
    * (hyperbolic) tahn activation: $\phi(z)\frac{1-e^{-2z}}{1+e^{-2z}}$
        * the derivative of tahn is: $1-tahn(x)^2$. In terms of vanishing gradient, it is better than sigmmoid. Although couping with RNN, there is still a vanishing gradient problem, that is due to the architecture of RNN.
    * Rectified Linear Unit (ReLU): $R(z)=max(0,z)$
        * the derivative of ReLU is: $\prime{f}(x)=\begin{cases} 1 & for x>0 \\0 & otherwise \end{cases}$.
        * when the input is negative the gradient will die
        * when using normal distribution to initialize the weight, half of weights are negative. Using ReLu means setting half of the gradients to zero. 
    * Parametric ReLU: $R(z)=max(\alpha,z)$
        * the derivative of ReLU is: $\prime{f}(x)=\begin{cases} 1 & for  x>0 \\\alpha & otherwise \end{cases}$.
        * if $\alpha = 0.01$, Leaky ReLU. To deal with dead neuron 
    * Gated Linear Unit: to activate a layer of d neurons, expand to dx2. The first d neurons will be linearly activated; the second d neuros will be activated using sigmoid. The final output is an Hadamard product of two parts.  
        * If previous activation is $Xw+b$, now the activation becomes $(Xw_1+b_1)\otimes{sigmoid(Xw_2+b_2)}$
        * So two sets of weights $w_1$ and $w_2$ needs to be learnt 
        
* Section 3 Optimization Algorithms in Neural Networks 

    * Gradient Descent:
        * minimize error with respect to weights of the neural network. Learning rate determines the step size of the update 
        * An Epoch is one complete pass through all the samples 
        
    * Batch Gradient Descent:
        * Take all samples and feed them into neural network. Calculate error based on all samples and update weights 
        * One iteration is one epoch
        * Not meomory efficient ; 
         
    * Stochastic Gradient Descent:
        * Take in one sample at a time 
        * Perform weights update for each sample at a time 
        * n samples = n iterations per epoch 
        * High variance of updates and cause the value of objective function to fluctuate 
        
    * Mini-batch Gradient Descent:
        * Perform weight updates on a batch of samples 
        * take n training samples (batch size) -> Feed to the network 
        * n samples / batch size = # iterations per Epoch 
        
    * Exponential Weighted Average and Bias Correction 
        * to smooth a series of values ${v_t}$, exponential weighted average is: 
            $v_t = (1-\beta)\theta + \beta v_{t-1}$, where $v_{t-1}$ is the pervious value of v. 
        Expanding the recursive terms,  
                $v_t = (1-\beta)\theta + \beta (\sum_{n=1,t=t}^{n=n,t=1}(1-\beta)^n\theta_{t-1} )$,
        * Because of the term $(1-\beta)^n$, at very  beginning $\theta_{0}$,$\theta_{1}$ ... are very small. Add a multiplicative term $\frac{1}{1-\beta^t}$ 
        * 







