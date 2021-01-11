## Probability, Machine Learning, Deep Learning Review (2020Q4-2021Q1)
### Machine Learning, A probability Perspective
#### Chapter 1,2,3
1.2.1.2 <code>The need for probablistic predictions</code>
`https://www.npmjs.com/get-npm`

#### udemy - The complete neural network boot camp 
* section 2 error functions
| Name        | Equations   | Notes    |
| ----------- | ----------- | ---------
| Mean Squared Error (MSE)| $\frac{1}{n}\sum_{n=1}^{n}(\tilde(y_i)-y_i)^2$ || more sensitive to outliers|
| Root Mean Squared Error (RMSE)| $\sqrt{\frac{1}{n}\sum_{n=1}^{n}(\tilde(y_i)-y_i)^2}$       | more sensitive to outliers|
|Mean Absolute Error (MAE)|$[\sum_{n=1}^{n}(\tilde(y_i)-y_i)]$|less sensitive to outliers, error is linear|
|Hubert Loss (Smoothed MAE)|$\begin{cases}
\frac{1}{2}(\tilde(y)-y)^2 & for [y-f(x)] <= \sigma \\
\sigma[y-f(x)]-\frac{1}{2}\sigma^2 & otherwise
\end{cases}$|when the error is big, MAE;otherwise MSE. Hyper-parameter $\sigma$ needs to be tuned|
|Binary Entropy Loss |$\sum_{i=1}^{n}(y_ilog(p_i)+(1-y_i)log(1-p_i)$|used when there are two classes 0 or 1; or the output value is between 0 and 1. In fact the MLE version of Bernoulli distributed output|
|Cross Entropy Loss |-\frac{1}{n}$\sum_{i=1}^{n}$\sum_{j=1}^{C}(y_jlog(\tilde{y_j})$|y_j is the actual label;loss term will only be non zero when y_j is 1. Penalizes probabilities of correct class only|
|Soft max function |\sigma{z}_j=\frac{e^(z_j){\sum_{k=1}{K}e^(z_k)}|hand in hand with cross entropy loss; apply softmax function to output before calculating cross entropy|
|KL divergence loss |$KL(y||\tilde{y}=\sum_{i}y_ilog\frac{1}{\tilde{y_i}-\sum_{i}y_ilog\frac{1}{y_i}$)|cross entropy - entropy|








