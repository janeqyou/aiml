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
|Mean Absolute Error (MAE)|$\bracket{\sum_{n=1}^{n}(\tilde(y_i)-y_i)}$|less sensitive to outliers, error is linear|
|Hubert Loss (Smoothed MAE)|$\begin{cases}
\frac{1}{2}(\tilde(y)-y)^2 & for \bracket{y-f(x)} <= \sigma \\
\sigma\bracket{y-f(x)}-\frac{1}{2}\sigma^2 & otherwise
\end{cases}$|when the error is big, MAE;otherwise MSE. Hyper-parameter $\sigma$ needs to be tuned|

