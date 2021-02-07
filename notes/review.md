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
    
    * Momentum:
        * SGD + Momentum is a very powerful optimization.
        * Can use exponentially moving average of gradient to reduce the oscillations in the vertical direction error and speed up the horizontal direction to converge 
        * Compute $dw$ and $db$ on your mini-batch. And calculate the new gradient direction as:
            <p style="text-align: center;">$v^{t}_{dw}=\beta*v^{t-1}_{dw}+(1-\beta)dw$,</p>
            <p style="text-align: center;">$b^{t}_{db}=\beta*b^{t-1}_{db}+(1-\beta)db$,</p>
        The new weight update is:
            <p style="text-align: center;">$w=w-\alpha*v^{t}_{dw}$</p>,
            </p>$b=b-\alpha*v^{t}_{db}$,</p>
        Effectively we are averaging $\frac{1}{1-\beta}$ points 
![momentum](/Users/qxy001/Documents/personal_src/aiml/notes/momentum.png)

        * Momentum still suffers from ocilations in vertical direction 
        
    * RMSProp: an adaptive learning rate method 
        * devides the learning rate by an expoentially decaying average of squared gradient
        * $\beta$ is suggested to be 0.9 and $\alpha$ is suggested to be 0.001
        * Each iteration the learning rate is changed to be:
            <p style="text-align: center;">$w=w-\frac{\alpha}{\sqrt{s^{t}_{dw}+\epsilon}}*dw$,</p>
            <p style="text-align: center;">$s^{t}_{dw}=\beta*s^{t-1}_{dw}+(1-\beta)*(dw)^2$,</p>
        * when $dw$ is really small, $dw$ is smaller, and $\frac{\alpha}{\sqrt{s^{t}_{dw}+\epsilon}}$ is large. Vice versa

    * Adam Optimization (Adaptive Moment Estimation)：
        * momentum update:$m_{t}=\beta_{1}m_{t-1}+(1-\beta)*dg$, 
        * RMSProp update: $v_{t}=\beta_{2}v_{t-1}+(1-\beta)*dg^2$,
        * To prevent $m_{t}$ and $v_{t}$ from becoming zero during the initial steps, adjustment needs to be made:
        
          <p style="text-align: center;">$\hat{m_{t}}=\frac{m_{t}}{1-\beta_{1}^{t-1}}$,</p>
          <p style="text-align: center;">$\hat{v_{t}}=\frac{v_{t}}{1-\beta_{2}^{t-1}}$,</p>
        * The final update is:
            <p style="text-align: center;">$w_{t+1}=w_{t}-\frac{\alpha}{\sqrt{\hat{v_{t}}}+\epsilon}*\hat{m_{t}}$</p>
        exponentially averaging past gradient and adjust by exponentially averaging past gradient squared
        * Adam optimization converges very fast 
   
    * Weight decaying 
        * When adding L2 regularization to vanilla SGD:
        
        <p style="text-align: center;">$\tilde{E(\textbf{w})}=E(\textbf{w})+\frac{\lambda}{2}\textbf{w}^2$</p>

        it is equivalent of changing the weight update rule to below:
        <p style="text-align: center;">$w_{t+1}=w_{t}-\alpha*\frac{\partial E}{\partial w}-\alpha*\lambda*w_{t}$</p>
        * But when adding L2 regularization to momentum and other optimization methods, the resulting weight update is not necessarily weight decaying
        * Decoupling weight decay: adding the term $\alpha*\lambda*w_{t}$
        
            * Add weight decay to momentum:
            <p style="text-align: center;">$w_{t+1}=w_{t}-\alpha*v^{t}_{dw}-\alpha*\lambda*w_{t}$</p>

            * Add weight decay to Adam:
            <p style="text-align: center;">$w_{t+1}=w_{t}-\frac{\alpha}{\sqrt{\hat{v_{t}}}+\epsilon}*\hat{m_{t}}-\alpha*\lambda*w_{t}$</p>

        * Adding weight decay to Adam is as powerful as SGD with momentum
        
    * AMSGrad: fix for short comings of Adam:
    
        * in Adam short term memory of the gradients becomes an obstacle. Adam can converge to suboptimal. Although there are minibatches where the gradient calculation yields large and informative gradient, those are area and averaged out. 
        
        * instead of using $\hat{v_{t}}$ we use the previous one if it is larger than current:
        <p style="text-align: center;">$\hat{v_{t}} = max(\hat{v_{t-1}},v_{t})$</p>  

        * The full AMSGrad update without bias correction is:
            <p style="text-align: center;">$m_{t}=\beta_{1}m_{t-1}+(1-\beta)*dg$,</p>
            <p style="text-align: center;">$v_{t}=\beta_{2}v_{t-1}+(1-\beta)*dg^2$,</p>
            <p style="text-align: center;">$\hat{v_{t}} = max(\hat{v_{t-1}},v_{t})$</p> 
            <p style="text-align: center;">$w_{t+1}=w_{t}-\frac{\alpha}{\sqrt{\hat{v_{t}}}+\epsilon}*m_{t}$</p>
        * It is debatable whether AMSGrad consistently outperforms Adam across data sets 
        
* Section 4 Hyper parameter tuning and learning rate schedule: choose the set of hyper parameters to optimize the validation set metrics 
        *  learning rate too small, very slow ; too big, never learn
    ![learning-rate](/Users/qxy001/Documents/personal_src/aiml/notes/learning_rate.png). Good learning rate (0.01, 0.001, $10e^{-4}$)
        * Step decay: decay every n epoch, with decay factor:
        <code>
        <p style="text-align: center;">$decay factor=decay rate^{fraction}$</p>
        <p style="text-align: center;">$fraction = (current_epoch-start_decy_epoch//n)$</p>
        <p style="text-align: center;">$/alpha = decay factor$</p>
        </code>>
        * learning rate and batch size:
            - when using larger batch, means gradient calculation has less noise more confident. Therefore the learning rate can be big. $\sqrt{k}$
 
* Section 5 weight initialization 
    * Zero weights intialization: 
        * <code>w[l]=np.zeros(layer_size[l], layer_size[l-1])</code>
        * no learning 
    * Same value initialization: 
        * Every neuron is updated the same. Symmetry breaking problem 
    * Normal Initilization: helps to break the symmetry 
        * <code>w[l]=np.random.randn(layer_size[l], layer_size[l-1])*0.01</code>
        * <code>np.random.randn(d0,d1,d2..) returns a sample from standard normal distribution with dimension[d0,d1,d2...]</code>
    * Xavier Initialization: 
        * keep variance among layers constant: $Var(w[l])=Var(w[l-1])$
        * <code>w[l]=np.random.randn(layer_size[l], layer_size[l-1])*np.sqrt(1/layer_size[l-1])</code>

* Regularization and normalization 
    * How to reduce overfitting:
        - train using more data 
        - data augmentation 
        - use early stopping, when validation error starts to increase 
    
    * Regularization:
        - L1 regularization. Loss function: $\sum_{i=1}^{n}(y_{i}-\sum_{j=1}^{p}x_{ij}w_{j})^2+\lambda\sum_{j=1}^{p}|w_{j}|$ 
        - L2 regularization. Loss function: $\sum_{i=1}^{n}(y_{i}-\sum_{j=1}^{p}x_{ij}w_{j})^2+\lambda\sum_{j=1}^{p}(w_{j})^2$ 
    
    * Dropout:
        - during training, values of neurons tend to heavily dependently on eac other. (Or simply because the parameters are too much ) The model might just memorize the nuances in the train data set resulted in overfiting. 
        - dropout forces neuros to learn independent and robust features that are useful in conjunction to many random substates of the other neurons. 
        - In the training phase, for a specific hidden layer, for each training sample, for each iteration, **randomly disable a fraction, p, of neuros and its activations**. In testing phase, do the inference without drop out. 
    * DropConnect: Weight dropout 
        - instead of deactivating neurons, set the weights to be zero 

    * Normalization and standarization
        - (similar as linear regression and logistic regression) Without normalization, large values will dominate the data sets and small values are meaningless. Gradient exploding can happen as well. 
        
        - min-max normalization. standarization to put input: $X~N(0,std)$
    * Batch normalization:
        - In addition to normalize/standarize input data set around its mean and std, also normalize the output of neurons in each hidden layer across a mini batch of data
        - Suppose the minibatch input has three samples, $x_1$,$x_2$,$x_3$. For a hidden neuron $z_{i}^{j}$ in jth layer, it will be activated three times $\sigma(z_{i1}^{j})$, $\sigma(z_{i2}^{j})$,$\sigma(z_{i3}^{j})$. Normalize those three values. So no dominating large value neurons in the network. Formally, if a mini batch has m samples, the mean of ${\sigma(z_{ik}^{j})}_{k=1}^{m}$ is:
            <p style="text-align: center;">$u_B=\frac{1}{m}\sum_{k=1}^{m}\sigma(z_{ik}^{j})$</p>
        The standard deviation of the mini batchis: 
        <p style="text-align: center;">$\sigma_{B}=\sqrt{\sum_{k=1}^{m}(\sigma(z_{ik}^{j}-u_B)^2}$</p>
        The new activation is:
        <p style="text-align: center;">$\hat{\sigma(z_{ik}^{j})}=\frac{\sigma(z_{ik}^{j})-u_B}{\sigma_B}$</p>
        The output can be scaled and shift using $\gamma$,$\beta$:
            <p style="text-align: center;">$y_ik=\gamma\hat{\sigma(z_{i}^{j})}+\beta$</p>
            
        - Benefits:
            + converge faster and reduce the need for drop out  
            + can normalize to what ever distribution that is best for the learning
            + eliminate the need for bias for a layer because batch normalization already shifts output by beta. $\gamma$,$\beta$ can be learnt 
    
    * Layer normalization: 
        - instead of mini batch, the mean and std are calculated across m neurons in a hidden layer. 
        - Difference vs batch normalization:
            + In batch normalization, the input to the same neurons are normalized across different data points ( e.g. image) in a mini batch 
            + In layer normalization, the input to the neurons of the same hidden layers are normalized, even in forward propogation of a single data point. 
    
- [Some other perspectives on neural networks](http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture05-rnnlm.pdf) 
    + Drop out: 
        + This prevents feature co-adaptation: A feature cannot only be useful in the presence of particular other features
        + In a single layer: A kind of middle-ground between Naïve Bayes (where all feature weights are set independently) and logistic regression models (where weights are set in the context of all others)
        + Can be thought of as a form of model bagging (i.e., like an ensemble model)
        + Nowadays usually thought of as strong, feature-dependent regularizer 
    + “Vectorization”
        * <code>timeit [W.dot(wordvectors_list[i]) for i in range(N)]</code>
        * <code>timeit W.dot(wordvectors_one_matrix)</code>. Much faster, especially implement in GPUs 
    + ![Activation](/Users/qxy001/Documents/personal_src/aiml/notes/activations-2.png)
    + 

#### Common [Evaluation Metrics](https://cs230.stanford.edu/section/7/) in ML/DL
* Object detection:
    * For one object in the image, IoU = intersection area / union, larger than threshold; object class is correct. We deem positive. 
    * Precision and Recall can be calculated from all objects in an image
    * mAP (mean average precision) and mAR (mean average recall)
* NLP: (TBD)

#### [Computer Vision](https://cs231n.github.io/)
* Neural Network architectures 
    -  Neural Networks are modeled as collections of neurons that are connected in an acyclic graph.Instead of an amorphous blobs of connected neurons, Neural Network models are often organized into distinct layers of neurons. For regular neural networks, the most common layer type is the *fully-connected layer*
    -  Single-layer neural network.No hidden layers. Usually called "ANN" or "multi-layer perceptron (MLP)"
    -  Output layer vs last layer. Output layer usually dont have activation, e.g. softmax layer 
    
* ConvNets:
    - if an image is wxhxd, having so many neurons in each layer of a fully connected network will dramatically increase the parameters. 
    - ConvNets use 3D volume inputs, every layer neurons are arranged in wxhxd 3D fashion, where d is the activation. Only a small portion of one layers neurons are connected to the previous layer.Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.
![Conv-nets](/Users/qxy001/Documents/personal_src/aiml/notes/conv-nets.png)
    - the architecture: INPUT - CONV - RELU - POOL - FC
        + INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
        + CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters. Depending on the convolution window size, the resulted volume can be nxnx12, n<32.
        + RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
        + POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
        + FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume. 
        
    - Convolution Layer.
        + Accepts a volume of size W1×H1×D1
        + Requires four hyperparameters:
            + Number of filters K. Each one is FxFxD1 so at depth direction weight is full. K is also the output channel. 
            + their spatial extent F, therefore there are FxFxD1xK weight parameters. 
            + the stride S, step size. How far in pixel the convolution window moves
            + the amount of zero padding P. So the size of local region matches the filter size 
            + Produces a volume of size W2×H2×D2 where: W2=(W1−F+2P)/S+1; H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry); D2=K
With parameter sharing, it *introduces F⋅F⋅D1 weights per filter*, for a total of *(F⋅F⋅D1)⋅K weights and K biases*.
In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.
    - Pooling Layer. Accepts a volume of size W1×H1×D1. Requires two hyperparameters:
        their spatial extent F,
        the stride S,
    Produces a volume of size W2×H2×D2 where:
        W2=(W1−F)/S+1
        H2=(H1−F)/S+1
        D2=D1
        Introduces zero parameters since it computes a fixed function of the input. For Pooling layers, it is not common to pad the input using zero-padding.    
    - FC layer. Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. 
    
    - Layper pattern:
    <code>INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC </code>

    - VGG: increased depth compared to AlexNet. Also included some materials from [here](https://medium.com/towards-artificial-intelligence/the-architecture-and-implementation-of-vgg-16-b050e5a5920b)
        + similarity: small convolution filter , small stride size
        + difference: more depth
        + The convolutional layers in VGG use a very small receptive field (3x3, the smallest possible size that still captures left/right and up/down). There are also 1x1 convolution filters which act as a linear transformation of the input, which is followed by a ReLU unit. The small-size convolution filters allows VGG to have a large number of weight layers; of course, more layers leads to improved performance.
        + The max pooling layer: It is performed over a max-pool window of size 2 x 2 with stride equals to 2, which means here max pool windows are non-overlapping windows.
        + The fully connected layer. three of fully connected layers
        + VGG has different configurations from VGG11-VGG19. See below table. layers are represented as <code>filter size-number of channel</code>. Number of channel in convolution layer is the number of filter. 
        ![vgg-configuration.png](/Users/qxy001/Documents/personal_src/aiml/notes/vgg-configuration.png)

    - VGG-16. above table configuration c, 13 convolution layers + 3 fc. See below graph. 
![vgg-16.png](/Users/qxy001/Documents/personal_src/aiml/notes/vgg-16-visualization.png)
        + 2 continous block of 2 layers of convnets, plus three block of 3 layers of convnets, max pooling serving as down sampling between blocks. Size of wxh decreases and channel number increases. 
    
    - ResNet: 
        + movtivation is with increased number of layers in deep nets, training error tends to increase. This is called the degradation problem. With the network depth increasing the accuracy saturates(the networks learns everything before reaching the final layer) and then begins to *degrade* rapidly if more layers are introduced. Suppose a deep net with n layers learns all the transformation, the accuracy saturates. Adding another m layers do not add any more value, and the optimal mapping the m layers can learn is identity mapping. Anything else learnt can be detrimental to the results. 
        + Fix of degradaption. Instead of learning a mapping H(x),  learning F(x)=H(x)-x and add a direction flow of x to the output of the mapping. H(x)=F(x)+x
![residual-block.png](/Users/qxy001/Documents/personal_src/aiml/notes/residual-block.png)
    
        If the optimal solution is the identity function, it is easier to push F(x) to be zero. If additional layers were useful, even with the presence of regularisation, the weights or kernels of the layers will be non-zero and model performance could increase slightly.
        + *Skip layer*: The approach of adding a shortcut or a skip connection that allows information to flow, well just say, more easily from one layer to the next’s next layer, i.e., you bypass data along with normal CNN flow from one layer to the next layer after the immediate next.
        + Two residual blocks: 
            * identity block: when the input activation is at same dimension of output activation:
            ![resnet-identity-block](/Users/qxy001/Documents/personal_src/aiml/notes/resnet-identity-block.png)
            
            * Convolution block: when the input activation is with different dimension of output activation: 
            ![resnet-conv-block](/Users/qxy001/Documents/personal_src/aiml/notes/resnet-conv-block.png)


        + Success of ResNet:
            * Won 1st place in the ILSVRC 2015 classification competition with top-5 error rate of 3.57% (An ensemble model)
            * Won the 1st place in ILSVRC and COCO 2015 competition in ImageNet Detection, ImageNet localization, Coco detection and Coco segmentation.
            * Replacing VGG-16 layers in Faster R-CNN with ResNet-101. They observed a relative improvements of 28%
        
        Using VGG architecture plain 34 layers, training error is higher than 18 layers. But if using ResNet, 34 layers still show a decrease in training error than 18 layers 
        ![resnet-18-34-compare](/Users/qxy001/Documents/personal_src/aiml/notes/resnet-18-34-compare.png) 
    
    - Fine tunning a CNN:
        + using CNN as a feature extractor. Use the FC ReLud value before the last classification layer. 
        + using new data sets to continue back propogation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. 
        
        + Pretrained model. Morden deep CNN models took 2-3 weeks to train, it is common for people to release check points of trained weights for other to continue
        + Strategies:
            * *New data set is large and very similar to the original data set*. Since we have more data, we can have more confidence that we won’t overfit if we were to try to fine-tune through the full network.
            * *New data set is small and very similar to the original data set* Keep doing back propogation may be overfitting. Extract higher level of features produced by CNN to train classifier. 
            
            * New data set is large and different to the original data set 
            It is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

            * New data set is small and different to the original data set.it might work better to train the SVM classifier from activations somewhere earlier in the network. 

* Object detection - some prep work 
    - Image gradient vector:
        + If an image can be represented as a function of pixels arranged in a 2D grids, $f(x,y)$. Its point-wise gradient can be calculated as:
        
             <p style="text-align: center;">$\begin{bmatrix}\textbf{g_x}\\\textbf{g_y}\end{bmatrix}=\begin{bmatrix} f(x+1,y)-f(x-1,y)\\ f(x,y+1)-f(x,y-1) \end{bmatrix}$</p>

        It is expensive to calculate gradient this way for every pixel. A bi directional convolution filter can achieve the same thing i.e.
            <p style="text-align: center;">$G_{x}=[-1,0,1]*[x-1,x,x+1]$,$G_{y}=[-1,0,1]*[y-1,y,y+1]$</p>
        + Magnitue of the gradients: $\sqrt{(g_x)^2+(g_y)^2}$
        + Direction of the gradients: $arctan(\frac{g_x}{g_y})$
    - Histogram of Gradient (HOG):
        + Preprocessing the image, substracting mean and divide the std
        + Calculate gradient at each (x,y)
        + Divide the image into 8x8 cells.
    
* [Mask R-CNN, fast R-CNN, faster R-CNN, Yolo and beyond](/Users/qxy001/Documents/personal_src/aiml/notes/object-detection.md) 

* Fine tuning, few shot learning, meta learning - DRAGON

* Table Detection - Harvest

#### NLP
* Stanford cs240N
* Word2Vec/GloveEmbedding
* Transformer Stack
* BERT model (generation, summarization, Q&A), different types of attention 
* LayoutLM - DRAGON 

#### Active Learning, Semi supervised learning, HITL
* 

#### Neural Nets in Personalization and CLR prediction
* imbalanced data sets 
* Cold start 
* 

#### Cross modality 


#### [Making decisions under uncertainty](https://docs.google.com/spreadsheets/d/1r_lTkVdEM89JUNuKbeWSULg0yMDcl_dWS_MDiG0KFm4/edit#gid=0)


#### Knowledge Graph and Common Sense Reasoning 

#### Other adavanced topic:
[Reasoning in AI](https://towardsdatascience.com/what-is-reasoning-526103fd217)
* https://papers.nips.cc/paper/2019/file/9c19a2aa1d84e04b0bd4bc888792bd1e-Paper.pdf

#### [Intro to Semi supervise learning](https://www.cs.ubc.ca/~schmidtm/Courses/LecturesOnML/semiSupervised.pdf)
* common situation: 
    - a small amount of labeled data set $(x_i,y_i)$
    - a large amount of unlabeled data set $(x_t)$ 
    - transductive Semi-Supervised Learning: Only interested in labels of the given unlabeled examples.
    - Inductive Semi-Supervised Learning: Interested in the test set performance on new examples. 
    - If unlabeled data set $(x_t)$ is totally random, nothing can be said about y. SSL is not possible 
    - SSL is only possible when $(x_t)$ contains some information to tell about y. e.g. clusters or manifold in unlabeled data set 
    
* SSL Approach 1: self-taught learning:
    - step 1. Train on labeled data set. model = fit($x_i$,$y_i$)
    - step 2. Guess label for ${x_t}$. $\hat{y}=model.predict(x_t)$
    - step 3. Train on bigger data set: model = fit($[x_i,x_t]^T$,$[y_i,\hat{y}]^T,\lambda$)
    - Go back to step 2 and repeat
    - Potential problems:
        + step 2 and step 3 can reinforce errors or even diverge 
        + Regularize the loss from the unlabeled examples:
        $f(w)=\frac{1}{2}\parallel X_w-Y \parallel^2+\frac{\lambda}{2}\mid X_t-\hat{Y} \mid$,$\lambda$ controls how much loss from unlabeled data is from

* SSL Approach 2: Co Training
* SSL Approach 3: Entropy Regularization 
* SSL Approach 4: Graph-based SSL (Label Propogation):
    - treat unknown labels $\bar{y}$ as varaibles, and minimize cost disagreement:
<p style="text-align: center;">$f(\bar{y})=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{t}w_{ij}(y_i-\bar{y_j})^2+\frac{1}{2}\sum_{i=1}^{t}\sum_{j=1}^{t}\bar{w_{ij}}(\bar{y_i}-\bar{y_j})^2$</p>
where $w_{ij}$ are weights between labeled and unlabeled samples, and $\bar{w_{ij}}$ are weights between unlabeled samples. Doing gradient on labels of unlabeled samples. Term $(y_i-\bar{y_j})$ makes unlabeled sample $\bar{y_j}$ similar to labeled neighbors; term $(\bar{y_i}-\bar{y_j})$ makes unlabeled sample in the same neighborhood agree each other 
    - comments on graph based ssl:
        + transductive method, only estimate unknown labels 
        + surprisingly effective even if you only have a few labels 
        + Does not need features if your features are reflective on the weighted graph 



    



