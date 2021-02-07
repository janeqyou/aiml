#### Human in the loop 
#### HumanGAN: generative adversarial network with human-based discriminator and its evaluation in speech perception modeling
#### abstract 
- A basic GAN composes of two deep networks, the *generator*, and the *discriminator*. 
- The purpose of training a generator is to represent a real-data distribution
- This is achieved by fooling the discriminator that distinguishes real and generated data. Therefore, the basic GAN cannot represent the outside of a real-data distribution. Or in other words the GAN can only be as good as the supervision provided to the discriminator. 

- In the case of speech perception, humans can recognize
not only human voices but also processed (i.e., a non-existent human) voices as human voice. *Such a human-acceptable distribution
is typically wider than a real-data one and cannot be modeled by
the basic GAN.* i.e. Human is a powerful discriminator. This is the human in the loop perspective that really exploits 

- We formulate a backpropagation-based generator training algorithm by
regarding human perception as a black-boxed discriminator. The
training efficiently iterates generator training by using a computer
and discrimination by crowdsourcing i.e. a group of people 

#### Introduction 

- “deep generative models,” can model a complicated data distribution. A generative adversarial network (GAN)is one of the most promising approaches in learning deep generative models.
- the basic GAN only represents real data distribution. 
- HumanGAN trains generator by fooling crowdworkers’ perceptual evaluation (i.e., humanbased discriminator), and generator finally represents humans’ perception distribution.
![human-gan](/Users/qxy001/Documents/personal_src/aiml/notes/human-gan.png)

- Evaluate the HumanGAN in the modeling of speech naturalness,
i.e., to what degree can humans accept synthesized speech as human
voice.

#### Basic GAN
- Basic formulation of GAN and training:
    + input real data: $x=[x_1,x_2,...x_n]$. 
    + Generator $G(\cdot)$ and discriminator Generator $D(\cdot)$.
    + $G(z)=\hat{x}=[\hat{x_1},\hat{x_2},\hat{x_3},...\hat{x_n}]$, where $z$ is prior noise with a fixed distribution e.g. uniform distribution, $z=[z_1,z_2,...z_n]$
    + Both $x$ and $\hat{x}$ are input to $D$, which outputs a posterior $p(x|data is real)$ that the input is real.
    + The objective function of GAN is:
    <p style="text-align: center;">$V(D,G)=\sum_{i=1}^{N}\log{D(x)} + \sum_{i=1}^{N}\log{1-D(G(z))}$</p>. 
    + Specifically for the discriminator part, the model pramaters of the discriminator $D(\cdot)$ $\theta_{D}$ can be estimated at:
    <p style="text-align: center;">$\theta_D =\underset{\theta_D}\arg\max\sum_{n=1}^{N}(\log{D(x)})$(1)</p>
    Equation (1) tries to maximize the posterior produced by D, i.e. trains a good discriminator on real data vs fake data 

    + For the generator $G(\cdot)$, the model parameter $\theta_{G}$ can be estimated by:

    <p style="text-align: center;">$\theta_G =\underset{\theta_G}\arg\min\sum_{n=1}^{N}(\log{1-D(G(z))})$</p>. i.e. push $D(G(z))$ to as big as possible. When $D(G(z))$ is a large value, it means even fake input $G(z)$ has a big value when evaluated by $D(\cdot)$. i.e. the discriminator is fooled. 

    + The final min max game of GAN:
    
    <p style="text-align: center;">$\underset{\theta_G}\arg\min\underset{\theta_D}\arg\max\sum_{n=1}^{N}(\log{D(x)}+\sum_{n=1}^{N}(\log{1-D(G(z))})$</p>
    
    

#### HumanGAN
- $D(\cdot)$ was replaced by human percention, outputs posterior distribution about “to what degree is the input perceptually acceptable” from 0 to 1. So there is no need to train $D(\cdot)$, the objective is training a generator that generates data distribution percetually acceptable as much as possible or being able to fool human as much as possible:

    $\underset{\theta_{G}}\arg\max V(D,G)=\sum_{i=1}^{N}(D(G(z)))$

- Training the generator by maximize the above value. Proposing an interative algorithm:
<p style="text-align: center;">$\theta_G \gets \theta_{G}+\alpha\frac{\partial V(G,D)}{\partial \theta_G}$</p>.
- Calculate the partial derivatives 
    - Usually $\frac{\partial V(G,D)}{\partial \theta_G}=\frac{\partial V(G,D)}{\partial \hat{x}}\frac{\partial \hat{x}}{\partial \theta_G}$. But in humanGAN, $D(\cdot)$ is not differentiable w.r.t $\hat{x}$.
    - To calcuate the partial derivatives, propose a training algorithm
that uses natural evolution strategies (NES) that approximates
gradients by using data perturbations. Natural evolution strategies (NES) are a family of numerical optimization algorithms for black box problems. Similar in spirit to evolution strategies, they iteratively update the (continuous) parameters of a search distribution by following the natural gradient towards higher expected fitness.

- Algorithm:
For i from 1 ot N: // for each fake/generated data sample $\hat{x_i}$ 
    + for r from 1 to R: // within each sample, repeat this R times 
        + generate a small pertubation $\Delta x_{i}^{(r)}$ from a multi variate Gaussian distribution $N(0,(\sigma)^2I)$. 
        + present human with two perturbed data point ${\hat{x_{i}}+\Delta x_{i}^{(r)},\hat{x_{i}}-\Delta x_{i}^{(r)}}$  
        + let human evaluate $\Delta D(x_{i}^{(r)}) = D(\hat{x_{i}}+\Delta x_{i}^{(r)})-D(\hat{x_{i}}-\Delta x_{i}^{(r)})$

        + $\Delta D(x_{i}^{(r)})=\begin{cases} 1 & \hat{x_{i}}+\Delta x_{i}^{(r)} is more acceptable \\ -1 & otherwise \end{cases}$ 

    + Finally $\frac{V(G,D)}{\hat{x_i}}=\frac{1}{2\sigma R}\sum_{r=1}^{R}D(x_{i}^{(r)})\Delta x_{i}^{(r)}$


#### Practical Limitation
- The value of $\sigma$. When the $\sigma$ is too small, human can not notice the difference which leads to vanishing gradient; when the $\sigma$ is too large, the gradient is not accurate. Experimenting with $\sigma$
- The time and cost involving human. Training this system needs queries/evaluations (number of training iterations x data samples N x number of perturbation R ). Experiments use small and simple neural network for generator, small number of data points and low-dimensional data to limit the cost 
- Another vanishing gradient remedy. When discriminator is too good i.e. the generated data is too far away from human accepted or perceptually accepted data, the generator could face vanishing gradient problem. So they use real data to intialize the generator.  (TLDR: For humans’ perceptual evaluation, a speech waveform was synthesized from the generated speech features. First, the first and second principal components were generated from a generator and de-normalized. The remaining speech features, i.e., the third and
upper principal components, the fundamental frequency, and aperiodicity components, were copied from one frame of an averagevoice speaker. Average-voice speaker means a speaker whose her
first and second principal components are the closest to 0. These
correspond to speech features of one frame (5 ms). To make the perceptual evaluation easy for crowdworkers,we copied the features for
200 frames and synthesized 1 s of a speech waveform by using the
WORLD vocoder)

#### Experiment set up 
- The speech data used was reading style speech from 199 female speakers. The speach features are 513 dimensional log spectral envelopes extracted every 5ms. Can perform PCA and extracted first 2 principle components 
- Human perceptual distribution covers a wider distribution than real data. In the following figure, color map shows the posterial distribution of "is the input perceptually acceptable". Bright yellow shows regions of accepted area. It covers more than real data. 
![human-gan-perceptual](/Users/qxy001/Documents/personal_src/aiml/notes/human-gan-perceptual-distribution.png)

- The generator was a small feed-forward
neural network consisting of a two-unit input layer, 2 × 4-unit sigmoid hidden layers, and two-unit linear output layer. The model parameters were randomly initialized, but we iterated the random initialization until the initial generator output data that covered ranges of a higher posterior probability. Below figure shows generated data (white points) overlays on posterior probablitly heat map. One can see white points start out randomly and after a few itertions it concentrates on high posterior probability area, and covers more than real data 

![human-gan-iterations](/Users/qxy001/Documents/personal_src/aiml/notes/human-gan-iterations.png)





#### Resources:
- J.-J. Zhu and J. Bento, “Generative adversarial active learning,”
in arXiv preprint arXiv:1702.07956, 2017.
- Y. Deng, K. Chen, Y. Shen, and H. Jin, “Adversarial active
learning for sequences labeling and generation,” in Proc. IJCAI, Stockholm, Sweden, Jul. 2018, pp. 4012–4018.











