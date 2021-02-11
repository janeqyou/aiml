### NLP Stanford cs224N
#### [Word2Vec]()
- Linguistics represent words as discrete symbols. For example, motel=[00001000..] and hotel=[001000...]. There is no similarity between the two vectors, although those two words are similar 
- *Distributional semantics*: A word’s meaning is given
by the words that frequently appear close-by
    + When a word w appears in a text, its context is the set of words
that appear nearby (within a fixed-size window).
    + Use the many contexts of w to build up a representation of w
- Word2Vec:
    + We have a large corpus (“body”) of text
    + Every word in a fixed vocabulary is represented by a vectors
        + Go through each position t in the text, which has a center word c and context(“outside”) words 
        + Use the similarity of the word vectors for c and o to calculate the probability of o given c i.e. P(o|c) - Skip Gram. or calculate the probabiliyt of c given context words {o}. - CBOW
        + Keep adjusting the word vectors to maximize this probability.In the case of *Skip-Gram*, this is equivalent of for each position $t=1,...,T$ predict context words within a window of fixed size $m$, given center word $w_j$, maximize the following likelihood:
        $L(\theta)=\prod_{t=1}^{T}\prod_{-m<=j<=m}P(w_{t+j}|w_{t},\theta)$, 
        and $P(w_{t+j}|w_{t})=\frac{exp(w_{t+j}^T\cdot w_{t})}{\sum_{u \in V}exp(u^T \cdot w_{t})}$
        + $\theta$ represents all the model parameters, in one long vector 
        with d-dimensional vectors and V-many words. The dimension of $\theta$ is $R^2dV$. Can use stochastic gradient descent by moving windows through text to maximize the likelihood
    ![skip gram visual](/Users/qxy001/Documents/personal_src/aiml/notes/skip-gram-visual.png)

    + (TODO) Noise Contrastive Estimation 
    + (TODO) Negative sampling s
- CBOW (contextual bag of words)
![cbow](/Users/qxy001/Documents/personal_src/aiml/notes/cbow-visual.png)

- Neural Dependency Parsing: Google probably has the best - [SyntaxNet](https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html)
- Language Models
    + What is a language model: a system assigns probability to a piece of text. For example we have some text $x^{(1)},x^{(2)},...,x^{(t)}$, then the probability of this text by LM is: $P(x^(1),x^(2),...,x^(t)) = P(x^(1))xP(x^(2)|x^(1))x...P(x^(T)|x^(T-1),x^(T-2),x^(T-3),...x^(1)) = \prod_{t=1}^{T}P(x^(t)|x^(t-1),x^(t-2),x^(t-3),...x^(1))$
    + *N-gram language model*: estimate $P(x^(1),x^(2),...,x^(t))$ from $P(x^(1),x^(2),...,x^(t-1))$.e.g. counting them in some large text corpus for statistical approximation:
    <p style="text-align: center;">$\frac{count(P(x^(1),x^(2),...,x^(t-1)))}{count(P(x^(1),x^(2),...,x^(t)))}$</p>
        * Consider 4-gram language model, "student opened their __" (book, exam, etc):
        <p style="text-align: center;">$\frac{count(P(students opened their \omega)}{count(P(students opened their)}$</p>
        * *sparsity problem*: what if $students opened their \omega$ never occurred in the corpus ? Adding a small non zero term, smooth 
        * *back off*: what if $students opened their$ never occurred in the corpus ? use $P(opened their)$
        * *memory in efficient*: need to store all 4-gram in the corpus 
        * Grammatically correct but not necessary semantically meaninful. if increasing n to factor in more context, sparsity and memory requirements will get worse 
    + Consider below a fixed window language model:
    ![fix window nn lm](/Users/qxy001/Documents/personal_src/aiml/notes/fix-window-nn-lm.png)

        * Improvements over n grams 
        * Do not need to store all n grams
        * different weights are applied to each word inputs
        * Fixed window size is usually to small. Windows size too large can cause detrimental effect 
        
    + Simple RNN Language Model (sequence to sequence)
    
        * core idea: apply the same weights $W_h$ and $W_e$ again and again. Can handle variable lengths of text sequences.  
        * get a large corpus of text which is a sequence of words $x^{(1)},x^{(2)},...,x^{(t)}$ and feeds language model to generate $y^{(1)},y^{(2)},...,y^{(t)}$. At each time step t, the loss function of lm is: $J^{(t)}(\theta) = CE(y^{(t)},\hat y^{(t)})=-\sum_{\omega \in V}y_{\omega}^{(t+1)}\log\hat{y}_{\omega}^{(t+1)}$. $y^{(t)}$ is in fact the next token in the sequence $y_{x_{t+1}}^{(t)}$, and the probability is 1. For all the other $\omega$, $y_{\omega}^{(t)}$ is zero. Therefore at time $t$, the loss function is $J^{(t)}_{(\theta)} = - \log\hat{y}_{x_{(t+1)}}$, the negative log likelihood of predicting the next token/word in sequence. The total error function of $T$ steps is:
        <p style="text-align: center;">$J(\theta)=-\sum_{t=1}^{T}J^{(t)}(\theta)=\frac{1}{T}\sum_{t=1}^{T} -\log\hat{y}_{x_{(t+1)}}$</p>
        * computing loss and gradient across $x^{(1)},x^{(2)},...,x^{(t)}$ is too expensive, calculate loss for batches of sentences instead 
        * Back propogation over time: $\frac{\partial J^{(t)}}{\partial W_h}=\sum_{i=1}^{t}\frac{\partial J^{(t)}}{\partial W_h}|_{i}$
        * Evaluation of language model, perplexity i.e. exponential value of the cross entropy loss $J(\theta)$: $exp(J(\theta))$. the lower the better. 
    + RNN can also be used as sentence encoding module. 
        * In the sentiment classification task, the final hidden state can be used as sentence encoding
        * Or the element-wise max or mean of all hidden states can be the the input sentence encoding module 
    + Vanishing Gradient
![vanishing gradient](/Users/qxy001/Documents/personal_src/aiml/notes/vanishing-gradient-intuition.png)
    
Mathematically, the gradient of loss $J^{(i)}(\theta)$ on step $i$, with respect to the hidden state $h^{(j)}$ at some previous step, $i>j$, $l=i-j$
is: 
<p style="text-align: center;">$\frac{\partial J^{(i)}{(\theta)}}{\partial h^{(j)}}=\frac{\partial J^{(i)}{(\theta)}}{\partial h^{(i)}}\prod_{(j<t<i)}W_t=\frac{\partial J^{(i)}{(\theta)}}{\partial h^{(i)}}W^{l}$</p>
When $W^l$ is really small, this term almost becomes zero, gradient vanishes. loss at step $i$ can not influence many steps back. long term dependencies become unreliable. 
    + Exploding Gradient. When the gradient becomes to large usually because having take a bad step and stuck in a bad parameter configuration: $\theta^{new}=\theta^{old}-\alpha \nabla J(\theta)$

- Bi directional RNN 
![bi directional RNN intuition](/Users/qxy001/Documents/personal_src/aiml/notes/bidirectional-rnn-intuition.png)

    * on time step t, $\overrightarrow{h^{(t)}}=\underset{FW}RNN(\overrightarrow{h^{(t-1)},x^{(t)}})$, $\overleftarrow{h^{(t)}}=\underset{BW}RNN(\overleftarrow{h^{(t+1)},x^{(t)}})$, $h^{(t)}=[\overrightarrow{h^{(t)}}, \overleftarrow{h^{(t)}}]$
    * Bidirectional RNNs are only applicable if you have access to the entire input sequence.They are not applicable to Language Modeling, because in LM you only have left context available.
    *  BERT (Bidirectional Encoder Representations from Transformers) is a
powerful pretrained contextual representation system built on bidirectionality
    * Use bidirectionality when possible 
         
- Multi-layer RNN
    + Multiple RNN can be stacked together to allows the network to compute more complex representations. The lower RNNs should compute lower-level features and the higher RNNs should compute higher-level features.
    + The hidden states from RNN layer i are the inputs to RNN layer i+1
    + High performing RNN are usually multi layer. Machine Translation usually has 2-4 layers RNN for encoding and 4 layers for decoding. *Transformer* can have up to 12-24 layers. 
    + Vanishing gradient in "deep" RNNs can be alleviated by  adding "skip connections"
    
- [GloVe embedding](https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6) 
    + Global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus.
    + Define a few terms: 
        <p style="text-align: center;">$X_{ij}$ tabulate the number of word j occuring in context of word i;</p>
        <p style="text-align: center;">$X_{i}=\sum_{k}X_{ik}$ occurrence of word i marginalize over all other words</p>
        <p style="text-align: center;">$P_{ij}=P(j|i)=X_{ij}/X_{i}$ occurrence of word i marginalize over all other words</p>
        <p style="text-align: center;">$F(w_i,w_j,\hat{w}_k)=\frac{P_{ij}}{P_{ik}}$ $w_i$ and $w_j$ are word vectors, $\hat{w}_k$ is the prob word.This formulation specifies the ratio between <i,j> and <i,k></p>
    To enforce 1) linearity 2) symmetry 
        <p style="text-align: center;">$F((w_i-w_j)^T,\hat{w}_k)=\frac{P_{ij}}{P_{ik}}$ and $F((w_i-w_j)^T,\hat{w}_k)=\frac{F((w_i)^T\hat{w}_k)}{F((w_i)^T(w_j)}$
    Therefore, $F((w_i)^T\hat{w}_k))=P_{ik}=\frac{X_{ik}}{X_i}$ and we can choose $F(x)=exp(x)$ as functional form, and 
        <p style="text-align: center;">$((w_i)^T\hat{w}_k)=log(P_{ik})=log(X_{ik})-log(X_i)$</p>
    and move around a few terms and absorb $log(X_i)$, we arrive at 
        <p style="text-align: center;">$((w_i)^T\hat{w}_k)+b_i+\hat(b)_k=log(X_{ik})$</p>
    Glove is designed to enfoce word vector dot products:
        <p style="text-align: center;">$J=\sum_{i,j=1}^{V}f(X_{ij})((w_i)^T\hat{w}_k)+b_i+\hat(b)_j-log(X_{ij}))^2$</p> and $f(x)$ is a weight factor such that if word occurrances are less than max, the weight is less than 1, and can be parameterized 

- [Universal Sentence Encoding](https://amitness.com/2020/06/universal-sentence-encoder/) 
    + Goal is to learn a model that can map a sentence to a fixed-length vector representation
    + Short comings of averaging word embeddings are loss of information and loss or order 
    + The idea is to design an encoder that summarizes any given sentence to a 512-dimensional sentence embedding. We use this same embedding to solve multiple tasks and based on the mistakes it makes on those, we update the sentence embedding 
    + Two variants of underlying neural network architectures:
        * Transformer Encoder 
    ![use-transformer](/Users/qxy001/Documents/personal_src/aiml/notes/use-transformer.png)   
        * Deep Averaging Network 
    ![use-dan](/Users/qxy001/Documents/personal_src/aiml/notes/use-dan.png)
    + Three tasks to pretrain USE
        * Modified skip-thought prediction 
        * Conversational Input-Reponse Prediction 
        * Natural Language Preference 


### Attention
### Transformer
### BERT
