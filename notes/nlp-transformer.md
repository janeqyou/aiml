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
         









### Transformer Stack
