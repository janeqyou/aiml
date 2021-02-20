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
    
        * Modified skip-thought prediction. Using only an encoder based on transformer or DAN to predict the previous sentence and next sentence using current sentence. Data source is Wikipedia. 
        ![use-skip-thoughts](/Users/qxy001/Documents/personal_src/aiml/notes/use-skip-thought.png)  


        * Conversational Input-Response Prediction. predict the correct response for a given input among a list of correct responses and other randomly sampled responses. 
        ![use-input-response.png](/Users/qxy001/Documents/personal_src/aiml/notes/use-input-response.png)
        
        * Natural Language Inference. Need to predict if a hypothesis entails, contradicts, or is neutral to a premise.The authors used the 570K sentence pairs from SNLI corpus to train USE on this task.
        ![use-language-inference](/Users/qxy001/Documents/personal_src/aiml/notes/use-language-inference.png)
### Attention
- Seq2Seq Attention (or Attention in Vanilla RNN)
    + Recap of Seq2Seq RNN. One RNN reads input and predict outputs. Input sequence is $X=[x_1,x_2,x_3...x_n]$ and output sequence is $Y=[y_1,y_2,y_3,...y_n]$. Hidden layer is bi directional RNN $h_i=[\overrightarrow{h_i};\overleftarrow{h_i}]$. The hidden state update is $h_i=f(Vh_{i-1}+W_eX_i)$. The prediction $y_i=SoftMax(W_h\cdot h_i+c)$.
    
    + Recap of Encoder Decoder RNN. Two RNNs. One, $\{h_i\}$ for encoding input sequence into a context vector $c$, the other one $\{s_i\}$ for decoding into output sequence. $c=\sum_{i=n}^{N} h_i$. $s_i=f(s_{i-1},y_{i-1},c_i)$

    + Adding attention to Encoder/Decoder RNN. To address the issue of context vector c being over simplifed. Present the following attention mechanism to align decoder hidden state with encoder hidden state. So the input step $i$'s impact on output step $t$ can be quantified. 
        * context vector for output $y_i$: $c_t=\sum_{i=1}^{N}a_{it}h_i$
        * *$a_{it}=align(y_i,x_i)=\frac{exp(score(s_{t-1},h_i))}{\sum_{\prime i=1}^{N}exp(score(s_{t-1},h_(\prime i))}$*. The alignment model assigns a score $a_it$ to the pair of input at position i and output at position t, $(y_t,x_i)$, based on how well they match. The set of {$a_ti$} are weights defining how much of each source hidden state should be considered for each output. The matrix of alignment score is a nice by product to visualize how 
        * *Additive attention (In Bahdanau’s paper)*:$v_{a}^{T}Tahn(W_a[s_t;h_i])$. The alignment score $a$ is parametrized by a feed-forward network with a single hidden layer and this network is jointly trained with other parts of the model. The score function is therefore in the following form, given that tanh is used as the non-linear activation function:  It could be forumlated as feed forward network $v_{a}$ and $W_a$ can be jointly learnt.
        * *(Scale) Product attention*: $\frac{{s_t}^Th_i}{\sqrt{n}}$, n is the length of input sequence. When n is really large, this to prevent vanishing gradient 
        * Adding attention mechanism to Encoder/Decoder RNN $s_i=f(s_{i-1},y_{i-1},\sum_{i=1}^{n}a_{ti}hi)$, $y_i=softmax(s_i)$, or in language model, we can use beam decoder to find $\{\hat{y_i}\}$ such to maximize the likelihood of $\sum_{i=1}^{N}P(y_{i_1}|y_{i})$
- [code/Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
-  Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence.

### Transformer
 - Entirely built on the *self-attention* mechanisms without using sequence-aligned recurrent architecture.
 - Key,Value and Query (K,V,Q). The transformer views the encoded representation of the *input as a set of key-value pairs, (K,V)*, both of dimension n (input sequence length); Both the keys and values are the encoder hidden states. In the decoder, the previous *output* is compressed into a *query (Q of dimension m)* and the next output is produced by mapping this query and the set of keys and values.
 
 - Scaled dot-product attention. the output is a weighted sum of the values, where the weight assigned to each value is determined by the dot-product of the query with all the keys:
 <p style="text-align: center;">$Attention(K,Q,V) = softmax(\frac{QK^T}{\sqrt{n}}) \cdot V$</p>. Here the dot product still happen among input and output. The result is attention wegiths assigned to value which is still the input 
 - Multi-head scaled dot-product attention mechanism. Rather than only computing the attention once, the multi-head mechanism runs through the scaled dot-product attention multiple times in parallel. $(K,V,Q)$ are all first linearly transferred then participate in calculating the self attention i.e. <p style="text-align: center;">$MultiHead(Q,K,V)=[head_1;…;head_h]W^O , where\ head_i = Attention(KW^k,QW^Q,VW^v)$)</p>
 - Encoder:
 ![encoder](/Users/qianyou/src/aiml/notes/encoder.png).
     + A stack of N=6 identical layers.
     + Each layer has a multi-head self-attention layer and a simple position-wise fully connected feed-forward network.
     + Each sub-layer adopts a residual connection and a layer normalization.
     + All the sub-layers output data of the same dimension dmodel=512.
     

- Decoder:
![decoder](/Users/qianyou/src/aiml/notes/decoder.png).
    + A stack of N = 6 identical layers
    + Each layer has two sub-layers of multi-head attention mechanisms and one sub-layer of fully-connected feed-forward network.
    + Similar to the encoder, each sub-layer adopts a residual connection and a layer normalization.
    + Decoder is usually used to input the target sequence or the next sequence. For example, the source sequence [How, are, you] and the targe sequence [<start>, I, am, fine] will be the input to encoder and decoder respectively. And the tokens in target sequence will be feed in one step by step as the previous output $y_{i-1}$. The prediction is done at output layer, it is essentially a multi class classification where the output vector has the size of vocab and the one with probability 1.0 will be the index of the predicted word. As the The first multi-head attention sub-layer is modified to prevent positions from attending to subsequent positions, as we don’t want to look into the future of the target sequence when predicting the current position. For example, 'am' can only attend to 'I' but not to 'fine' because ''
    
- Full Architecture 
![transformer](/Users/qianyou/src/aiml/notes/transformer.png).
    + In the decoder in the second sublayer module, output of source sequence encoder is K and V, output of target sequence is Q.
    
### [BERT](From Udemy Class:Master Deep Learning and Neural Networks Theory and Applications with Python and PyTorch! Including NLP and Transformers)
- BERT is pre-trained on large text corpus with two tasks:
    + Masked Language Modeling. Before feeding words sequences to BERT, 15% of the words in each sequence are replaced with a [MASK] token or randomly replaced with other words. The model then attempts to predict masked words, based on the context provided by the other non-masked words in the sequence. The loss function takes into consideration only the prediction of the masked words and ignores the prediction of the non-masked words. 
![bert-masked-lm](/Users/qianyou/src/aiml/notes/bert-architecture.png).


    + Sentence Matching / Next sentence prediction 
![bert-sent-matching-1](/Users/qianyou/src/aiml/notes/bert-sent-matching-1.png)
![bert-sent-matching-2](/Users/qianyou/src/aiml/notes/bert-sent-matching-2.png)

Then it is finetuned for downstream task. 
    + Bi-directional. The self-attention mechanism in a transformer allows each words of the sequence to attend to relevant words of the same sequence. Left or right context. Therefore transformer is considedred *bidirectional*
- Architecture. One of BERT's aim is to produce rich representation or embeddings for words. So it removed transformer's decoder. Is based on multi layers of transformers encoder.

-  
- Use its embeddings 
+ And as the model trains to predict, it learns to produce a powerful internal representation of words as word embedding. 
