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

