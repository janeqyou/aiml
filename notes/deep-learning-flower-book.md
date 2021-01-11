#### Chapter 5 Challenges motivating Deep Learning 
* Limitations of traditional machine Learning
    * Difficulty to generalize 
    * inefficient of calculate complicated functions of high dimensions
* The Curses of Dimensionality 
    * In high-dimensional spaces, the number of configurations is huge, much larger than our number of examples. A typical grid cell has no examples associted with it
* Local Constancy and Smoothness Regularization 
    * To generalize well, machine learning algorithm usually needs to be guided by prior beliefs about what kind of function they should learn. 
        - explicit beliefs as in the form of probability distributions over parameters of the model 
        - prior belief influncing function directly, but influencing parameters in directly 
        - prior belief as implicitly choosing one class of algorithm over another 
    * **smoothness piror** or **local constancy prior**. The function we learn should not change very much within a small region:
        $f^*(x) = f^*(x+\varepsilon)$

    * Much of modern deep learning is motivated by studying the local template matching and how deep learning can address when local template matching failed. For example. to distinguish $O(K)$ regions usually machine learning algorithm needs K training set, one in each region; $O(K)$ parameters, $O(1)$ within each region. The local constancy or smoothness assumption works well as long as there are enough training samples to observe most of peaks and valleys of the true underlying function to be learnt. 
    
    To learn a complex function has more regions than training examples available, machine learning fell short. For example if a function takes a form of color of a checkerboard. If learner can correctly guess the color of new points, if there is a point of same color in the train and in the region. However, if there is only points away in a different checker board square, the learning can be wrong. 

    How deep learning represents a complicated function efficiently, or generalie well ? The key insight is that a very large number of regions, such as $O(2^k)$ can be defined by $O(K)$ training samples. As long as we can introduce dependencies among regions through additional assumption of data generating distributions. 

    * In Neural Network family algorithms, no strong and task specific task assumptions are made. So we could learn algorithms that address more general purpose assumptions 







