#### [Lil'log Object detection for dummies](https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html)
##### image segmentation
- Felzenszwalb and Huttenlocher's algorithm, graph based image segmentation algorithms
- undirected graph $G=(V,E)$ to represent an input image. 
    + One vertex $v_{i} \in V$ represents one pixel. 
    + One edge $e=(v_i,v_j)\in$ connects two vertices $v_i$ and $v_j$. 
    + Weight $w(v_i,v_j)$ measures the dissimilarity between $v_i$ and $v_j$. The dissimilarity can be quantified in dimensions like color, location, intensity, etc. The higher the weight, the less similar two pixels are. 
    + A segmentation solution S is a partition of V into multiple connected components, ${C}$.
- selective search: Selective search is a common algorithm to provide region proposals that potentially contain objects. Used to refine regional proposal returned by image segmentation.
    + At the initialization stage, apply Felzenszwalb and Huttenlocher’s graph-based image segmentation algorithm to create regions to start with.
    + Use a greedy algorithm to iteratively group regions together
    + The process of grouping the most similar regions (Step 2) is repeated until the whole image becomes a single region.
    + Common similarity measures: 
        * color: pixel intensity similarity
        * Texture,Size (small regions are encouraged to merge together), Shape (Ideally one region can fill the gap of the other.)
##### CNN for object detection 
- Overleaf is an examplary model that combines classification, localization and detection.
    + Do image classification at different locations on regions of multiple scales of the image in a sliding window fashion, 
    + Predict the bounding box locations with a regressor trained on top of the same convolution layers.
    + training phase:
        1. Train a cnn classifier of image classification task. (with multiple class label possible ) ??
        2. Replace the top classifier layers by a regression network and train it to predict object bounding boxes at each spatial location and scale. The regressor is class-specific, each generated for one image class.
            2.1 Input: Images with classification and bounding box.
            2.2 Output: $(x_left,x_right,y_top,y_bottom)$, 4 values in total, representing the coordinates of the bounding box edges.  
            2.3 loss is l2 norm 

- R-CNN
    + Pre-train a CNN network on image classification tasks. Can use VGG or ResNet backbone. The classification involves N classes
    + Propose category-independent regions of interest by selective search (~2k candidates per image). Those region proposal may or may not contain objects and can be of different sizes 
    + Region candidates are *warped* to have a fixed size as required by CNN.
    + Continue fine-tuning the CNN on warped proposal regions for K + 1 classes; The additional one class refers to the background (no object of interest)
    + Use the fine-tuned CNN as feature extractor to generate feature representations which are used to train binary classifiers for each class independently. 
    + To correct the localization, a bounding box regressor then is trained to 
    correct the predicted detection window on bounding box correction offset using CNN features.
    + Shortcomings:
        * CNN, SVM and BBox Regressions are independent trained. The whole system is very hard to train and slow. 

- Fast R-CNN: training procedure by unifying three independent models into one jointly trained framework and increasing shared computation results, named Fast R-CNN. 
    * Model aggregates them into one CNN forward pass over the entire image and the region proposals share this feature matrix. Then the same feature matrix is branched out to be used for learning the object classifier and the bounding-box regressor.
    * RoI is no longer warped but is subject to a RoI pooling layer for fixed size required by CNN. For any RoI of size $hxw$, RoI pooling layer will impose a fixed size grid $HxW$ and performing max pooling in each cell to obtain a fixed size output 
    * training a Fast R-CNN:
        - pre-train a CNN backbone 
        - use selective search for regional proposals
        - Alter pretrained CNN:
            + replace the last max pooling layer in R-CNN with ROI pooling 
            + replace the k class softmax into k+1 softmax
        - Finally the network branches out to two output layers
            + A softmax estimator of K+1 classes, outputting a probabiliy for each RoI
            + Bounding box prediction 
            + the loss function $L=L_{cls}+L_{box}$. Smoothed L1 loss can be used because of outliers 
- Faster R-CNN: construct a single, unified model composed of *RPN (region proposal network)* and fast R-CNN with shared convolutional feature layers. There is no longer a seperate selective search stage 
    + RPN: has a classifier and a regressor. The authors have introduced the concept of anchors. Anchor is the central point of the sliding window. Classifier determines the probability of a proposal having the target object. Regression regresses the coordinates of the proposals. 
    + Each anchor can be labeled by comparing to ground truth box. For each center of sliding window, anchor number is scale * ratio
    + RPN operates on a convolution feature map which can be initialized by a pretrained CNN 
    + Fine tuning RPN is using the positively labeled anchor boxes to change the convolution feature map so the anchors class prediction and granularity of bounding box can be refined
    + Model workflow:
        * Pre-train a CNN network on image classification tasks.
        * Fine-tune the RPN (region proposal network) end-to-end for the region proposal task,
        * Train a Fast R-CNN object detection model using the proposals generated by the current RPN. (Fast R-CNN has two branches output layer)
        * Then use the Fast R-CNN network to initialize RPN training. (Re trainig RPN). While keeping the shared convolutional layers (from the original pre trained classifier), only fine-tune the RPN-specific layers. At this stage, RPN and the detection network have shared convolutional layers!
        * Finally fine-tune the unique layers of Fast R-CNN
        * Alternate fine tuning RPN and fine tuning unique layers for Fast R-CNN 
    ![Faster R-CNN](/Users/qxy001/Documents/personal_src/aiml/notes/Faster-RCNN-RPN.png)
    In the above image, feature map is convolution layer output which is shared by RPN and later fast R-CNN. RPN is used to initialize Fast R-CNN. And in turn fine tuned fast R-CNN can re initialize RPN to be trained again 

- Mask R-CNN. Extends Faster R-CNN to pixel-level image segmentation. 
        + The key point is to decouple the classification and the pixel-level mask prediction tasks. 
        + it added a third branch for predicting an object mask in parallel with the existing branches for classification and localization. The mask branch is a small fully-connected network applied to each RoI, predicting a segmentation mask in a pixel-to-pixel manner.
        + Because pixel-level segmentation requires much more fine-grained alignment than bounding boxes, mask R-CNN improves the RoI pooling layer (named “RoIAlign layer”) so that RoI can be better and more precisely mapped to the regions of the original image.

- You only look once (YOLO)
    + two stage object detector vs one stage object detector. Most of R-CNN family of object detectors are two stage detectors i.e. proposing regions first then training a CNN classifier or bounding box regressor. An alternative approach is to skip the region proposal stage and runs detection directly over a dense sampling of possible locations. This is faster and simpler, but might potentially drag down the performance a bit.
    + Yolo does not go through the regional proposal stage. It only predicts a limited number of bounding boxes. Workflow as follow:
        * Pre-train a CNN network on image classification task.
        * Split an image into S×S cells. If an object’s center falls into a cell, that cell is “responsible” for detecting the existence of that object. Each cell predicts:
            * the location of B bounding boxes within the area covered by the cell, $(x,y,w,h)$. When the cell is 'responsible' for an object, this set of coordinates positions will be penalized if they are off compared to ground truth bounding boxes ; if the cell is not 'responsible', then their prediction of bounding box positions are unconstrained. 
            * a confidence score, Pr(containing an object) x IoU(pred, truth); where Pr = probability and IoU = interaction under union. Only cells responsible for objects confidence scores are captured in loss function and those errors are penalized 
            * a probability of object class conditioned on the existence of an object in the bounding box. Probability of this object belonging to every class Ci,i=1,…,K: Pr(the object belongs to the class C_i | containing an object). Only one set of class probabilities will be predicted, regardless of  bounding boxes B.  
            * one image contains S×S×B bounding boxes, 4 coordinates, 1 confidence score and K conditional probabilities, the total prediction values for one image  is S×S×(5B+K).
        * The loss consists of two parts, the localization loss for bounding box offset prediction and the classification loss for conditional class probabilities. Most of loss are calculated in the cell and bounding boxes that contain an object. And those indicators can be calculated from ground truth from train set 
        ![Yolo-loss](/Users/qxy001/Documents/personal_src/aiml/notes/Yolo-loss.png)
        ![Yolo-loss-2](/Users/qxy001/Documents/personal_src/aiml/notes/Yolo-loss-2.png)


TODO:
- Understand RoIAlignment layer better 
- Understand Anchor Box / Regional Proposal Network Layer Better 
