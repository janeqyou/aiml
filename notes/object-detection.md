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
        1. train a cnn classifier 
