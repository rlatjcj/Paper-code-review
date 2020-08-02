# End-to-End Object Detection with Transformers
## Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko
## Facebook AI Research (FAIR)
## Arxiv 2020

[[arxiv]](https://arxiv.org/abs/2005.12872) [[official code (pytorch)]](https://github.com/facebookresearch/detr) [[unofficial code (tensorflow)]](https://github.com/Leonardo-Blanger/detr_tensorflow)  
[The DETR model](#The-DETR-model)

# The DETR model
- A set prediction loss that forces unique matching between predicted and ground truth boxes
- An architecture that predicts (in a single pass) a set of objects and models their relation
## Object detection set prediction loss
- A fixed-size set of <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> predictions
    - <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> : larget than the typical number of objectes in an image
### Matching loss
- An optimal bipartite matching between predicted and ground truth objects
- Object-specific (bounding box) losses
    - <img src="https://latex.codecogs.com/svg.latex?\;y" title="y" /> : the ground truth set of objects
    - <img src="https://latex.codecogs.com/svg.latex?\;\hat{y} = \{\hat{y}_{i}\}^{N}_{i=1}" title="\hat{y} = \{\hat{y}_{i}\}^{N}_{i=1}" /> : the set of N predictions
    <p align="center"><img width="100%" src="img/DETR_eq1.PNG" /></p>

    - <img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})" title="\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})" /> : a pair-wise *matching cost* between ground truth <img src="https://latex.codecogs.com/svg.latex?\;y_i" title="y_i" /> and a prediction with index <img src="https://latex.codecogs.com/svg.latex?\;\sigma(i)" title="\sigma(i)" />
    - Hungarian algorithm [[arxiv]](https://arxiv.org/abs/1506.04878)
    - <img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})=\mathbb{1}_{\{c_{i}\neq\emptyset\}}\hat{p}_{\sigma(i)}(c_i)+\mathbb{1}_{c_{i}\neq\emptyset}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})" title="\mathcal{L}_{match}(y_i,\hat{y}_{\sigma(i)})=\mathbb{1}_{\{c_{i}\neq\emptyset\}}\hat{p}_{\sigma(i)}(c_i)+\mathbb{1}_{c_{i}\neq\emptyset}\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})" />
- *Hungarian loss*
    <p align="center"><img width="100%" src="img/DETR_eq2.PNG" /></p>

    - <img src="https://latex.codecogs.com/svg.latex?\;\hat{\sigma}" title="\hat{\sigma}" /> : the optimal assignment computed in the first step (eq.1)
### Bounding box loss
- A linear combination of the <img src="https://latex.codecogs.com/svg.latex?\;l_1" title="l_1" /> loss
- The generalized IoU loss
- <img src="https://latex.codecogs.com/svg.latex?\;\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})=\lambda_{iou}\mathcal{L}_{iou}(b_i,\hat{b}_{\sigma(i)})+\lambda_{L1}||b_i-\hat{b}_{\sigma(i)}||_1" title="\mathcal{L}_{box}(b_i,\hat{b}_{\sigma(i)})=\lambda_{iou}\mathcal{L}_{iou}(b_i,\hat{b}_{\sigma(i)})+\lambda_{L1}||b_i-\hat{b}_{\sigma(i)}||_1" />

## DETR architecture
<p align="center"><img width="100%" src="img/DETR_fig2.PNG" /></p>

### Backbone
- <img src="https://latex.codecogs.com/svg.latex?\;x_{img}\in\mathbb{R}^{3\times{H_0}\times{W_0}}" title="x_{img}\in\mathbb{R}^{3\times{H_0}\times{W_0}}" /> : the initial image
- <img src="https://latex.codecogs.com/svg.latex?\;f\in\mathbb{R}^{C\times{H}\times{W}}" title="f\in\mathbb{R}^{C\times{H}\times{W}}" /> : a lower-resolution activation map
    - <img src="https://latex.codecogs.com/svg.latex?\;C=2048" title="C=2048" /> and <img src="https://latex.codecogs.com/svg.latex?\;{H},{W}=\frac{H_0}{32},\frac{W_0}{32}" title="{H},{W}=\frac{H_0}{32},\frac{W_0}{32}" />

### Transformer encoder
- A 1x1 convolution reduces the channel dimension of the high-level activation map <img src="https://latex.codecogs.com/svg.latex?\;f" title="f" /> from <img src="https://latex.codecogs.com/svg.latex?\;C" title="C" /> to a smaller dimension <img src="https://latex.codecogs.com/svg.latex?\;d" title="d" /> creating a new feature map <img src="https://latex.codecogs.com/svg.latex?\;z_0\in\mathbb{R}^{d\times{H}\times{W}}" title="z_0\in\mathbb{R}^{d\times{H}\times{W}}" />.
- The spatial dimensions of <img src="https://latex.codecogs.com/svg.latex?\;z_0" title="z_0" /> is collapsed into one dimension, resulting in a <img src="https://latex.codecogs.com/svg.latex?\;d\times{HW}" title="d\times{HW}" /> feature map.
- Each encoder layer has a standard architecture and consists of a multi-head self-attention module and a feed forward network (FFN).
- Each encoder layer is supplemented with fixed positional encodings that are added to the input of each attention layer.

### Transformer decoder
- Decoder decodes the <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> objects in parallel at each decoder layer.
- The <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> input embeddings are learnt positional encodings that are refered as *object queries*.
- Object queries are added to the input of each attention layer.
- Output embeddings are *independently* decoded into box coordinated and class labels b y a feed forward network, resulting <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> final predictions.

### Prediction feed-forward networks (FFNs)
- A 3-layer perceptron with ReLU activation function and hidden dimension <img src="https://latex.codecogs.com/svg.latex?\;d" title="d" />, and a linear projection layer
- Because of a fixed size set of <img src="https://latex.codecogs.com/svg.latex?\;N" title="N" /> bounding boxes, an additional special class label <img src="https://latex.codecogs.com/svg.latex?\;\emptyset" title="\emptyset" /> is used to represent that no object is detected within a slot. This class plays a similar role to the "background" class in the standard object detection approaches.

### Auxiliary decoding losses
- To help the model output the correct number of objects of each class
