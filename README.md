# 🎲 Paper-code-review

<img src=https://img.shields.io/badge/%20-Classification-brightgreen> <img src=https://img.shields.io/badge/%20-Object--Detection-lightblue> <img src=https://img.shields.io/badge/%20-Segmentation-green> <img src=https://img.shields.io/badge/%20-XAI-yellowgreen> <img src=https://img.shields.io/badge/%20-Knowledge_distillation-blueviolet> <img src=https://img.shields.io/badge/%20-Modeling-yellow> <img src=https://img.shields.io/badge/%20-Weakly--supervised-blue> <img src=https://img.shields.io/badge/%20-Semi--supervised-lightgrey> <img src=https://img.shields.io/badge/%20-Representation-orange> <img src=https://img.shields.io/badge/%20-Self--supervised-red> <img src=https://img.shields.io/badge/%20-NAS-yellow>

## 📖 Table of whole paper reviews 👉🏻 [link](https://www.notion.so/2ebb78f709c64d379b3faf277f9bf7e3?v=566189643a944cab996418b7921c3e46)  
## 👩🏻‍💻 Organization with custom implementation codes 👉🏻 [link](https://github.com/PaperCodeReview)


## Transformer
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
Arxiv | 2020 | H. Touvron et al. | [Training Data-efficient Image Transformer & Distillation through Attention](https://arxiv.org/abs/2012.12877) | Facebook AI Research (FAIR) and Sorbonne Univ. | [[official code]](https://github.com/facebookresearch/deit) [[summary]](https://www.notion.so/DeiT-Training-Data-efficient-Image-Transformer-Distillation-through-Attention-16eb2e66178945cf9c85173c72c3bc92)
Arxiv | 2020 | A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, and X. Zhai et al. | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | Google Research | [[official code]](https://github.com/google-research/vision_transformer)


## Self-supervised learning
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
Arxiv | 2020 | Z. Xie and Y. Lin et al. | [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043) | Tsinghua Univ., Xi'an Jiaotong Univ. and Microsoft Research Asia | [[custom code]](https://github.com/PaperCodeReview/PixPro)
Arxiv | 2020 | X. Chen et al. | [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297) | Facebook AI Research (FAIR) | [[official code]](https://github.com/facebookresearch/moco) [[summary]](https://www.notion.so/MoCo-v2-Improved-Baselines-with-Momentum-Contrastive-Learning-34887b3020144f28b913ede0d506ee0d)
Arxiv | 2020 | J. Grill, F. Strub, F. Altche, C. Tallec, and P. H. Richemond et al. | [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) | DeepMind, Imperial College | [[official code]](https://github.com/deepmind/deepmind-research/tree/master/byol) [[summary]](https://www.notion.so/BYOL-Bootstrap-Your-Own-Latent-A-New-Approach-to-Self-Supervised-Learning-7c87bb790b63414bad626a5892a1e2a6)
Arxiv | 2020 | P. Khosla et al. | [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) | Google Research | [official code] [[custom code]](https://github.com/PaperCodeReview/SupCL-TF) [[summary]](https://www.notion.so/Supervised-Contrastive-Learning-e2140caa8eba4fbca2ebe53a8b78dad7)
CVPR | 2020 | K. He et al. | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | Facebook AI Research (FAIR) | [[official code]](https://github.com/facebookresearch/moco) [[custom code]](https://github.com/PaperCodeReview/MoCo-TF) [[summary]](https://www.notion.so/MoCo-v1-Momentum-Contrast-for-Unsupervised-Visual-Representation-Learning-85ebd5422a02428c8bb105bf18e6a836)


## Semi-supervised learning
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
Arxiv | 2020 | K. Sohn et al. | [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) | Google Research | [[official code]](https://github.com/google-research/fixmatch) [custom code] [[summary]](https://www.notion.so/FixMatch-FixMatch-Simplifying-Semi-Supervised-Learning-with-Consistency-and-Confidence-a42958190f6c450191365d70cce08961)
CVPR | 2020 | D. Wang et al. | [FocalMix: Semi-Supervised Learning for 3D Medical Image Detection](https://arxiv.org/abs/2003.09108) | Peking University, Yizhun Medical AI | [official code] [custom code] [summary](https://www.notion.so/FocalMix-FocalMix-Semi-Supervised-Learning-for-3D-Medical-Image-Detection-564861b38e2e4daba0b3d313e3d833cd)
ICLR | 2020 | D. Berthelot et al. | [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785) | Google Research and Google Cloud AI | [[official code]](https://github.com/google-research/remixmatch) [custom code] [summary]
NeurIPS | 2019 | D. Berthelot et al. | [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) | Google Research | [[official code]](https://github.com/google-research/mixmatch) [custom code] [[summary]](https://www.notion.so/MixMatch-MixMatch-A-Holistic-Approach-to-Semi-Supervised-Learning-6b21345edf934c209ae6d4c44ef7b3e4)
Arxiv | 2019 | Q. Xie et al. | [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) | Google Research, Carnegie Mellon University | [[official code]](https://github.com/google-research/uda) [custom code] [[summary]](https://www.notion.so/UDA-Unsupervised-Data-Augmentation-for-Consistency-Training-0c7094b7b53e48618888867fe7a26e3c) [[ppt]](https://github.com/mi2rl/MI2RL-PaperStudy/blob/master/reviews/200914_Unsupervised%20Data%20Augmentation%20for%20Consistency%20Training.pdf)


## Knowledge distillation
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2020 | S. Yun and J. Park et al. | [Regularizing Class-wise Predictions via Self-knowledge Distillation](https://arxiv.org/abs/2003.13964) | KAIST | [[official code]](https://github.com/alinlab/cs-kd) [custom code] [[summary]](https://www.notion.so/CS-KD-Regularizing-Class-wise-Prredictions-via-Self-knowledge-Distillation-c4062198e53b4a0a8cd884f28e7a61f5) [[ppt]](https://github.com/DeepPaperStudy/DPS-5th/blob/master/20200808-Regularizing%20Class-wise%20Predictions%20via%20Self-knowledge%20Distillation.pdf)
CVPR | 2020 | Y. Liu et al. | [Search to Distill: Pearls are Everywhere but not the Eyes](https://arxiv.org/abs/1911.09074) | Google AI, Google Brain | [official code] [custom code] [[summary]](https://www.notion.so/Search-to-Distill-Pearls-are-Everywhere-but-not-the-Eyes-d8e57509df4244b68ef295273febc262)
ICCV | 2019 | L. Zhang et al. | [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/abs/1905.08094) | Tsinghua University | [official code] [custom code] [[summary]](https://www.notion.so/Be-Your-Own-Teacher-Improve-the-Performance-of-Convolutional-Neural-Networks-via-Self-Distillation-76e662fdb61a47b5a72376f2e54bf0d5)
ICCV | 2019 | B. Heo et al. | [A Comprehensive Overhaul of Feature Distillation](https://arxiv.org/abs/1904.01866) | NAVER Corp, Seoul National University | [[official code]](https://github.com/clovaai/overhaul-distillation) [custom code] [summary]
NeurIPS | 2018 | X. Wang et al. | [KDGAN: Knowledge Distillation with Generative Adversarial Networks](https://papers.nips.cc/paper/7358-kdgan-knowledge-distillation-with-generative-adversarial-networks) | University of Melbourne |  [[official code]](https://github.com/xiaojiew1/KDGAN/) [custom code] [[summary]](https://www.notion.so/KDGAN-KDGAN-Knowledge-Distillation-with-Generative-Adversarial-Networks-852fc740c4ea4423b57a9aa93623c3f5) [[ppt]](https://github.com/DeepPaperStudy/DPS-4th/blob/master/20200314-KDGAN-%EA%B9%80%EC%84%B1%EC%B2%A0.pdf)
ICML | 2018 | S. Srinivas et al. | [Knowledge Transfer with Jacobian Matching](https://arxiv.org/abs/1803.00443) | Idiap Research Institute & EPFL |  [official code] [custom code] [[summary]](https://www.notion.so/Knowledge-Transfer-with-Jacobian-Matching-40027da0d38943f2a69623aae25b2eed)


## Modeling & NAS
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
Arxiv | 2020 | T. Nguyen et al. | [Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth](https://arxiv.org/abs/2010.15327) | Google Research | -
CVPR | 2020 | I. Radosavovic et al. | [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) | Facebook AI Research (FAIR) | [[official code]](https://github.com/facebookresearch/pycls) [[custom code]](https://github.com/PaperCodeReview/RegNet-TF) [[summary]](https://www.notion.so/RegNet-Designing-Network-Design-Spaces-455b9494747c46a29b3b6eb9e70425c0) [[ppt]](https://github.com/DeepPaperStudy/DPS-5th/blob/master/20200530-Designing%20Network%20Design%20Spaces-SungchulKim.pdf)
CVPR | 2019 | T. He et al. | [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) | Amazon Web Services | -
ICML | 2019 | S. Kornblith et al. | [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) | Google Brain | [[official code]](https://github.com/google-research/google-research/tree/master/representation_similarity)


## XAI (Explainable AI)
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
Arxiv | 2019 | M. Yang et al. | [Benchmarking Attribution Methods with Relative Feature Importance](https://arxiv.org/abs/1907.09701) | Google Brain | [[official code]](https://github.com/google-research-datasets/bam) [summary]
NeurIPS | 2019 | S. Hooker et al. | [A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758) | Google Brain | [[summary]](https://www.notion.so/A-Benchmark-for-Interpretability-Methods-in-Deep-Neural-Networks-fc219d4e2d8242509d0f732d17aeb0fe) [[ppt]](https://github.com/DeepPaperStudy/DPS-4th/blob/master/20200201-ROAR-%EA%B9%80%EC%84%B1%EC%B2%A0.pdf)
ICCV Workshop | 2019 | B. Kim et al. | [Why are Saliency Maps Noisy? Cause of and Solution to Noisy Saliency Maps](https://arxiv.org/abs/1902.04893) | KAIST | [[official code]](https://github.com/1202kbs/Rectified-Gradient)
NeurIPS | 2018 | J. Adebayo et al. | [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) | Google Brain | -
ICML Workshop | 2018 | J. Seo et al. | [Noise-adding Methods of Saliency Map as Series of Higher Order Partial Derivative](https://arxiv.org/abs/1806.03000) | Satrec Initiative, KAIST | -
CVPR | 2018 | Q. Zhang et al. | [Interpretable Convolutional Neural Networks](https://arxiv.org/abs/1710.00935) | University of California | [[official code]](https://github.com/zqs1022/interpretableCNN)
ICCV | 2017 | R. Selvaraju et al. | [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) | Georgia Institute of Technology |  [[official code]](https://github.com/ramprs/grad-cam/) [[ppt]](https://github.com/DeepPaperStudy/DPS-2nd/blob/master/Grad-CAM.pdf)
CVPR | 2017 | D. Smilkov et al. | [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825) | Google Inc. | -


## Registration for medical image
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
ICCV | 2019 | S. Zhao et al. | [Recursive Cascaded Networks for Unsupervised Medical Image Registration](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_Recursive_Cascaded_Networks_for_Unsupervised_Medical_Image_Registration_ICCV_2019_paper.pdf) | Tsinghua Univ., Beihang Univ., and Microsoft Research | [[official code]](https://github.com/microsoft/Recursive-Cascaded-Networks) [custom code] [summary]
Journal of Biomedical and Health Informatics | 2019 | S. Zhao et al. | [Unsupervised 3D End-to-End Medical Image Registration with Volume Tweening Network](https://arxiv.org/abs/1902.05020) | Tsinghua Univ., Beihang Univ., and Microsoft Research | [[official code]](https://github.com/microsoft/Recursive-Cascaded-Networks) [[summary]](https://www.notion.so/VTN-Unsupervised-3D-End-to-End-Medical-Image-Registration-with-Volume-Tweening-Network-01c73219e7e94984add6f0baabd769c4)
CVPR, TMI | 2018 | G. Balakrishman et al. | [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](https://arxiv.org/abs/1809.05231) | MIT and Cornell Univ. | [[official code]](https://github.com/voxelmorph/voxelmorph) [[summary]](https://www.notion.so/VoxelMorph-VoxelMorph-A-Learning-Framework-for-Deformable-Medical-Image-Registration-b19edd095c284bd49bd0504cd29e98a2)


## Weakly-supervised learning
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2019 | J. Lee et al. | [FickleNet: Weakly and Semi-supervised Semantic Image Segmentation using Stochastic Inference](https://arxiv.org/abs/1902.10421) | Seoul National Univ. | [official code] [custom code] [[summary]](https://www.notion.so/FickleNet-FickleNet-Weakly-and-Semi-supervised-Semantic-Image-Segmentation-using-Stochastic-Infere-888599b7fbb44ee8bb2dd99495e9d4df) [[ppt]](https://github.com/mi2rl/MI2RL-PaperStudy/blob/master/reviews/200803_FickleNet.%20Weakly%20and%20Semi-supervised%20Semantic%20Image%20Segmentation%20using%20Stochastic%20Inference.pdf)


## Representation learning
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2019 | J. Deng et al. | [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) | Imperial College London, InsightFace, FaceSoft | [[official code]](https://github.com/deepinsight/insightface) [[summary]](https://www.notion.so/ArcFace-ArcFace-Additive-Angular-Margin-Loss-for-Deep-Face-Recognition-1b9b623a42c74ad689cd4b8042edb033)
CVPR | 2017 | W. Liu et al. | [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063) | Georgia Institute of Technology, Carnegie Mellon Univ., and Sun Yat-Sen Univ. | [[official code]](https://github.com/wy1iu/sphereface) [[summary]](https://www.notion.so/SphereFace-SphereFace-Deep-Hypersphere-Embedding-for-Face-Recognition-4b08a3bac24745a898d4a2ee7642a660)


## Attention
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2020 | I. Kim and W. Baek et al. | [Spatially Attentive Output Layer for Image Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Spatially_Attentive_Output_Layer_for_Image_Classification_CVPR_2020_paper.pdf) | Kakao Brain | [[summary]](https://www.notion.so/SAOL-Spatially-Attentive-Output-Layer-for-Image-Classification-6ce7069724a140bda40bd8ba45bfae1a)
ECCV | 2020 | M. KIm and J. Park et al. | [Learning Visual Context by Comparison](https://arxiv.org/abs/2007.07506) | Lunit Inc. and Seoul National Univ. Hospital | [[official code]](https://github.com/mk-minchul/attend-and-compare) [[summary]](https://www.notion.so/Learning-Visual-Context-by-Comparison-e976c8c83f8f4c8c8f2caa2ddaf7710a)
CVPR | 2018 | J. Hu et al. | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | University of Chinese Academy of Sciences |  [[official code]](https://github.com/hujie-frank/SENet) [custom code] [summary]
ECCV | 2018 | S. Woo et al. | [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) | KAIST |  [[official code]](https://github.com/Jongchan/attention-module) [custom code] [[summary]](https://www.notion.so/CBAM-CBAM-Convolutional-Block-Attention-Module-85f161eda58a417d84a20e6d4a3ed97c)


## Object Segmentation
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2019 | A. Kirillov et al. | [Panoptic Segmentation](https://arxiv.org/abs/1801.00868) | Facebook AI Research (FAIR) and  Heidelberg Univ. | [[ppt]](https://github.com/DeepPaperStudy/DPS-3rd/blob/master/190817_Panoptic%20Segmentation.pdf)
ECCV | 2018 | L. Chen et al. | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) | Google Inc. |  [[official code]](https://github.com/tensorflow/models/tree/master/research/deeplab) [[ppt]](https://github.com/DeepPaperStudy/DPS_1st/blob/master/architecture/Deeplabv1%2Cv2%2Cv3%2Cv3%2B.pdf)
MIDL | 2018 | O. Oktay et al. | [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) | Imperial College London, Babylon Heath |  [[official code]](https://github.com/ozan-oktay/Attention-Gated-Networks)
MICCAI | 2016 | Ö. Çiçek et al. | [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) | University of Freiburg | -
MICCAI | 2015 | Ö. Ronneberger et al. | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | University of Freiburg | -
CVPR | 2015 | J. Long et al. | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | UC Berkeley |  [[official code]](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn) [[summary]](https://www.notion.so/FCN-Fully-Convolutional-Networks-for-Semantic-Segmentation-4b8bee55e681449ebe8a1a6a9e3b8fc9)
ICLR | 2015 | L. Chen et al. | [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062) | University of California, Google Inc., and CentraleSupelec |  [[official code]](https://github.com/tensorflow/models/tree/master/research/deeplab) [[summary]](https://www.notion.so/DeepLab-v1-Semantic-Image-Segmentation-with-Deep-Convolutional-Nets-and-Fully-Connected-CRFs-693d8f64cae1433b9cb454c17681c404) [[ppt]](https://github.com/DeepPaperStudy/DPS_1st/blob/master/architecture/Deeplabv1%2Cv2%2Cv3%2Cv3%2B.pdf)


## Object Detection
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
CVPR | 2020 | G. Song et al. | [Revisiting the Sibling Head in Object Detector](https://arxiv.org/abs/2003.07540) | SenseTime X-Lab and The Chinese Univ. of Hong Kong | [[official code]](https://github.com/Sense-X/TSD) [[summary]](https://www.notion.so/TSD-Revisiting-the-Sibling-Head-in-Object-Detector-f45900bffc364666bfe390131ea1a7ec)
ECCV | 2020 | N. Carion et al. | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | Facebook AI Research (FAIR) | [[official code]](https://github.com/facebookresearch/detr) [[custom code]](https://github.com/PaperCodeReview/DETR-TF) [[summary]](https://www.notion.so/DETR-End-to-End-Object-Detection-with-Transformers-b5f0a27e7edc4c519c5d16ba99b90be4)
CVPR | 2019 | H. Rezatofighi et al. | [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630) | Stanford Univ., The University of Adelaide, and Aibee Inc | [official code] [custom code] [summary] [[ppt]](https://github.com/DeepPaperStudy/DPS-2nd/blob/master/GIoU.pdf)


## Classification
From | Year | Authors | Paper | Institution | url
---- | ---- | ---- | ---- | ---- | ----
ICML | 2019 | M. Tan et al. | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | Google Research | [[official code]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) [custom code] [summary] [[ppt]](https://github.com/DeepPaperStudy/DPS-3rd/blob/master/191019_EfficientNet.pdf)
CVPR | 2017 | S. Xie et al. | [Aggregated Residual Transformations for Deep Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) | UC San Diego |  [[official code]](https://github.com/facebookresearch/ResNeXt) [custom code] [summary]
CVPR | 2017 | F. Chollet et al. | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | Google Inc. |  [official code] [custom code] [summary]
CVPR | 2016 | K. He et al. | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Microsoft Research |  [[official code]](https://github.com/KaimingHe/deep-residual-networks) [custom code] [summary](https://www.notion.so/ResNet-Deep-Residual-Learning-for-Image-Recognition-a5bd9f094f394707abbdd218d6390a4a)
ICLR | 2015 | k. Simonyan et al. | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | University of Oxford |  [official code] [custom code] [summary]
CVPR | 2015 | C. Szegedy et al. | [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) | Google Inc. |  [official code] [custom code] [summary] [[ppt]](https://github.com/DeepPaperStudy/DPS_1st/blob/master/architecture/InceptionV1.pdf)
