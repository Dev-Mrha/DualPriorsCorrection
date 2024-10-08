<div align="center">
<h1>Combining Generative and Geometry Priors for Wide-Angle Portrait Correction</h1>

<h4 align="center">Lan Yao<sup>1</sup>, Chaofeng Chen<sup>2</sup>, Xiaoming Li<sup>1</sup>, Zifei Yan<sup>1</sup>, Wangmeng Zuo<sup>1,3</sup></h4>

<div>
    <sup>1</sup> Harbin Institute of Technology,
    <sup>2</sup> Nanyang Technological University,
    <sup>3</sup> Pazhou Lab, Huangpu
</div>

<!-- [Paper]() | [Project Page]() -->


<p><B>we introduce a framework that leverages generative and geometry priors to rectify wide-angle distortions in faces and  backgrounds.
Experiments demonstrate that our approach outperforms previous methods by a large margin, excelling not only in quantitative measures such as line straightness and shape consistency metrics but also in terms of perceptual visual quality</B></p>

<img src="./figures/intro.png" width="800px">

<!-- <p align="justify">Given a wide-angle image, LineCNet generate the flow to correct distortion in the background for the whole image, while FaceCNet deal with each faces crop and align by face alignment. In FaceCNet, the face first inverse into StyleGAN  latent space by e4e encoder, and a U-Net framework combining the multi-scale features from StyleGAN generate the correction flow. After seperate correction, a post-process is done for face fusion. </p> -->

<p align="justify">Given a wide-angle image, LineCNet and FaceCNet deal with the correction in background and face region seperatly, then face fusion is done by a post-process. </p>

</div>

## TODO
- [x] Release the source code.
- [x] Release the checkpoint model.


## Installation

All dependencies for defining the environment are provided in environment.yaml.

## Evaluate

Because of our post process, ShapeAcc and Landmark Distance cannot directly generated by an output flow, which we calculate by using a face detection tool to get each faces keypoints and calculate according to ```./evaluate.py```.

### Step 0: download the weights
> [checkpoints](https://pan.baidu.com/s/1B9a5BPZyfX2-zwA5PzpZ4w?pwd=g1d4) should be put in ```./pretrained_models/```

### Step 1: download the dataset

[public dataset](https://pan.baidu.com/share/init?surl=MvwulIIs2CowfQ-8d0gcsQ&pwd=5pe5) released by work [Practical Wide-Angle Portraits Correction with Deep Structured Models](https://github.com/TanJing94/Deep_Portraits_Correction?tab=readme-ov-file)

### Step 2: generate the output of test dataset

```
python evaluate.py --option generate --test-dir ../test/ --device cuda:0 --e4e-path ./pretrained_models/e4e_best_model.pth --linenet-path ./pretrained_models/linenet.pt --facenet-path ./pretrained_models/facenet.pt
```

The generated output is before post-process part, which means for each input xx.jpg, you now have a xx_out.jpg and xx_mask.jpg that you may need a image inpainting alogorithm to generate the final output. You can use [lama](https://github.com/advimman/lama) for this part. 

### Step 3: Evaluate

Cause in our method, when we generate the output, we do not have a whole flow for the entire image. So when we calculate the evaluation metrics, it extraly cost to get the final combined flow, which means this evaluation need much longer time than step 3 (generate output).

```
python evaluate.py --option evaluate --test-dir ../test/ --device cuda:0 --e4e-path ./pretrained_models/e4e_best_model.pth --linenet-path ./pretrained_models/linenet.pt --facenet-path ./pretrained_models/facenet.pt
```

## Training
Data for finetune e4e encoder:
- generated distorted face data using FFHQ and CelebA-HQ

Training Data for FaceCNet:
- generated distorted face data using FFHQ and CelebA-HQ
- cropped face from [public datset](#step-1:-download-the-dataset)

```
python train_facecnet.py
```

Training Data for lineCNet:
- [public dataset](#step-1:-download-the-dataset)

```
python train_linecnet.py
```


## Acknowledgement
This project is built based on [encoder4editing](https://github.com/omertov/encoder4editing). 


