# Cover<font color=#7555DA>H</font><font color=#F6C852>u</font>nter: An efficient and accurate system of CSI(Cover Song Identification)

This is the official PyTorch implementation of the following paper:

https://arxiv.org/abs/2306.09025

> **COVERHUNTER: COVER SONG IDENTIFY WITH REFINED ATTENTION AND ALIGNMENTS**
> Feng Liu, Deyi Tuo, Yinan Xu, Xintong Han

> **Abstract**: *Cover song identification (CSI) focuses on finding the same 
music with different versions in reference anchors given a query track. 
In this paper, we propose a novel system named CoverHunter that overcomes the 
shortcomings of existing detection schemes by exploring richer features with 
refined attention and alignments. 
CoverHunter contains three key modules: 1) A convolution-augmented transformer 
(i.e., Conformer) structure that captures both local and global feature 
interactions in contrast to previous methods mainly relying on convolutional 
neural networks; 2) An attention-based time pooling module that further 
exploits the attention in the time dimension; 3) A novel coarse-to-fine 
training scheme that first trains a network to roughly align the song chunks 
and then refines the network by training on the aligned chunks. 
At the same time, we also summarize some important training tricks used in our 
system that help achieve better results. Experiments on several standard CSI 
datasets show that our method significantly improves over state-of-the-art methods 
with an embedding size of 128 (2.3% on SHS100K-TEST and 17.7% on DaTacos).*

more code and model will be released soon.

- [x] add main code
- [x] add egs demo for covers80 
- [x] add code about course-to-fine training
- [x] release model mentioned in paper
- [ ] train model again for popular song with chinese, with dataset released by TME.


## Usage

We take Covers80 as an example to show the whole process.

### Data Prepare

Before extracting features and training, we need to prepare our dataset as json-format.
Every line of data file is a json-format contains information about wav file path, duration, speaker id and so on.
An example is as below: 

```text
{"utt": "cover80_00000000_0_0", "wav": "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav", "dur_s": 316.728, "song": "A_Whiter_Shade_Of_Pale", "version": "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale"}
{"utt": "cover80_00000000_0_1", "wav": ...}
...
```

Only audio with sample rate of 16k is enough for CSI task. 
And Covers80 dataset can be download from http://labrosa.ee.columbia.edu/projects/coversongs/covers80/, and make sure
audio file is available in the dataset.

### Feature extract

As mentioned in the paper, We use cqt feature extracted from signal. The script to extract features is as below, and it is worth noting that augmentation will be implement at this stage(very important for improving MAP). 

```bash
python3 -m tools.extract_csi_features data/covers80/
```
 
### Train

It is easy to run the train stage as:
```
python3 -m tools.train egs/covers80/
```

And our code supports run on multi gpus. Just use:
```
torchrun -m --nnodes=1 --nproc_per_node=2 tools.train egs/covers80/
```

### Evaluation

For the convenience of viewing during training, we plot MAP at tensorboard after every epoch.
A script is also offered to calculate MAP, top10, and rank1 with pre-trained model.
The features of test-set data needs to be extracted first with tools.extract_csi_features.
Note for test-set feature, we do not need augmentation, so the hparams file should be as one at data/covers80_testset.

After feature extracting, we can run as below: 

```
python3 -m tools.eval_testset pretrain-model-dir query_path ref_path 
``` 
Another choice is to use pre-trained model. Download my model from https://drive.google.com/file/d/1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j/view.
It is trained with SHS100k-train and Covers80 is not included in the train-set. 
After unzip it, you can run to eval Covers80 and get results shown like this:

```
2023-07-05 16:38:46,621 INFO Test, map:0.9266781356046699 rank1:3.0853658536585367
```

## Other details

### coarse-to-fine details
As stated in the paper, the CSI task can be better accomplished 
with additional alignment information.
In order to keep the training code simple and readable,
these alignment information is not contained in the dataloader code.
However, a basic code is offered to 
explain our process of obtaining alignment information.
And you can run the following code:

```bash
python3 -m tools/alignment_for_frames pretrained-model-dir, data-path, output-alignment-path
```

---
Hope that this project can help beginners to get started in this CSI field more quickly. 
If you have any questions, feel free to send me an email(liufeng900204@163.com) or ask in issue. Good luck!