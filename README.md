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
- [ ] add code about course-to-fine training
- [ ] release model mentioned in paper
- [ ] train model again for popular song with chinese, with dataset released by TME.


## Usage

### Data Prepare

Before extracting features and training, we need to prepare our dataset as json-format.
Every line of data file is a json-format contains information about wav file path, duration, speaker id and so on.
An example is as below: 

```text
{"utt": "cover80_00000000_0_0", "wav": "data/covers80/wav_16k/annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale.wav", "dur_s": 316.728, "song": "A_Whiter_Shade_Of_Pale", "version": "annie_lennox+Medusa+03-A_Whiter_Shade_Of_Pale"}
{"utt": "cover80_00000000_0_1", "wav": ...}
...
```

And we offered raw wav files of Covers80 and dataset.txt in this project.

### Feature extract

As mentioned in the paper, We use cqt feature extracted from signal. The script to extract features is as below, and it is worth noting that augmentation will be implement at this stage(very important for improving MAP). 

```bash
python3 -m tools.extract_csi_features data/covers80/
```
 
### Train

It is easy to run the train stage as:
```
python3 -m tools.extract_csi_features egs/covers80/
```

And our code supports run on multi gpus. Just use:
```
torchrun -m --nnodes=1 --nproc_per_node=2 tools.train egs/covers80/
```

### Evaluation

For the convenience of viewing during training, we plot MAP at tensorboard after every epoch.
A script is also offered to calculate MAP, top10, and rank1 with pretrained model.
The features of test-set data needs to be extracted first with  tools.extract_csi_features.

```
python3 -m tools.eval_testset pretrain-model-dir query_path ref_path
```

 
