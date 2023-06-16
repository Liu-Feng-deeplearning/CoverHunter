# Cover<font color=#7555DA>H</font><font color=#F6C852>u</font>nter: An efficient and accurate system of CSI(Cover Song Identification)

This is the official PyTorch implementation of the following paper:

https://arxiv.org/abs/2306.09025

> **COVERHUNTER: COVER SONG IDENTIFY WITH REFINED ATTENTION AND ALIGNMENTS**
> Feng Liu, Deyi Tuo, Yinan Xu, Xintong Han

> **Abstract**: *Cover song identification (CSI) focuses on finding the same music with different versions in reference anchors given a query track. In this paper, we propose a novel system named CoverHunter that overcomes the shortcomings of existing de- tection schemes by exploring richer features with refined at- tention and alignments. CoverHunter contains three key mod- ules: 1) A convolution-augmented transformer (i.e., Con- former) structure that captures both local and global feature interactions in contrast to previous methods mainly relying on convolutional neural networks; 2) An attention-based time pooling module that further exploits the attention in the time dimension; 3) A novel coarse-to-fine training scheme that first trains a network to roughly align the song chunks and then re- fines the network by training on the aligned chunks. At the same time, we also summarize some important training tricks used in our system that help achieve better results. Experi- ments on several standard CSI datasets show that our method significantly improves over state-of-the-art methods with an embedding size of 128 (2.3% on SHS100K-TEST and 17.7% on DaTacos).*

more code and model will be released soon.

- [ ] add main code
- [ ] add code about course-to-fine training
- [ ] release model mentioned in paper
- [ ] train model again for popular song with chinese, with dataset released by TME.




