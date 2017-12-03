# Mask RCNN in PyTorch
**Unfortunately, we could not fit the model into the GPU we have and there is some ambiguity in the paper as well, so we decided to stop the project and wait until the official code being released.**


1. This is a [PyTorch](https://github.com/pytorch/pytorch) implementation of [Mask RCNN](https://arxiv.org/abs/1703.06870) which attempts to reproduce the results in [Mask RCNN](https://arxiv.org/abs/1703.06870).
2. This project provides an implementation of ROI Align in CUDA C as well as a PyTorch nn.Module for it.
3. The model architecture is based on the awesome [Faster RCNN with PyTorch](https://github.com/longcw/faster_rcnn_pytorch) repo.
4. The experiment setup is based on [Image Classification Project Killer in Pytorch](https://github.com/felixgwu/img_classification_pk_pytorch) to minimize the effort of doing experiments and developing new models.

Spectial thanks to [Fast Mask RCNN](https://github.com/CharlesShang/FastMaskRCNN) for being the catalyst of this project.

## Progress (with expected time)
- [x] ROIAlign layer (It is not explained in the paper how four regular locations are chosen, so currently, this implementation uses 4 corners of each bin. This may be changed either after hyper-parameter tuning stage or after the release of the original code.)
- [x] COCO dataloader with mask
- [x] FastRCNN with ROIAlign using ResNet-50-C4
- [x] Training code
- [ ] Test the FastRCNN with ROI Algin (In progress)
- [ ] MaskRCNN using ResNet-50-C4
- [ ] FPN backbone
- [ ] Testing all code
- [ ] Turing hyper-parameters
- [ ] Considering removing dependency of numpy and cython
- [ ] Speed up
- [ ] Clean the code and comment
- [ ] Pretrained models
- [ ] Demo code
