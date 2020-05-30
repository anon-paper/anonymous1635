# Faster and Unsupervised Neural Architecture Evolution for Visual Representation Learning

Unsupervised visual representation learning is one of the  hottest topics in computer vision, which still lag behind on the performance when compared with the dominant supervised learning methods. At the same time, neural architecture search (NAS) has produced state-of-the-art results in  various visual tasks. It is a natural idea to explore NAS to improve the unsupervised representation learning, which however remains largely unexplored. In this paper, we propose faster and unsupervised neural architecture evolution (FU-NAE) to evolve an existing architecture manually designed or searched in one small dataset to a new architecture on another large dataset.  This partial optimization can utilize the prior knowledge to reduce search cost and improve search efficiency. The evolution is self-supervised where contrast loss is used as the evaluation metric in the student-teacher framework. The evolution process is  significantly accelerated 
by eliminating  the inferior or the least promising operation. Experimental results show that we achieve the state-of-the-art performance for the downstream applications, such as object recognition, object detection and instance segmentation.

Here we provide our test codes and pretrained models.

## Requirements

- python 3.6
- PyTorch 1.0.0

## Run examples
You need to modified your path to dataset using ```--data_path```.

To evaluate the model in **ImageNet**, just run

```bash
sh script/validation.sh
```

