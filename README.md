# HagPF: Hierarchical-annotation-guided Phenotypic Framework for Stem Instance Segmentation and Length Measurement in plant point clouds

## Introduction
As a critical structural component that connects almost all other types of plant organs, the stem system not only supports the weight of the total plant, but also serves as a vital channel for nutriment transportation. Accurate phenotypic measurement of stem instances is of practical significance for assessing crop growth dynamics and predicting yield. To address current 3D phenotyping challenges of crops such as the difficulty in separating stem segments from the stem system and the low accuracy in stem length measurement, we propose a Hierarchical-annotation-guided Phenotypic Framework (HagPF) for Stem Instance Segmentation and Length Measurement in plant point clouds. 

## Key results
On a 3D dataset comprising four crop species, the proposed framework achieved an Intersection over Union (IoU) of 95.45% for organ semantic segmentation and a Mean Weighted Coverage (mWcov) of 87.87% for instance segmentation (both stem and leaf). Regarding to the stem length measurement, the method obtained an average Root Mean Square Error (RMSE) of 1.044 cm and a relative error of 11.907%, outperforming 7 mainstream methods. We believe that HagPF will shed new light on the field of 3D crop stem phenotyping. 

## Dataset
We invented a novel leaf-stem organ instance annotation strategy to label semantics and instances for both leaf and stem organs from crop point clouds. And the dataset used is available at Zenodo:(file: data.zip) (https://doi.org/10.5281/zenodo.19564655) contains 663 sets of colourless point cloud data for four dicotyledonous crops (104 tobaccos, 310 tomatoes, 168 peppers, and 81 soybean point clouds).

All raw point clouds are represented in the form of txt files. Each txt file represents a 3D plant. Each line of the txt file represents a point in the point cloud. Each txt file contains 5 columns, of which the first three columns show the "xyz" spatial information, the fourth column is the instance label, and the fifth column is the semantic label.

## Quick Start

### Prerequisites

```bash
# Clone this repository
git clone https://github.com/Jinx00/stem-length-measurement.git
cd stem-length-measurement
```


## Overall Pipeline
Our workflow comprises of the stage of 3D organ instance segmentation clouds and the stage of shape-adaptive stem length measurement. 

The core functionality of this repository focuses on stem length measurement, which relies on high-quality stem instance segmentation results in the stage of 3D organ instance segmentation clouds. Here we use the PSegNet, which is a deep learning network specifically designed for plant point clouds, capable of performing both semantic segmentation (distinguishing stems from leaves) and instance segmentation (distinguishing individual leaves). The official code and pre-trained models for PSegNet can be found in the following repository:
https://github.com/Huang2002200/PlantNet-and-PSegNet/

## Script Descriptions
The following scripts are included in this repository, and they should be executed in order.

First, run 00 only retain the stem part based on semantic tags.py. This script keeps only the stem points based on semantic labels and removes leaf points. Then run 01 ins tags are aligned with gt.py to align the predicted stem instance labels with the ground truth labels, ensuring correct error calculation.Next, execute 02 complete processing after replacing the SOM experiment.py, which is the main program for stem length calculation. Because the point cloud data may have undergone scaling changes during preprocessing or network output, you need to first compute the scaling factor: run 03-1 calculate the scaling face.py to calculate the scaling factor of the point cloud data, then run 03-2 scale the true value.py to scale the ground truth length values accordingly. Finally, run 04 error result calculation.py to compute evaluation metrics such as Root Mean Square Error (RMSE) and relative error.


