
## Revisiting Reverse Distillation for Anomaly Detection (CVPR 2023)

Code of CVPR 2023 paper: Revisiting Reverse Distillation for Anomaly Detection.

[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf)

<div align="center">

<br>
  <img width="100%" alt="AFA flowchart" src="./docs/method_training.png">
</div>

<!-- ## Abastract -->

The paper proposes RD++ approach for anomaly detection by enriching feature compactness and anomalous signal suppression through multi-task learning design. For feature compactness task, by introducing the self-supervised optimal transport method . For anomalous signal suppression task: by simulating pseudo-abnormal samples with simplex noise and minimizing the reconstruction loss. RD++ achieves a new state-of-the-art benchmark on the challenging MVTec dataset for both anomaly detection and localization. More importantly, Comparing with recent SOTA methods, RD++ runs 6.x faster than PatchCore, and 2.x faster than CFA but introduces a negligible latency compared to RD. 

<div align="center">

<br>
  <img width="100%" alt="AFA flowchart" src="./docs/inference_time.jpeg">
</div>

## Libraries
###   geomloss 
###   numba

## Data Preparations
Download MVTEC dataset from [[Link]](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Train
To start both training and evaluating results, for example: two classes: carpet, leather, please run:
```
python train_test.py --save_folder RD++  --classes carpet leather

```
Besides, the notebook file is provided for conveniently training, testing on google colab
### Evalution
If you just need to inference with checkpoints, please run
```
python inference_checkpoints.py --checkpoint_folder RD++  --classes carpet leather
```

## Citation
Please cite our paper if you find it's helpful in your work.

``` bibtex
@InProceedings{Tien_2023_CVPR,
    author    = {Tien, Tran Dinh and Nguyen, Anh Tuan and Tran, Nguyen Hoang and Huy, Ta Duc and Duong, Soan T.M. and Nguyen, Chanh D. Tr. and Truong, Steven Q. H.},
    title     = {Revisiting Reverse Distillation for Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24511-24520}
}
```

## Acknowledgement

We use [RD](https://github.com/hq-deng/RD4AD) as the baseline. Also, we use the [Simplex Noise](https://github.com/Julian-Wyatt/AnoDDPM). We are thankful to their brilliant works!


