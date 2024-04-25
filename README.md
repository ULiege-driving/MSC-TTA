#  Multi-Stream Cellular Test-Time Adaptation of Real-Time Models Evolving in Dynamic Environments

This repository provides the implementation of our paper (add link to paper), presented at CVPR WAC 2024.


<div align="center">
<div></div>
<img src="images/MSC-TTA-pipeline.png" width=auto% height=auto> 
</div>

## DADE Dataset

Please follow our other repository to download and install the [DADE dataset](https://github.com/ULiege-driving/DADE).

## Environment and preparation
We first need to create the Python environment:

```bash
conda create -y --name MSC python=3.9.18
conda activate MSC
```
Navigate inside the repository and install the required libraries:

```bash
cd MSC-TTA
pip3 install -r requirements.txt
```
Create symbolic links to the datasets, a storage folder for pretrained models and a storage folder to save confusion matrices :
```bash
cd MSC-TTA
ln -s path/to/DADE/ data
ln -s path/to/pretrained/weight/storage/ pretrained
ln -s path/to/confusion/matrices/storage/ confusion_matrices
```


## Pretraining

This step is not mandatory if you only want to adapt models from scratch. The `pretrain.sh` script shows an example to pretrain 7 models (one per location) on the two-first hours of DADE-static.

```bash
bash pretrain.sh
```

Similarly, you can call `pretraining_dynamic.py` to obtain pretrained models on DADE-dynamic. Please check the `--help` flag for more options.

## Adaptation

The `adapt.sh` script provide an example to adapt 7 models from scratch on the three-last hours of DADE-static. Per default, the configuration does not save confusion matrices and simply logs the accumulated mIoU over the adaptation period.
```bash
bash adapt.sh
```


## Citation

If you find this dataset useful in your research, please consider citing:

- the MSC-TTA paper: 
```bibtex
@inproceedings{Gerin2024MultiStream,
        title = {Multi-Stream Cellular Test-Time Adaptation of Real-Time Models Evolving in Dynamic Environments},
        author = {G\'erin, Beno{\^{\i}}t and Halin, Ana{\"\i}s and Cioppa, Anthony and Henry, Maxim and Ghanem, Bernard and Macq, Beno{\^{\i}}t and De Vleeschouwer, Christophe and Van Droogenbroeck, Marc},
        booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
        month = {June},
        year = {2024},
        address = {Seattle, Washington, USA}
}
```

- the DADE dataset:
```bibtex
@data{Halin2023DADE,
  author    = {Halin, Ana\"is and G\'erin, Beno\^it and Cioppa, Anthony and Henry, Maxim and Ghanem, Bernard and Macq, Beno\^it and De Vleeschouwer, Christophe and Van Droogenbroeck, Marc},
  publisher = {ULi\`ege Open Data Repository},
  title     = {{DADE dataset}},
  year      = {2023},
  version   = {V1},
  doi       = {10.58119/ULG/H5SP5P},
  url       = {https://doi.org/10.58119/ULG/H5SP5P}
}
```

## License
[CC-BY-4.0](https://github.com/ULiege-driving/DADE/blob/main/LICENSE)
