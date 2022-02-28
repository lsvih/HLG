# HLG.pytoch

Pytorch implementation of "Enhancing Chinese Pre-trained Language Model via Heterogeneous Linguistics Graph", ACL 2022

### Usage

All configuration are described in `args.py`:

```bash
python3 run_class.py --help
python3 run_match.py --help
```

### Preparatory

Please download pre-trained language models and configure the corresponding path in `args.py`.

### Train

```bash
python3 run_class.py --exp_name <exp name> --do_train
```

### About Dataset

Part of the processed datasets are uploaded in `dataset` folder. 
If you want to conduct experiments on other datasets, please follow the format of these datasets.

### About Graph Construction

Please refer to `collate` function in `dataloader/classifier.py` or `dataloader/matcher.py`.

### About Model and the Multi-Step Information Propagation

Please refer to `model.py`.

### About Trained Models/Checkpoints

Not released, will consider if necessary.
