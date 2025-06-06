# QM9

Download the QM9 dataset with

```bash
python qm9_dataset.py
```


## Training
To train the model in the paper, run this command:

```bash
python qm9_run.py --model SE3-Hyena --num_epochs 100 --num_degrees 4 --num_layers 7 --num_channels 32 --name qm9-homo --num_workers 4 --batch_size 32 --task homo --div 2 --pooling max --head 8
```

## Evaluation

To evaluate a pretrained model, run:

```bash
python qm9_eval.py --model SE3-Hyena --num_degrees 4 --num_layers 7 --num_channels 32 --name qm9-homo --num_workers 4 --batch_size 32 --task homo --div 2 --pooling max --head 8 --restore <path-to-model>
```

## Pre-trained Models

- [ ] TODO


