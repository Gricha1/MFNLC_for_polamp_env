# Download weights


# Docker Install

```commandline
cd docker
sh build.sh
sh start.sh
```


# Train SPEIS
## point
```commandline
cd exps
sh train_speis_point.sh
```
## doggo
```commandline
cd exps
sh train_speis_doggo.sh
```

# start tensorboard
tensorboard --logdir mfnlc/mfnlc_data/ --bind_all