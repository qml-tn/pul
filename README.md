# Positive unlabeled learning with tensor networks

The main code is in the `tnpul` dir and the executable programs should be placed in the `bin` dir. 

Right now we are using `torchmps` and `tensorgrad` packages. Which are placed in the respective directories inside the `tnpul` folder.

The `papers` folder contains descriptions of papers with the instructions how to reproduce the results.

You can reproduce the results of the paper by first installing the package with `pip install -e .` and then running for example the following code (for a single run)

```python pul.py --D=20 --S=10 --alpha1=1 --alpha_max=10 --alpha_min=0.1 --beta1=4 --beta2=4. --beta3=1. --beta4=2. --beta5=2. --beta6=1. --beta7=4 --beta8=0 --labp=0.5 --bs=1024 --crop=20 --d=6 --leps=10 --lr=0.01 --nepoch=500 --np=100 --p=0.5 --step=500 --stop_patience=500 --angle=0.05 --scale=0.2 --positive_class=0 --negative_class=-1 --verbose=0 --wandb=0 --alpha3=0 --ninds=0 --nshuffle=0```

Many of the flags are not used in the pul paper and should be set to default values or as in the example above.

If you have WANDB you can set the `wandb` flag to 1. 

In case you are using `wandb_offline=1` flag you also have to specify the `WANDB_API_KEY`.