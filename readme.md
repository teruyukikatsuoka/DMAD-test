# Statistical Test for Generated Hypotheses by Diffusion Model
This pacakge is the implementation of the paper "Statistical Test for Generated Hypotheses by Diffusion Model" for experiments.

## Installation & Requirements
This pacakage has the following dependencies:
- Python (version 3.10 or higher, we use 3.10.13)
- sicore (we use version 1.0.0)
- tensorflow (we use version 2.15.0)
- scipy (we use version 1.11.4)
- numpy (we use version 1.26.2)
- omegaconf (we use version 2.3.0)
- tqdm

Please install these dependencies by pip.
```
pip install sicore
pip install tensorflow
pip install scipy
pip install numpy
pip install omegaconf
pip install tqdm
pip install matplotlib
```

## Reproducibility

Since we have already got the results in advance, you can reproduce the figures by running following code by running plot.ipynb. The results will be saved in "/results/fig" folder.

To reproduce the results, please see the following instructions after installation step.
The results will be saved in "./results" folder as pickle file.

For indipendence (type I error rate experiment).
Before execute the command, please set data/category to `iid` in config.yml
```
sh fpr1.sh
sh fpr2.sh
sh fpr3.sh
sh fpr4.sh
```

For indipendence (power experiment).
Before execute the command, please set data/category to `iid` in config.yml
```
sh power1.sh
sh power2.sh
sh power3.sh
sh power4.sh
```

For correlation (type I error rate experiment).
Before execute the command, please set data/category to `corr` in config.yml
```
sh fpr1.sh
sh fpr2.sh
sh fpr3.sh
sh fpr4.sh
```

For correlation (power experiment).
Before execute the command, please set data/category to `corr` in config.yml
```
sh power1.sh
sh power2.sh
sh power3.sh
sh power4.sh
```
