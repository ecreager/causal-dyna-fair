# causal-dyna-fair
Code accompanying the paper "Causal Modeling for Fairness in Dynamical Systems", presented at ICML 2020.

ArXiV: https://arxiv.org/abs/1909.09141

ICML results can be reproduced by `./bin/icml_results.sh`.

Package dependencies are specified in `requirements.txt`.
We strongly recommend using a fresh virtual environment and with packages installed via `pip install -r requirements.txt`.
Finally, we note that the `whynot` dependency may need to be installed from source as follows:
```
git clone git@github.com:zykls/whynot.git
cd whynot
pip install .
```