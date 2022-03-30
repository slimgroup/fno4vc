# Velocity continuation with Fourier neural operators

Code to reproduce results in [Velocity continuation with Fourier neural
operators for accelerated uncertainty
quantification](https://arxiv.org/abs/2203.14386).

## Installation

Run the commands below to install the required packages.

```bash
git clone https://github.com/slimgroup/fno4vc
cd fno4vc/
conda env create -f environment.yml # Use environment-cpu.yml for CPU only.
source activate fno4vc
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `source activate fno4vc`, the
following times.

## Questions

Please contact alisk@gatech.edu for further questions.

## Acknowledgements

We thank Zongyi Li for providing the code for the Fourier neural
operators at
[https://github.com/zongyi-li/fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator),
which we based our implementation on.

## Author

Ali Siahkoohi
