# NASCUP: Nucleic Acid Sequence Classification by Universal Probability

- This is the official github repository of  [NASCUP: Nucleic Acid Sequence Classification by Universal  Probability][paper].
- This repository is still under construction.

## Installation
You can install NASCUP by cloning our git repository and making executable files from the source codes:
```
git clone https://github.com/nascup/nascup.git
cd nascup
g++ -O2 -Wall -fopenmp -o nascupbuild train.cpp
g++ -O2 -Wall -fopenmp -o nascupscan test.cpp
```


## Usage
### nascupbuild
You can train context tree models of each group by running nascupbuild. For example, `./nascupbuild -i train.fasta -o model.ctm` creates `model.ctm` from `train.fasta`.

### nascupscan
You can classify query sequences based on context tree models by running nascupscan. For example, after running `./nascupscan -c model.ctm -i test.fasta -o result.out`, `result.out` includes labels of given queries in `test.fasta`.


## Data
For the datasets we used in the paper can be downloaded as follows:
```
wget https://data.snu.ac.kr/nascup/data/FILENAME.tar.gz
```
<FILENAME> are b_RF, b_RD, b_GG, b_SS, b_SL, b_AR, b_DV, 10BGG for RF, RD, GG, SS, SL, AR, DV, BGG dataset, respectively.


## Citing
If you use NASCUP in your research, please cite our paper:
```
@article{kwon2015nascup,
title={NASCUP: Nucleic Acid Sequence Classification by Universal Probability},
author={Kwon, Sunyoung and Kim, Gyuwan and Lee, Byunghan and Yoon, Sungroh and Kim, Young-Han},
journal={arXiv preprint arXiv:1511.04944},
year={2015}
}
```
We are preparing an updated version of the paper. Stay tuned!


## Contact
- For paper-related inquires, please contact authors of the paper.
- For code-related issues, please create a new issue or contact [Gyuwan Kim][gyuwan] ([@kgwmath][gyuwan-github])


[paper]: https://arxiv.org/abs/1511.04944
[gyuwan]:  mailto:kgwmath@gmail.com
[gyuwan-github]:  https://github.com/kgwmath
