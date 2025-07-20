# BanditPam multithreading implementation

The cost functions of both built and swap step is parallelized with multithreading

## setup

code was tested with python3.8

To Install the packages

```bash
pip install -r requirements.txt
```
we can also use docker to run ```Dockerfile``` is attached

## Usage

```python
python bandit_pam_mt_batch.py -k 5 -N 1000  -m L2 -d random -w 2
```
- k is number of medoids
- N is number of samples to be passed
- m metric to calculate distance
- d dataset whether ```random``` or ```mnist```
- w number of workers runs in parellel