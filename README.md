# Fastmax
To run on the cluster, follows these steps:
1. copy the following 4 files into your project directory: fastmax_cuda.cpp, fastmax_cuda_backward.cu, fastmax_cuda_forward.cu, setup.py.
2. Create a virtual environment and run `pip install -r requirements`.
3. Open a session on a compute node with a GPU. `setup.py` requires GPU access in addition to `nvcc`.
2. Run the following linux commands in order (assuming you're already in your virtual env.): "module load cuda", "module load gcc", "python setup.py install" (to run "python setup.py install" you need 32gb DRAM memory).
3. You have now created a library named fastmax_cuda in your environment. To use fastmax in your transformer, follow the example.py file. In summary, copy line 1 to 93 in example.py on top of the python file you call multihead attention. To use Fast attention, replace wherever you call multihead attention with line 113 to 115.

**NOTE**: The size of batch $\times$ head must be larger than 32. Also, for the best performance have batch $\times$ head $\times$ dimension be either a multiple of the number of your cuda cores, or much larger than it. For example, if using RTX A6000, having batch = 64, head = 16, dimension = 32 is a good choice (or batch = 64, head = 8, dimension = 64).

# Some forward pass numbers on RTX A6000
B = 64, H = 8, N = 1000, D = 32, P = 2:
- With custom gradient (my CUDA code): 0.11s
- With Pytorch: 0.08s

B = 64, H = 8, N = 10000, D = 32, P = 2:
- With custom gradient (my CUDA code): 1.09s
- With Pytorch: 1.26s
  
B = 64, H = 16, N = 1000, D = 32, P = 2:
- With custom gradient (my CUDA code): 0.13s
- With Pytorch: OOM

B = 64, H = 16, N = 10000, D = 32, P = 2:
- With custom gradient (my CUDA code): 1.28s
- With Pytorch: OOM

B = 64, H = 8, N = 1000, D = 64, P = 2:
- With custom gradient (my CUDA code): 0.32s
- With Pytorch: OOM

B = 64, H = 8, N = 10000, D = 64, P = 2:
- With custom gradient (my CUDA code): 3.24s
- With Pytorch: OOM

B = 64, H = 16, N = 1000, D = 64, P = 2:
- With custom gradient (my CUDA code): 0.65s
- With Pytorch: OOM

B = 64, H = 16, N = 10000, D = 64, P = 2:
- With custom gradient (my CUDA code): 6.66s
- With Pytorch: OOM
