# Fastmax
To run on the cluster, follows these steps:
1. copy the following 4 files into your project directory: fastmax_cuda.cpp, fastmax_cuda_backward.cu, fastmax_cuda_forward.cu, setup.py.
2. Run the following linux commands in order (assuming you're already in your virtual env.): "module load cuda", "module load gcc", "python setup.py install" (to run "python setup.py install" you need 32gb DRAM memory).
3. You have now created a library named fastmax_cuda in your environment. To use fastmax in your transformer, follow the example.py file. In summary, copy line 1 to 93 in example.py on top of the python file you call multihead attention. To use Fast attention, replace wherever you call multihead attention with line 113 to 115.

**NOTE**: The size of batch*head must be larger than 32. Also, for the best performance have batch*head*dimension be either a multiple of the number of your cuda cores, or much larger than it. For example, if using RTX A6000, having batch = 32, head = 16, dimension = 64 is a good choice (or batch = 16, head = 32, dimension = 64).
