# Fastmax
To run on the cluster, follows these steps:
1. copy the following 4 files into your project directory: fastmax_cuda.cpp, fastmax_cuda_backward.cu, fastmax_cuda_forward.cu, setup.py.
2. Create a virtual environment and run `pip install -r requirements`.
3. Open a session on a compute node with a GPU. `setup.py` requires GPU access in addition to `nvcc`.
2. Run the following linux commands in order (assuming you're already in your virtual env.): "module load cuda", "module load gcc", "python setup.py install" (to run "python setup.py install" you need 32gb DRAM memory).
3. You have now created a library named fastmax_cuda in your environment. To use fastmax in your transformer, follow the example.py file. In summary, copy line 1 to 278 in example.py on top of the python file you call multihead attention. To use Fast attention, replace wherever you call multihead attention with line 303 to 305.

**NOTE**: The size of batch $\times$ head must be larger than 32. Also, for the best performance have batch $\times$ head $\times$ dimension be either a multiple of the number of your cuda cores, or much larger than it. For example, if using RTX A6000, having batch = 64, head = 16, dimension = 32 is a good choice (or batch = 64, head = 8, dimension = 64).

# Parameters
**use_custom_gradient**: When instantiating, you can set whether to use eithr our CUDA code or the Pytorch version using a boolean flag: fastmax = FASTMultiHeadAttention(use_custom_gradient=True), where use_custom_gradient is the mentioned flag, and it's True by default.
**q, k, v**: Query, Key, and Value. Should be either 3-d (b*h,n,d) or 4-d (b,h,n,d), where b, h, n and d are batch size, number of heads, number of tokens and dimension per head. Output will have the same shape as q.
**drop_noise**: A matrix with the same shape as q that is used as an alternative to the dropout mechanism. Should not be touched.
**rpe_matrix**: The matrix containing the Relative Positional Encodings. It's created using the "rpe_matrix_creator". It's currently dissabled in the CUDA code since using it causes 3\times FLOPs increas, and 3\times slowdown as a result. I recommend using RoPE (rotary poaitional encoding) instead. If youre using a recent transformer code, it probably has RoPE implemented already, and you just need to give the q and k with the RoPE applied to the Fastmax function (TODO: add an example to show how to use RoPE).
**mask**: Boolean; indicates if causal mask should be applied.
**dropout rate**: Float between 0 and 1; controls the dropout rate.
**normalize**: Boolean; indicates if normalization should be applied. If True, q and k will be normalized to be confined within an eliptic with maximum radius "lim".
**temperature**: Float; scales the q.k prooduct to q.k/temperature. Default is 1.
**a0,a1,a2**: Float; controls the attention kernel. The kernel is a0+a1x+a2x^2. Default is 1, 1, 0.5, which is the second order Taylor expansion of exponential.
**lim**: Float; look at **normalize**.
**p**: int; should be either 1 or 2. If p = 1, the kernel is a0 + a1x, if p = 2, a0+a1x+a2x^2.


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
