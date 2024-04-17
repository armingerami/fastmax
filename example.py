import math
import torch
import fastmax_cuda
import numpy as np

class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, drop_noise, rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0):
        b = 0
        if len(q.shape) == 4:
          b = q.shape[0]
          q = q.reshape((q.shape[0]*q.shape[1],q.shape[2],q.shape[3])) # (b,h,n,d) -> (b*h,n,d)
          k = k.reshape((k.shape[0]*k.shape[1],k.shape[2],k.shape[3])) # (b,h,n,d) -> (b*h,n,d)
          v = v.reshape((v.shape[0]*v.shape[1],v.shape[2],v.shape[3])) # (b,h,n,d) -> (b*h,n,d)
          drop_noise = drop_noise.reshape((drop_noise.shape[0]*drop_noise.shape[1],drop_noise.shape[2],drop_noise.shape[3])) # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3: print("q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).")

        if rpe_matrix is None:
          print("Relative Positional Encoding must be given. Send a 2*n-1 by d matrix of all zeros if you don't want to use RPE.")

        q = q.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        k = k.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        v = v.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        drop_noise = drop_noise.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)

        o = fastmax_cuda.forwardpass(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperature)
        ctx.save_for_backward(q,k,v,o)
        ctx.mask = mask
        ctx.b = b
        ctx.t = temperatue
        o = o[:,:q.shape[1],:].permute(2,0,1).contiguous() # (n,d,b*h) -> (b*h,n,d)
        if b != 0: o = o.reshape((b,int(o.shape[0]/b),o.shape[1],o.shape[2])) # (b*h,n,d) -> (b,h,n,d)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q,k,v,o = ctx.saved_tensors
        mask = ctx.mask
        b = ctx.b
        t = ctx.t

        if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
        grad_output = grad_output.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,mask)

        gradq = gradq.permute(2,0,1).contiguous() # (n,d,b*h) -> (b*h,n,d)
        gradk = gradk.permute(2,0,1).contiguous() # (n,d,b*h) -> (b*h,n,d)
        gradv = gradv.permute(2,0,1).contiguous() # (n,d,b*h) -> (b*h,n,d)

        if(b != 0):
          gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
          gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
          gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        return gradq, gradk/t, gradv, None, None, None, None, None, None


class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(FASTMultiHeadAttention, self).__init__()

    def forward(self, q,k,v,drop_noise,rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperatue = 1.0):
        return FASTMultiHeadAttention_Function.apply(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperatue)
    

def rpe_matrix_creator(n, d, device, dtype, structured = True):
    """
    Creates the relative positional encoding matrix
    Inputs: (assuming query is a (b,h,n,d) or (b*h,n,d) tensor)
      - n (int): number of tokens
      - d (int): dimesion/channel per head
      - data type: must be torch.float32. This input is used to make sure the datatype used by the attention head is torch.float32.
      - Structured (bool): if True, produces sin/cos based RPE, and randomized matrx otherwise.
    Output:
      - rpe: a (2*n-1,d) matrix.
    """
    if(dtype != torch.float32): print("The data type must be float32 in order for Fastmax to work")
    if(structured):
        pe_positive = torch.zeros(n, d,device=device,dtype=dtype)
        pe_negative = torch.zeros(n, d,device=device,dtype=dtype)
        position = torch.arange(0, n, device=device,dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, device=device,dtype=dtype) * -(math.log(10000.0) / d))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        rpe = torch.cat([pe_positive, pe_negative], dim=0)
    else: rpe = torch.normal(0,1,size=(2*n-1,d),device=device,dtype=dtype)
    return rpe

# the inputs of fastmax are query, key, and value (q,k,v) in shape of  4-dimensional tensors (b, h, n, d); i.e. (batch, head, token length, dimension/channel per head)
fastmax = FASTMultiHeadAttention()

assert torch.cuda.is_available()
torch.set_default_device('cuda')
b = 16
h = 32
n = 4000
d = 64

q = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)
k = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)
v = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)

dtype = torch.float32
device = torch.device(0)

mask = True
dropout = 0.0 # between 0 and 1
normalize = False
temperatue = 1.0

rpe_matrix = rpe_matrix_creator(q.shape[-2],q.shape[-1],q.device,q.dtype)
drop_noise = torch.normal(0,1,size=(q.shape),dtype=q.dtype,device=q.device)
output = fastmax(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperatue)

