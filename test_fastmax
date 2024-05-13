import math
import torch
from torch import cuda
import fastmax_cuda
import numpy as np
import time

class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, drop_noise, rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0):
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

        # q = q.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        # k = k.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        # v = v.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        q = q.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        k = k.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        v = v.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        drop_noise = drop_noise.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        # print(torch.cuda.memory_allocated())
        o = fastmax_cuda.forwardpass(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperature,a0,a1,a2,lim)
        # print(torch.cuda.memory_allocated())
        # print('a')
        ctx.save_for_backward(q,k,v,o)
        ctx.mask = mask
        ctx.b = b
        ctx.t = temperatue
        ctx.a0 = a0
        ctx.a1 = a1
        ctx.a2 = a2
        o = o[:,:,:q.shape[2]]
        o = o.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        if b != 0: o = o.reshape((b,int(o.shape[0]/b),o.shape[1],o.shape[2])) # (b*h,n,d) -> (b,h,n,d)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q,k,v,o = ctx.saved_tensors
        mask = ctx.mask
        b = ctx.b
        t = ctx.t
        a0 = ctx.a0
        a1 = ctx.a1
        a2 = ctx.a2

        if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
        grad_output = grad_output.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,mask,a0,a1,a2)

        gradq = gradq.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradk = gradk.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradv = gradv.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)

        if(b != 0):
          gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
          gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
          gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        
        return gradq, gradk/t, gradv, None, None, None, None, None, None, None, None, None, None


class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(FASTMultiHeadAttention, self).__init__()

    def forward(self, q,k,v,drop_noise,rpe_matrix = None, mask = False, dropout = 0.0, normalize = False, temperatue = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0):
        return FASTMultiHeadAttention_Function.apply(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperatue,a0,a1,a2,lim)
    

def rpe_matrix_creator(n, d, device, dtype, structured = True, is_zero = False):
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
    else: 
        if is_zero:
            rpe = torch.zeros(size=(2*n-1,d),device=device,dtype=dtype)
        else:
            rpe = torch.normal(0,1,size=(2*n-1,d),device=device,dtype=dtype)
    return rpe


def fastmax(q, k, v, mask=0, denum_Term=1, normalize=2, p=2, create_attn_matrix = 0, dropout_rate = 0.0):
    """
    Input: query, key, and value matrices (b, h, n, d)
        b: batch size
        h: number of heads
        n: number of tokens
        d: dimension per attention head (d = d_model / h)
    mask: boolean indicating whether to apply causal masking
    denum_term: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/denum_term
        Stdev of <q, k> is important in general with attention, but even more so when using a taylor
        expansion to approximate an exponential because the error increases with the stdev of <q, k>.
        In normal attention, stdev equates to the "temperature" of the softmax function, and with a
        taylor approximation, higher temperature also means we drift further from the true softmax.
        For positive inputs, this drifting error actually lowers the temperature, and for negative inputs
        it raises the temperature.
    Output: The result of Attention matrix * Value (b, h, n, d)
    """
    if create_attn_matrix == 0:
        if normalize == 1:
            denum_term = 1
            # q = q - torch.mean(q,dim = 3).unsqueeze(-1)
            # k = k - torch.mean(k,dim = 3).unsqueeze(-1)
            qn = torch.linalg.norm(q, dim = 3)
            kn = torch.linalg.norm(k, dim = 3)
            q = q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
            k = k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        else:
            denum_term = denum_Term*math.sqrt(q.shape[3])
            denum_term = 1
        denum_term2 = 2*denum_term*denum_term

        # Prepare the quadratic terms with respect to k and q:
        if p == 2:
            # Prepare the quadratic terms with respect to k and q:
            k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            k2 = k2.flatten(-2)                     # (b, h, n, d*d)
            q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
            q2 = q2.flatten(-2)                     # (b, h, n, d*d)
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k2 = drop_attn(k2)
            q2 = drop_attn(q2)

            if mask == 0:
                first_term = torch.sum(v,-2)  # (b, h, d)

                second_term = torch.matmul(k.swapaxes(-2,-1),v)/denum_term  # (b, h, d, d)

                third_term = torch.matmul(k2.swapaxes(-2,-1),v)/denum_term2  # (b, h, d^2, d)

                div1 = torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)
                div3 = torch.sum(k2,-2).unsqueeze(-1) # (b, h, d^2, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                ans3 = torch.matmul(q2,third_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(denum_term) # (b, h, n, 1)
                div3 = torch.matmul(q2,div3)/(denum_term2) # (b, h, n, 1)

                ans = ans2+ans3 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2+div3 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = torch.cumsum(v,2) # (b, h, n, d)
                second = torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/denum_term # (b, h, n, d)
                third = torch.einsum("bhij,bhijk -> bhik",[q2,torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k2,v]),2)])/denum_term2 # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                k2cs = torch.cumsum(k2,-2) # (b, h, n, d^2)
                div1 = torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = torch.einsum("bhij,bhij -> bhi",[q,kcs])/denum_term # (b, h, n)
                div3 = torch.einsum("bhij,bhij -> bhi",[q2,k2cs])/denum_term2 # (b, h, n)
                div = (div1 + div2 + div3).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second + third # (b, h, n, d)
                ans /= div # (b, h, n, d)
            
        # Taylor series with constant and linear terms:
        elif p == 1:
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k = drop_attn(k)
            q = drop_attn(q)
            if mask is None or not mask:
                first_term = torch.sum(v,-2)  # (b, h, d)
                second_term = torch.matmul(k.swapaxes(-2,-1),v)/denum_term  # (b, h, d, d)

                div1 = torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(denum_term) # (b, h, n, 1)

                ans = ans2 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = torch.cumsum(v,2) # (b, h, n, d)
                second = torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/denum_term # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                div1 = torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = torch.einsum("bhij,bhij -> bhi",[q,kcs])/denum_term # (b, h, n)
                div = (div1 + div2).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second # (b, h, n, d)
                ans /= div # (b, h, n, d)
        
        else:
            raise ValueError(f"p must be 1 or 2, got: {p}")
    
    else:
        a = 0

    # else:
    #     denum_term = denum_term*math.sqrt(q.shape[3])
    #     denum_term2 = 2*denum_term*denum_term

    #     k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
    #     k2 = k2.flatten(-2)                     # (b, h, n, d*d)
    #     q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
    #     q2 = q2.flatten(-2)    
    #     attn = 1 + torch.matmul(q, torch.swapaxes(k, -2, -1))/denum_term + torch.matmul(q2, torch.swapaxes(k2, -2, -1))/denum_term2
    #     if mask is not None:
    #         attn = torch.where(mask == 0, 0, attn)
    #     attn /= (torch.sum(attn, axis=3)).unsqueeze(-1)
    #     ans = torch.matmul(attn,v)
    return ans




# the inputs of fastmax are query, key, and value (q,k,v) in shape of  4-dimensional tensors (b, h, n, d); i.e. (batch, head, token length, dimension/channel per head)

fastmax_custom = FASTMultiHeadAttention()
torch.no_grad()

def softmax(q,k,v):
  A = q @ k.mT
  A /= 8*math.sqrt(q.shape[0])
  A = torch.softmax(A, dim=-1)
  ans = A @ v
  # print(ans.shape)
  return ans

# assert torch.cuda.is_available()
# torch.set_default_device('cuda')

# Leslie look here
b = 20 # batch
h = 12 # head
d = 32 # dimension per head (i.e. model dimension/h)
# n changes from 10^strt to 10^endd. The number of test points are count
count = 1
strt = 3 # log scale
endd = 3 # log scale


soft_time = np.zeros(count)
fast_time = np.zeros(count)
fast_time_custom = np.zeros(count)
dtype = torch.float32
device = torch.device(0)
mask = False
dropout = 0.0 # between 0 and 1
normalize = True
temperatue = 1.0
rep = 20

a0 = 1.0
a1 = 1.0
a2 = 0.5
lim = 1.0

j = -1
for i in np.logspace(strt, endd, count):
  j += 1
  print(int(i))
  for ii in range(rep):
    torch.cuda.empty_cache()
    q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
    k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
    v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
    rpe_matrix = rpe_matrix_creator(k.shape[-2],q.shape[-1],q.device,q.dtype,structured = True,is_zero = False)
    drop_noise = torch.normal(0,1,size=(q.shape),dtype=q.dtype,device=q.device)
    start_time = time.time()
    e = fastmax_custom(q,k,v,drop_noise,rpe_matrix,mask,dropout,normalize,temperatue,a0,a1,a2,lim)
    cuda.synchronize()
    end_time = time.time()
    fast_time_custom[j] += (end_time - start_time)/rep


rep = 100
# # j = -1
# # for i in np.logspace(strt, endd, count):
# #   j += 1
# #   print(int(i))
# #   if(i > 20000): continue
# #   for ii in range(rep):
# #     torch.cuda.empty_cache()
# #     q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
# #     k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
# #     v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))

# #     start_time = time.time()
# #     e = softmax(q,k,v)
# #     cuda.synchronize()
# #     end_time = time.time()
# #     soft_time[j] += (end_time - start_time)/rep

print("############################################")
j = -1
for i in np.logspace(strt, endd, count):
  if(i > 60000): continue
  j += 1
  print(int(i))
  for ii in range(rep):
    torch.cuda.empty_cache()
    q = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
    k = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))
    v = torch.normal(0,1,[b,h,int(i),d],device=torch.device('cuda'))

    start_time = time.time()
    e = fastmax(q,k,v)
    cuda.synchronize()
    end_time = time.time()
    fast_time[j] += (end_time - start_time)/rep

# print("softmax = \n")
# print(soft_time)
print("fastmax with custom gradient = \n")
print(fast_time_custom)
print("fastmax = \n")
print(fast_time)

