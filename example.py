import math
import torch
from torch import cuda
import fastmax_cuda
import numpy as np
class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, drop_noise, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0, p=2):
        # print(q.get_device())
        if q.get_device() == -1: 
            ctx.save_for_backward(q,k,v,q)
            return q

        b = 0
        if len(q.shape) == 4:
            b = q.shape[0]
            q = q.reshape((q.shape[0]*q.shape[1],q.shape[2],q.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            k = k.reshape((k.shape[0]*k.shape[1],k.shape[2],k.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            v = v.reshape((v.shape[0]*v.shape[1],v.shape[2],v.shape[3])) # (b,h,n,d) -> (b*h,n,d)
            drop_noise = drop_noise.reshape((drop_noise.shape[0]*drop_noise.shape[1],drop_noise.shape[2],drop_noise.shape[3])) # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3: print("q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).")


        q = q.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        k = k.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        v = v.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        drop_noise = drop_noise.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)

        # device = "cuda:0"
        # q = torch.tensor(q, device = device)
        # k = torch.tensor(k, device = device)
        # v = torch.tensor(v, device = device)
        # drop_noise = torch.tensor(drop_noise, device = device)
        # print(torch.cuda.memory_allocated())
        
        o = fastmax_cuda.forwardpass(q,k,v,drop_noise,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
        # print(torch.cuda.memory_allocated())
        # print('a')
        ctx.save_for_backward(q,k,v,o)
        ctx.mask = mask
        ctx.p = p
        ctx.b = b
        ctx.t = temperature
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
        if q.get_device() == -1: 
            return q, q, q, None, None, None, None, None, None, None, None, None, None

        mask = ctx.mask
        p = ctx.p
        b = ctx.b
        t = ctx.t
        a0 = ctx.a0
        a1 = ctx.a1
        a2 = ctx.a2

        if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
        grad_output = grad_output.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,mask,a0,a1,a2,p)

        gradq = gradq.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradk = gradk.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)
        gradv = gradv.permute(1,0,2).contiguous() # (n,b*h,d) -> (b*h,n,d)

        if(b != 0):
          gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
          gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
          gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        
        return gradq, gradk/t, gradv, None, None, None, None, None, None, None, None, None, None
    

def fastmax_function(q, k, v, mask=0, dropout_rate = 0.0, normalize=0, temperature=1, a0=1,a1=1,a2=0.5,lim=1,p=2, create_attn_matrix = 0):
    """
    Input: query, key, and value matrices (b, h, n, d)
        b: batch size
        h: number of heads
        n: number of tokens
        d: dimension per attention head (d = d_model / h)
    mask: boolean indicating whether to apply causal masking
    temperature: Hyperparameter to control the standard deviation of <q, k>; stdev(<q, k>) = 1/temperature
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
            temperature = 1
            # q = q - torch.mean(q,dim = 3).unsqueeze(-1)
            # k = k - torch.mean(k,dim = 3).unsqueeze(-1)
            qn = torch.linalg.norm(q, dim = 3)
            kn = torch.linalg.norm(k, dim = 3)
            q = lim*q/torch.linalg.norm(qn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
            k = lim*k/torch.linalg.norm(kn, dim = 2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        else:
            temperature = temperature*math.sqrt(q.shape[3])
            temperature = 1
        temperature2 = temperature*temperature

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
                first_term = a0*torch.sum(v,-2)  # (b, h, d)

                second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

                third_term = a2*torch.matmul(k2.swapaxes(-2,-1),v)/temperature2  # (b, h, d^2, d)

                div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)
                div3 = a2*torch.sum(k2,-2).unsqueeze(-1) # (b, h, d^2, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                ans3 = torch.matmul(q2,third_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)
                div3 = torch.matmul(q2,div3)/(temperature2) # (b, h, n, 1)

                ans = ans2+ans3 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2+div3 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = a0*torch.cumsum(v,2) # (b, h, n, d)
                second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)
                third = a2*torch.einsum("bhij,bhijk -> bhik",[q2,torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k2,v]),2)])/temperature2 # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                k2cs = torch.cumsum(k2,-2) # (b, h, n, d^2)
                div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
                div3 = a2*torch.einsum("bhij,bhij -> bhi",[q2,k2cs])/temperature2 # (b, h, n)
                div = (div1 + div2 + div3).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second + third # (b, h, n, d)
                ans /= div # (b, h, n, d)
            
        # Taylor series with constant and linear terms:
        elif p == 1:
            drop_attn = torch.nn.Dropout(p=dropout_rate)
            k = drop_attn(k)
            q = drop_attn(q)
            if mask is None or not mask:
                first_term = a0*torch.sum(v,-2)  # (b, h, d)
                second_term = a1*torch.matmul(k.swapaxes(-2,-1),v)/temperature  # (b, h, d, d)

                div1 = a0*torch.ones([k.shape[0],k.shape[1],1,1], device=k.device)*k.shape[2] # (b, h, 1, 1)
                div2 = a1*torch.sum(k,-2).unsqueeze(-1) # (b, h, d, 1)

                ans2 = torch.matmul(q,second_term)  # (b, h, n, d)
                div2 = torch.matmul(q,div2)/(temperature) # (b, h, n, 1)

                ans = ans2 # (b, h, n, d)
                ans = torch.add(ans.permute(2,3,1,0) ,first_term.permute(2,1,0)).permute(3,2,0,1) # (b, h, n, d)
                div = div2 # (b, h, n, d)
                div = torch.add(div.permute(2,3,1,0) ,div1.permute(3,2,1,0)).permute(3,2,0,1) # (b, h, n, 1)
                ans = ans/div # (b, h, n, d)

            else:
                first = a0*torch.cumsum(v,2) # (b, h, n, d)
                second = a1*torch.einsum("bhij,bhijk -> bhik",[q, torch.cumsum(torch.einsum("bhij,bhik -> bhijk",[k,v]),2)])/temperature # (b, h, n, d)

                kcs = torch.cumsum(k,-2) # (b, h, n, d)
                div1 = a0*torch.cumsum(torch.ones([q.shape[0],q.shape[1],q.shape[2]], device=k.device),2) # (b, h, 1)
                div2 = a1*torch.einsum("bhij,bhij -> bhi",[q,kcs])/temperature # (b, h, n)
                div = (div1 + div2).unsqueeze(-1) # (b, h, n, 1)

                ans = first + second # (b, h, n, d)
                ans /= div # (b, h, n, d)
        
        else:
            raise ValueError(f"p must be 1 or 2, got: {p}")
        return ans

    else:
        # temperature = temperature*math.sqrt(q.shape[3])
        temperature2 = temperature*temperature

        k2 = k.unsqueeze(-1) @ k.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        k2 = k2.flatten(-2)                     # (b, h, n, d*d)
        q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)  # (b, h, n, d, 1) @ (b, h, n, 1, d) -> (b, h, n, d, d)
        q2 = q2.flatten(-2)    
        attn = a0 + a1*torch.matmul(q, torch.swapaxes(k, -2, -1))/temperature + a2*torch.matmul(q2, torch.swapaxes(k2, -2, -1))/temperature2
        if mask is not None:
            attn = torch.where(mask == 0, 0, attn)
        attn /= (torch.sum(attn, axis=3)).unsqueeze(-1)
        ans = torch.matmul(attn,v)
        return ans, attn

class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self, use_custom_gradient = True):
        super(FASTMultiHeadAttention, self).__init__()
        self.use_custom_gradient = use_custom_gradient

    def forward(self, q,k,v,drop_noise=None, mask = False, dropout = 0.0, normalize = False, temperature = 1.0, a0 = 1.0, a1 = 1.0, a2 = 0.5,lim = 1.0,p=2):
        if self.use_custom_gradient and mask: return FASTMultiHeadAttention_Function.apply(q,k,v,drop_noise,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
        else: return fastmax_function(q,k,v,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)

fastmax_custom = FASTMultiHeadAttention()

assert torch.cuda.is_available()
torch.set_default_device('cuda')
b = 2
h = 2
n = 3
d = 2

q = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)
k = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)
v = torch.normal(0,1,[b,h,n,d],device=torch.device('cuda'),requires_grad=True)

dtype = torch.float32
device = torch.device(0)

mask = False
dropout = 0.0 # between 0 and 1
normalize = True
temperature = 1.0
a0 = 1.0
a1 = 1.0
a2 = 0.5
lim = 1.0
p = 1

drop_noise = torch.normal(0,1,size=(q.shape),dtype=q.dtype,device=q.device).contiguous()
output = fastmax_custom(q,k,v,drop_noise,mask,dropout,normalize,temperature,a0,a1,a2,lim,p)
print(output)

