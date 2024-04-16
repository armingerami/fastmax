import math
import torch
from torch import cuda
# import fastmax_cuda_double as fastmax_cuda
import fastmax_cuda
import numpy as np
import time
from torch.autograd import gradcheck
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q,k,v, rpe_matrix = None, mask = False, normalize = False):
        # q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).
        if normalize:
            # q = q - torch.mean(q,dim = 3).unsqueeze(-1)
            # k = k - torch.mean(k,dim = 3).unsqueeze(-1)
            qn = torch.linalg.norm(q, dim = -1)
            kn = torch.linalg.norm(k, dim = -1)
            q = q/torch.linalg.norm(qn, dim = -2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
            k = k/torch.linalg.norm(kn, dim = -2, ord = float('inf')).unsqueeze(-1).unsqueeze(-1)
        else:
            # k /= (math.sqrt(q.shape[-1]))
            k = k

        b = 0
        if len(q.shape) == 4:
          b = q.shape[0]
          q = q.reshape((q.shape[0]*q.shape[1],q.shape[2],q.shape[3])) # (b,h,n,d) -> (b*h,n,d)
          k = k.reshape((k.shape[0]*k.shape[1],k.shape[2],k.shape[3])) # (b,h,n,d) -> (b*h,n,d)
          v = v.reshape((v.shape[0]*v.shape[1],v.shape[2],v.shape[3])) # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3: print("q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d).")

        if rpe_matrix is None:
        #   rpe_matrix = torch.zeros((2*k.shape[-2]-1, k.shape[-1]),dtype = q.dtype,device="cuda:0")
          print("Relative Positional Encoding must be given. Send a 2*n-1 by d matrix of all zeros if you don't want to use RPE.")

        q = q.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        k = k.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        v = v.permute(1,0,2).contiguous() # (b*h,n,d) -> (n,b*h,d)
        # o_in = torch.zeros((q.shape[0],q.shape[1],q.shape[2]+1),dtype = q.dtype).contiguous()
        # cons = torch.zeros((q.shape[1],q.shape[2]+1),dtype = q.dtype).contiguous()
        # lin = torch.zeros((q.shape[1],q.shape[2]+1,q.shape[2]),dtype = q.dtype).contiguous()
        # quad = torch.zeros((q.shape[1],q.shape[2]+1,q.shape[2],q.shape[2]),dtype = q.dtype).contiguous()

        # o = fastmax_cuda.forwardpass(q,k,v,o_in,rpe_matrix,cons,lin,quad,mask)
        o, lin, quad = fastmax_cuda.forwardpass(q,k,v,rpe_matrix,mask)
        ctx.save_for_backward(q,k,v,o,lin,quad)
        ctx.mask = mask
        ctx.b = b

        o = o[:,:,:q.shape[-1]].permute(1,0,2).contiguous()
        if b != 0: o = o.reshape((b,int(o.shape[0]/b),o.shape[1],o.shape[2]))
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q,k,v,o,lin,quad = ctx.saved_tensors
        mask = ctx.mask
        b = ctx.b
        # gradq = torch.zeros((q.shape[0],q.shape[1],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradk = torch.zeros((q.shape[0],q.shape[1],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradv = torch.zeros((q.shape[0],q.shape[1],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradk_coeffs0v = torch.zeros((q.shape[0],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradk_coeffs0o = torch.zeros((q.shape[0],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradk_coeffs1v = torch.zeros((q.shape[0],q.shape[2],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradk_coeffs1o = torch.zeros((q.shape[0],q.shape[2],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradv_coeffs0 = torch.zeros((q.shape[0],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradv_coeffs1 = torch.zeros((q.shape[0],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradv_coeffs2 = torch.zeros((q.shape[0],q.shape[2],q.shape[2],q.shape[2]),dtype = q.dtype, device = q.device)
        # gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,gradq,gradk,gradv,lin,quad,gradk_coeffs0v,gradk_coeffs0o,gradk_coeffs1v,gradk_coeffs1o,gradv_coeffs0,gradv_coeffs1,gradv_coeffs2,mask)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print(o.shape)
        # print(grad_output.shape)
        # print(lin.shape)
        # print(quad.shape)
        if(b != 0): grad_output = grad_output.reshape((grad_output.shape[0]*grad_output.shape[1],grad_output.shape[2],grad_output.shape[3])).contiguous()
        grad_output = grad_output.permute(1,0,2).contiguous()
        gradq, gradk, gradv = fastmax_cuda.backwardpass(q,k,v,o,grad_output,lin,quad,mask)
        # print("gradk = ", gradk)
        # print(gradq.shape)
        # print(gradk.shape)
        # print(gradv.shape)
        gradq = gradq.permute(1,0,2).contiguous()
        gradk = gradk.permute(1,0,2).contiguous()
        gradv = gradv.permute(1,0,2).contiguous()
        # print(gradq.shape)
        # print(gradk.shape)
        # print(gradv.shape)
        if(b != 0):
          gradq = gradq.reshape((b,int(gradq.shape[0]/b),gradq.shape[1],gradq.shape[2])).contiguous()
          gradk = gradk.reshape((b,int(gradk.shape[0]/b),gradk.shape[1],gradk.shape[2])).contiguous()
          gradv = gradv.reshape((b,int(gradv.shape[0]/b),gradv.shape[1],gradv.shape[2])).contiguous()
        # print(gradq.shape)
        # print(gradk.shape)
        # print(gradv.shape)
        return gradq, gradk, gradv, None, None, None


class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(FASTMultiHeadAttention, self).__init__()

    def forward(self, q,k,v,rpe_matrix = None, mask = False, normalize = False):
        return FASTMultiHeadAttention_Function.apply(q,k,v,rpe_matrix,mask,normalize)
    


def fastmax(q, k, v, mask=0, denum_Term=1, normalize=0, p=2, create_attn_matrix = 0, dropout_rate = 0):
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

            if mask is None or not mask:
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
mask = 0
p = 2
fastmax_custom = FASTMultiHeadAttention()


device = torch.device(0)
torch.cuda.set_device(0)

b = 1
h = 1
n = 1
d = 1


dtype = torch.float32
rpe_matrix = torch.zeros((2*n-1,d),dtype = dtype,device=torch.device(device))

q = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d)).contiguous()
k = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d)).contiguous()
v = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d)).contiguous()
q = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device)).contiguous()
k = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device)).contiguous()
v = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device)).contiguous()
# v = torch.ones((b,h,n,d), dtype = torch.float32,device=torch.device(device),requires_grad=True)
o = fastmax(q,k,v)
print(o)
o = fastmax_custom(q,k,v,rpe_matrix)
print("CUSTOM##################")
print(o)

# # o = fast_grad(q,k,v)
# # print(o)
print("##################")
print("##################")
print("##################")
print(torch.autograd.functional.jacobian(fastmax, (q,k,v)))
print("CUSTOM##################")
print(torch.autograd.functional.jacobian(fastmax_custom, (q,k,v,rpe_matrix))[:3])

# print(torch.autograd.functional.jacobian(fastmax, (q,k,v)))
# print("###########################################################")
# print(torch.autograd.functional.jacobian(fastmax_custom, (q,k,v)))
# print("###########################################################")
# print(torch.autograd.functional.jacobian(fast_grad, (q,k,v)))

# print(torch.autograd.functional.jacobian(fastmax, (q,k,v)))
# q = torch.arange(b*h*n*d, dtype = torch.double,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# k = torch.arange(b*h*n*d, dtype = torch.double,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# v = torch.arange(b*h*n*d, dtype = torch.double,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# # testt = gradcheck(fastmax, (q, k, v), eps=1e-3, atol=1e-2)
# # print(testt)

# print("################################################################")
# q = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# k = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# v = torch.arange(b*h*n*d, dtype = torch.float32,device=torch.device(device),requires_grad=True).reshape((b,h,n,d))
# o = fastmax_custom(q,k,v)
# print(o)
# print(torch.autograd.functional.jacobian(fastmax_custom, (q,k,v)))

#####################################################
#####################################################
#####################################################
# dtype = torch.double
# rpe_matrix = torch.zeros((2*n-1,d),dtype = dtype,device=torch.device(device))
# q = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device))
# k = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device))
# v = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True,device=torch.device(device))
# # o = torch.randn(b,h,n,d,dtype=dtype,requires_grad=True)
# # denum = torch.randn(b,h,n,dtype=dtype,requires_grad=True)
# print(gradcheck(fastmax_custom,(q,k,v,rpe_matrix),eps=1e-3, atol=1e-2))


# def softmax(q,k,v):
#   A = q @ k.mT
#   A /= 8*math.sqrt(q.shape[0])
#   A = torch.softmax(A, dim=-1)
#   ans = A @ v
#   # print(ans.shape)
#   return ans

# b = 1
# h = 1
# n = 100
# d = 64

# count = 1
# rep = 10

# strt = 3
# endd = 3

# soft_time = np.zeros(count)
# fast_time = np.zeros(count)
# fast_time_custom = np.zeros(count)

# mask = torch.tensor(False,device=torch.device(device))
# p = torch.tensor(2, dtype = torch.int32,device=torch.device(device))
# mask = False
# p = 2

# j = -1
# for i in np.logspace(strt, endd, count):
#   j += 1
#   print(int(i))
#   for ii in range(rep):
#     print(ii)
#     q = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
#     k = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
#     v = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))

#     start_time = time.time()
#     e = fastmax_custom(q,k,v)
#     cuda.synchronize
#     end_time = time.time()
#     fast_time_custom[j] += (end_time - start_time)/rep


# # j = -1
# # for i in np.logspace(strt, endd, count):
# #   j += 1
# #   print(int(i))
# #   for ii in range(rep):
# #     q = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
# #     k = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
# #     v = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))

# #     start_time = time.time()
# #     e = softmax(q,k,v)
# #     cuda.synchronize
# #     end_time = time.time()
# #     soft_time[j] += (end_time - start_time)/rep


# # j = -1
# # for i in np.logspace(strt, endd, count):
# #   j += 1
# #   print(int(i))
# #   for ii in range(rep):
# #     q = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
# #     k = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))
# #     v = torch.normal(0,1,[b,h,int(i),d],device=torch.device(device))

# #     start_time = time.time()
# #     e = fastmax(q,k,v)
# #     cuda.synchronize
# #     end_time = time.time()
# #     fast_time[j] += (end_time - start_time)/rep


# print("softmax = \n")
# print(soft_time)
# print("fastmax = \n")
# print(fast_time)
# print("fastmax custom = \n")
# print(fast_time_custom)

class test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qq, kk, vv, mask = 0, p = 1, normalize = 0, denum_Term = 1):
      if p != 1 and p != 2:
        print("p must be either 2 or 1")
        return 0
        
      b = qq.shape[0]
      h = qq.shape[1]
      n = qq.shape[2]
      d = qq.shape[3]
      if normalize == 1:
        qn = torch.linalg.norm(qq, axis = 3)
        kn = torch.linalg.norm(kk, axis = 3)
        # qq /= torch.linalg.norm(qn, axis = 2, ord = float('inf')).reshape((qq.shape[0],qq.shape[1],1,1))
        kk /= denum_Term*torch.linalg.norm(kn, axis = 2, ord = float('inf')).reshape((qq.shape[0],qq.shape[1],1,1))
      denum_term = 1
      denum_term2 = 0.5

      q = np.asarray(qq)
      k = np.asarray(kk)
      v = np.asarray(vv)

      o = np.zeros([b,h,n,d])
      div = np.zeros([b,h,n])

      const = 0
      lin = np.zeros([d])
      quad = np.zeros([d,d])
      const_div = 0
      lin_div = np.zeros([d])
      quad_div = np.zeros([d,d])

      if mask == 0:
        # calc denum terms
        for i in range(b):
            for j in range(h):
              const = 0
              lin = np.zeros([d])
              quad = np.zeros([d,d])
              for l in range(n):
                const += 1
                for m in range(d):
                  lin[m] += denum_term*k[i,j,l,m]
                  if p == 2:
                    for r in range(d):
                      quad[m,r] += denum_term2*k[i,j,l,r]*k[i,j,l,m]
              for l in range(n):
                div[i,j,l] = const
                for m in range(d):
                  div[i,j,l] += q[i,j,l,m]*lin[m]
                  if p == 2:
                    for r in range(d):
                      div[i,j,l] += q[i,j,l,m]*q[i,j,l,r]*quad[m,r]

        # calc num terms
        for outer in range(d):
          for i in range(b):
            for j in range(h):
              const = 0
              lin = np.zeros([d])
              quad = np.zeros([d,d])
              for l in range(n):
                const += v[i,j,l,outer]
                for m in range(d):
                  lin[m] += denum_term*k[i,j,l,m]*v[i,j,l,outer]
                  if p == 2:
                    for r in range(d):
                      quad[m,r] += denum_term2*k[i,j,l,r]*k[i,j,l,m]*v[i,j,l,outer]
              for l in range(n):
                o[i,j,l,outer] = const
                for m in range(d):
                  o[i,j,l,outer] += q[i,j,l,m]*lin[m]
                  if p == 2:
                    for r in range(d):
                      o[i,j,l,outer] += q[i,j,l,m]*q[i,j,l,r]*quad[m,r]
                o[i,j,l,outer] /= div[i,j,l] # element-wise div

      else:
        # calc denum terms
        for i in range(b):
          for j in range(h):
            const = 0
            lin = np.zeros([d])
            quad = np.zeros([d,d])
            for l in range(n):
              const += 1
              for m in range(d):
                lin[m] += denum_term*k[i,j,l,m]
                if p == 2:
                  for r in range(d):
                    quad[m,r] += denum_term2*k[i,j,l,r]*k[i,j,l,m]

              div[i,j,l] = const
              for m in range(d):
                div[i,j,l] += q[i,j,l,m]*lin[m]
                if p == 2:
                  for r in range(d):
                    div[i,j,l] += q[i,j,l,m]*q[i,j,l,r]*quad[m,r]

        # calc num terms
        for outer in range(d):
          for i in range(b):
            for j in range(h):
              const = 0
              lin = np.zeros([d])
              quad = np.zeros([d,d])
              for l in range(n):
                const += v[i,j,l,outer]
                for m in range(d):
                  lin[m] += denum_term*k[i,j,l,m]*v[i,j,l,outer]
                  if p == 2:
                    for r in range(d):
                      quad[m,r] += denum_term2*k[i,j,l,r]*k[i,j,l,m]*v[i,j,l,outer]

                o[i,j,l,outer] = const
                for m in range(d):
                  o[i,j,l,outer] += q[i,j,l,m]*lin[m]
                  if p == 2:
                    for r in range(d):
                      o[i,j,l,outer] += q[i,j,l,m]*q[i,j,l,r]*quad[m,r]
                o[i,j,l,outer] /= div[i,j,l] # element-wise div

      oo = torch.tensor(o)
      # save required data for cal. backward pass
      ctx.save_for_backward(qq, kk, vv, oo, torch.tensor(div), torch.tensor(mask), torch.tensor(p))
      return oo

    @staticmethod
    def backward(ctx, grad_ooutput):
      grad_output = np.asarray(grad_ooutput)
      qq, kk, vv, oo, div, mask, p = ctx.saved_tensors
      q = np.asarray(qq)
      k = np.asarray(kk)
      v = np.asarray(vv)
      o = np.asarray(oo)
      g = np.asarray(div)
      b = qq.shape[0]
      h = qq.shape[1]
      n = qq.shape[2]
      d = qq.shape[3]
      gradq = np.zeros([b,h,n,d])
      gradq_coeffs0v = np.zeros([d])
      gradq_coeffs0o = np.zeros([d])
      gradq_coeffs1v = np.zeros([d,d])
      gradq_coeffs1o = np.zeros([d,d])
      gradk = np.zeros([b,h,n,d])
      gradk_coeffs0v = np.zeros([d])
      gradk_coeffs0o = np.zeros([d])
      gradk_coeffs1v = np.zeros([d,d])
      gradk_coeffs1o = np.zeros([d,d])
      gradv = np.zeros([b,h,n,d])
      gradv_coeffs0 = 0
      gradv_coeffs1 = np.zeros([d])
      gradv_coeffs2 = np.zeros([d*d])
      k2 = np.zeros([d*d])
      if mask == 0:
        # grad terms for q
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradq_coeffs0v = np.zeros([d])
              gradq_coeffs0o = np.zeros([d])
              if p == 2: gradq_coeffs1v = np.zeros([d,d])
              if p == 2: gradq_coeffs1o = np.zeros([d,d])
              for l in range(n):
                gradq_coeffs0v += v[i,j,l,outer]*k[i,j,l]
                gradq_coeffs0o += k[i,j,l]
                if p == 2: 
                  gradq_coeffs1v += v[i,j,l,outer]*np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)
                  gradq_coeffs1o += np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)

              for l in range(n):
                gradq[i,j,l] += (gradq_coeffs0v - (gradq_coeffs0o)*o[i,j,l,outer])*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2: gradq[i,j,l] += (gradq_coeffs1v@q[i,j,l] - (gradq_coeffs1o@q[i,j,l])*o[i,j,l,outer])*grad_output[i,j,l,outer]/g[i,j,l]
              
        # grad terms for k
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradk_coeffs0v = np.zeros([d])
              gradk_coeffs0o = np.zeros([d])
              if p == 2: gradk_coeffs1v = np.zeros([d,d])
              if p == 2: gradk_coeffs1o = np.zeros([d,d])
              for l in range(n):
                gradk_coeffs0v += q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                gradk_coeffs0o += o[i,j,l,outer]*q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2:
                  gradk_coeffs1v += np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)*grad_output[i,j,l,outer]/g[i,j,l]
                  gradk_coeffs1o += o[i,j,l,outer]*np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)*grad_output[i,j,l,outer]/g[i,j,l]

              for l in range(n):
                gradk[i,j,l] += ((gradk_coeffs0v)*v[i,j,l,outer] - (gradk_coeffs0o))
                if p ==2: gradk[i,j,l] += ((gradk_coeffs1v@k[i,j,l])*v[i,j,l,outer] - ( gradk_coeffs1o@k[i,j,l]))

        # grad terms for v
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradv_coeffs0 = 0
              gradv_coeffs1 = np.zeros([d])
              if p == 2: gradv_coeffs2 = np.zeros([d*d])
              for l in range(n):
                gradv_coeffs0 += grad_output[i,j,l,outer]/g[i,j,l]
                gradv_coeffs1 += q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2: gradv_coeffs2 += 0.5*((np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)).reshape(d*d))*grad_output[i,j,l,outer]/g[i,j,l]

              for l in range(n):
                k2 = (np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)).reshape(d*d)
                gradv[i,j,l,outer] = gradv_coeffs0 + np.dot(gradv_coeffs1,k[i,j,l])
                if p == 2: gradv[i,j,l,outer] += np.dot(gradv_coeffs2,k2)


      else:
        # grad terms for q
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradq_coeffs0v = np.zeros([d])
              gradq_coeffs0o = np.zeros([d])
              if p == 2: gradq_coeffs1v = np.zeros([d,d])
              if p == 2: gradq_coeffs1o = np.zeros([d,d])
              for l in range(n):
                gradq_coeffs0v += v[i,j,l,outer]*k[i,j,l]
                gradq_coeffs0o += k[i,j,l]
                if p == 2:
                  gradq_coeffs1v += v[i,j,l,outer]*np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)
                  gradq_coeffs1o += np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)

                gradq[i,j,l] += (gradq_coeffs0v- (gradq_coeffs0o)*o[i,j,l,outer])*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2: gradq[i,j,l] += (gradq_coeffs1v@q[i,j,l] - (gradq_coeffs1o@q[i,j,l])*o[i,j,l,outer])*grad_output[i,j,l,outer]/g[i,j,l]
              
        # grad terms for k
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradk_coeffs0v = np.zeros([d])
              gradk_coeffs0o = np.zeros([d])
              if p == 2: gradk_coeffs1v = np.zeros([d,d])
              if p == 2: gradk_coeffs1o = np.zeros([d,d])
              for l in range(n-1, -1, -1):
                gradk_coeffs0v += q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                gradk_coeffs0o += o[i,j,l,outer]*q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2: 
                  gradk_coeffs1v += np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)*grad_output[i,j,l,outer]/g[i,j,l]
                  gradk_coeffs1o += o[i,j,l,outer]*np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)*grad_output[i,j,l,outer]/g[i,j,l]

                gradk[i,j,l] += ((gradk_coeffs0v)*v[i,j,l,outer] - (gradk_coeffs0o ))
                if p == 2: gradk[i,j,l] += ((gradk_coeffs1v@k[i,j,l])*v[i,j,l,outer] - (gradk_coeffs1o@k[i,j,l]))

        # grad terms for v
        for i in range(b):
          for j in range(h):
            for outer in range(d):
              gradv_coeffs0 = 0
              gradv_coeffs1 = np.zeros([d])
              if p == 2: gradv_coeffs2 = np.zeros([d*d])
              for l in range(n-1, -1, -1):
                gradv_coeffs0 += grad_output[i,j,l,outer]/g[i,j,l]
                gradv_coeffs1 += q[i,j,l]*grad_output[i,j,l,outer]/g[i,j,l]
                if p == 2: gradv_coeffs2 += 0.5*((np.expand_dims(q[i,j,l],1)@np.expand_dims(q[i,j,l],0)).reshape(d*d))*grad_output[i,j,l,outer]/g[i,j,l]

                k2 = (np.expand_dims(k[i,j,l],1)@np.expand_dims(k[i,j,l],0)).reshape(d*d)
                gradv[i,j,l,outer] = gradv_coeffs0 + np.dot(gradv_coeffs1,k[i,j,l])
                if p == 2: gradv[i,j,l,outer] += np.dot(gradv_coeffs2,k2)

      return torch.tensor(gradq), torch.tensor(gradk), torch.tensor(gradv), None,None,None,None


fast_grad = test.apply