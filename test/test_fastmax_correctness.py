import os
import sys
import unittest
import math
import torch
import fastmax_cuda
import einops
import numpy as np


class FASTMultiHeadAttention_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        drop_noise,
        rpe_matrix=None,
        mask=False,
        dropout=0.0,
        normalize=False,
        temperature=1.0,
        a0=1.0,
        a1=1.0,
        a2=0.5,
        lim=1.0,
    ):
        b = 0
        if len(q.shape) == 4:
            b = q.shape[0]
            q = q.reshape(
                (q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            k = k.reshape(
                (k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            v = v.reshape(
                (v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
            )  # (b,h,n,d) -> (b*h,n,d)
            drop_noise = drop_noise.reshape(
                (
                    drop_noise.shape[0] * drop_noise.shape[1],
                    drop_noise.shape[2],
                    drop_noise.shape[3],
                )
            )  # (b,h,n,d) -> (b*h,n,d)
        elif len(q.shape) != 3:
            print(
                "q, k, and v should be either 3 or 4 dimensional tensors. If 3D: (b*h,n,d), if 4D: (b,h,n,d)."
            )

        if rpe_matrix is None:
            print(
                "Relative Positional Encoding must be given. Send a 2*n-1 by d matrix of all zeros if you don't want to use RPE."
            )

        # q = q.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        # k = k.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        # v = v.permute(1,2,0).contiguous() # (b*h,n,d) -> (n,d,b*h)
        q = q.permute(1, 0, 2).contiguous()  # (b*h,n,d) -> (n,b*h,d)
        k = k.permute(1, 0, 2).contiguous()  # (b*h,n,d) -> (n,b*h,d)
        v = v.permute(1, 0, 2).contiguous()  # (b*h,n,d) -> (n,b*h,d)
        drop_noise = drop_noise.permute(1, 0, 2).contiguous()  # (b*h,n,d) -> (n,b*h,d)
        # print(torch.cuda.memory_allocated())
        o = fastmax_cuda.forwardpass(
            q,
            k,
            v,
            drop_noise,
            rpe_matrix,
            mask,
            dropout,
            normalize,
            temperature,
            a0,
            a1,
            a2,
            lim,
        )
        # print(torch.cuda.memory_allocated())
        # print('a')
        ctx.save_for_backward(q, k, v, o)
        ctx.mask = mask
        ctx.b = b
        ctx.t = temperature
        ctx.a0 = a0
        ctx.a1 = a1
        ctx.a2 = a2
        o = o[:, :, : q.shape[2]]
        o = o.permute(1, 0, 2).contiguous()  # (n,b*h,d) -> (b*h,n,d)
        if b != 0:
            o = o.reshape(
                (b, int(o.shape[0] / b), o.shape[1], o.shape[2])
            )  # (b*h,n,d) -> (b,h,n,d)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o = ctx.saved_tensors
        mask = ctx.mask
        b = ctx.b
        t = ctx.t
        a0 = ctx.a0
        a1 = ctx.a1
        a2 = ctx.a2

        if b != 0:
            grad_output = grad_output.reshape(
                (
                    grad_output.shape[0] * grad_output.shape[1],
                    grad_output.shape[2],
                    grad_output.shape[3],
                )
            ).contiguous()
        grad_output = grad_output.permute(
            1, 0, 2
        ).contiguous()  # (b*h,n,d) -> (n,b*h,d)
        gradq, gradk, gradv = fastmax_cuda.backwardpass(
            q, k, v, o, grad_output, mask, a0, a1, a2
        )

        gradq = gradq.permute(1, 0, 2).contiguous()  # (n,b*h,d) -> (b*h,n,d)
        gradk = gradk.permute(1, 0, 2).contiguous()  # (n,b*h,d) -> (b*h,n,d)
        gradv = gradv.permute(1, 0, 2).contiguous()  # (n,b*h,d) -> (b*h,n,d)

        if b != 0:
            gradq = gradq.reshape(
                (b, int(gradq.shape[0] / b), gradq.shape[1], gradq.shape[2])
            ).contiguous()
            gradk = gradk.reshape(
                (b, int(gradk.shape[0] / b), gradk.shape[1], gradk.shape[2])
            ).contiguous()
            gradv = gradv.reshape(
                (b, int(gradv.shape[0] / b), gradv.shape[1], gradv.shape[2])
            ).contiguous()

        return (
            gradq,
            gradk / t,
            gradv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FASTMultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(FASTMultiHeadAttention, self).__init__()

    def forward(
        self,
        q,
        k,
        v,
        drop_noise,
        rpe_matrix=None,
        mask=False,
        dropout=0.0,
        normalize=False,
        temperature=1.0,
        a0=1.0,
        a1=1.0,
        a2=0.5,
        lim=1.0,
    ):
        return FASTMultiHeadAttention_Function.apply(
            q,
            k,
            v,
            drop_noise,
            rpe_matrix,
            mask,
            dropout,
            normalize,
            temperature,
            a0,
            a1,
            a2,
            lim,
        )


def rpe_matrix_creator(n, d, device, dtype, structured=True, is_zero=False):
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
    if dtype != torch.float32:
        print("The data type must be float32 in order for Fastmax to work")
    if structured:
        pe_positive = torch.zeros(n, d, device=device, dtype=dtype)
        pe_negative = torch.zeros(n, d, device=device, dtype=dtype)
        position = torch.arange(0, n, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=dtype) * -(math.log(10000.0) / d)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0])
        pe_negative = pe_negative[1:]
        rpe = torch.cat([pe_positive, pe_negative], dim=0)
    else:
        if is_zero:
            rpe = torch.zeros(size=(2 * n - 1, d), device=device, dtype=dtype)
        else:
            rpe = torch.normal(0, 1, size=(2 * n - 1, d), device=device, dtype=dtype)
    return rpe


class TestFastmax(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-5
        B = 5
        H = 8
        N = 512
        D = 32
        #B = 1 
        #H = 1
        #N = 3
        #D = 3
        self.q = torch.randn(
            B,
            H,
            N,
            D,
            dtype=torch.float,
            requires_grad=True,
            device=torch.device("cuda"),
        )
        self.k = torch.randn(
            B,
            H,
            N,
            D,
            dtype=torch.float,
            requires_grad=True,
            device=torch.device("cuda"),
        )
        self.v = torch.randn(
            B,
            H,
            N,
            D,
            dtype=torch.float,
            requires_grad=True,
            device=torch.device("cuda"),
        )

        #norm_term = math.sqrt(D)
        #self.s = (
        #    einops.einsum(self.q, self.k, "b h i d, b h j d -> b h i j") / norm_term
        #)
        self.s = (
            einops.einsum(self.q, self.k, "b h i d, b h j d -> b h i j")
        )
        f_p2 = lambda x: 1 + x + x**2 / 2

        # Compute unmasked attention and output
        fp2_um = f_p2(self.s)
        sums_p2_um = einops.reduce(fp2_um, "b h n1 n2 -> b h n1", "sum")
        sums_p2_um = einops.repeat(sums_p2_um, "b h n1 -> b h n1 n", n=N)
        self.a_p2_um = fp2_um / sums_p2_um

        self.o_p2_unmasked = einops.einsum(
            self.a_p2_um, self.v, "b h i n, b h n j -> b h i j"
        )

        # Compute masked attention and output
        upper_mask = torch.triu(
            torch.ones((N, N), dtype=bool, device=self.s.device), diagonal=1
        )
        upper_mask = einops.repeat(upper_mask, "i j -> b h i j", b=B, h=H)

        fp2_m = torch.zeros_like(self.s, device=self.s.device)
        fp2_m = f_p2(torch.masked_fill(self.s, upper_mask, float("inf")))
        inf_mask = torch.isinf(fp2_m)
        fp2_m[inf_mask] = 0.0
        sums_p2_m = einops.reduce(fp2_m, "b h n1 n2 -> b h n1", "sum")
        sums_p2_m = einops.repeat(sums_p2_m, "b h n1 -> b h n1 n", n=N)
        self.a_p2_m = fp2_m / sums_p2_m

        self.o_p2_masked = einops.einsum(
            self.a_p2_m, self.v, "b h i n, b h n j -> b h i j"
        )
        self.fm = FASTMultiHeadAttention()
        self.drop_noise = torch.zeros(
            size=(self.q.shape), dtype=self.q.dtype, device=self.q.device
        )
        self.rpe_matrix = rpe_matrix_creator(
            self.k.shape[-2],
            self.q.shape[-1],
            self.q.device,
            self.q.dtype,
            structured=False,
            is_zero=True,
        )
        self.temperature = 1.0
        self.a0 = 1.0
        self.a1 = 1.0
        self.a2 = 0.5
        self.lim = 1.0

    def test_fastmax_forward_p2_unmasked(self):
        o_fm_p2_unmasked = self.fm(
            q=self.q,
            k=self.k,
            v=self.v,
            drop_noise=self.drop_noise,
            rpe_matrix=self.rpe_matrix,
            mask=False,
            dropout=False,
            normalize=False,
            temperature=self.temperature,
            a0=self.a0,
            a1=self.a1,
            a2=self.a2,
            lim=self.lim,
        )
        err_unmasked = torch.max(torch.abs(self.o_p2_unmasked - o_fm_p2_unmasked))
        self.assertTrue(
            err_unmasked < self.eps, msg=f"Failed with error = {err_unmasked}"
        )

    def test_fastmax_forward_p2_masked(self):
        o_fm_p2_masked = self.fm(
            q=self.q,
            k=self.k,
            v=self.v,
            drop_noise=self.drop_noise,
            rpe_matrix=self.rpe_matrix,
            mask=True,
            dropout=False,
            normalize=False,
            temperature=self.temperature,
            a0=self.a0,
            a1=self.a1,
            a2=self.a2,
            lim=self.lim,
        )
        err_masked = torch.max(torch.abs(self.o_p2_masked - o_fm_p2_masked))
        self.assertTrue(err_masked < self.eps, msg=f"Failed with error = {err_masked}")

    @unittest.skip("Not implemented yet")
    def test_fastmax_backward_p2_unmasked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=False,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    @unittest.skip("Not implemented yet")
    def test_fastmax_backward_p2_masked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=True,
            normalize_term=1,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    @unittest.skip("Not implemented yet")
    def test_fastmax_backward_p2_norm_term_unmasked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=False,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    @unittest.skip("Not implemented yet")
    def test_fastmax_backward_p2_norm_term_masked(self):
        f = lambda q, k, v: fm(
            q,
            k,
            v,
            mask=True,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=False,
        )
        self.assertTrue(
            torch.autograd.gradcheck(f, (self.q, self.k, self.v), eps=1e-6, atol=1e-4)
        )

    @unittest.skip("Not implemented yet")
    def test_fastmax_iscausal(self):
        _, a_fm = fm(
            self.q,
            self.k,
            self.v,
            mask=True,
            normalize_term=42,
            tensors_normalized=False,
            p=2,
            dropout_rate=0.0,
            create_attn=True,
        )
        B, H, N, _ = a_fm.shape
        for b in range(B):
            for h in range(H):
                for i in range(N):
                    for j in range(i + 1, N):
                        self.assertEqual(a_fm[b, h, i, j], 0.0)


if __name__ == "__main__":
    unittest.main()
