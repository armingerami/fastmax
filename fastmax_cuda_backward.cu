#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.166666;
// __device__ float a2 = 0.145833;
// __device__ float a22 = 0.145833*2;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 2;

// // // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.0;
// __device__ float a2 = 0.5;
// __device__ float a22 = 0.5*2;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 1;

namespace {
__global__
void calc_gradq_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(m < d && i < bh){
    // UNMASKED PART ////////////////////////////
    // calc q 0 
    for(int outer = 0; outer < d; ++outer){
      tv = 0;
      t = 0;
      for(int l = 0; l < nk; l++){
        tv += a1*k[l][i][m]*v[l][i][outer];
        t += a1*k[l][i][m];
      }
      for(int l = 0; l < nq; l++){
        gradq[l][i][m] += (tv - t*o[l][i][outer]) * grad_output[l][i][outer];
      }
    }

  }
}

__global__
void calc_gradq_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(m < d && i < bh){
    // MASKED PART ////////////////////////////
    // calc q 0 
    for(int outer = 0; outer < d; ++outer){
      tv = 0;
      t = 0;
      for(int l = 0; l < nk-nq; l++){
        tv += a1*k[l][i][m]*v[l][i][outer];
        t += a1*k[l][i][m];
      }
      for(int l = 0; l < nq; l++){
        tv += a1*k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
        t += a1*k[nk-nq+l][i][m];
        gradq[l][i][m] += (tv - t*o[l][i][outer]) * grad_output[l][i][outer];
      }
    }

  }
}


__global__
void calc_gradk_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(m < d && i < bh){

    // UNMASKED PART ////////////////////////////
    // calc k 0 
    for(int outer = 0; outer < d; ++outer){
      tv = 0;
      t = 0;
      for(int l = 0; l < nq; l++){
        tv += a1*q[l][i][m]*grad_output[l][i][outer];
        t += a1*o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
      }
      for(int l = 0; l < nk; l++){
        gradk[l][i][m] += tv*v[l][i][outer] - t;
      }
    }

  }
}

__global__
void calc_gradk_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(m < d && i < bh){

    // MASKED PART ////////////////////////////
    // calc k 0 
    for(int outer = 0; outer < d; ++outer){
      tv = 0;
      t = 0;
      for(int l = nq-1; l >= 0; --l){
        tv += a1*q[l][i][m]*grad_output[l][i][outer];
        t += a1*o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
        gradk[l][i][m] += tv*v[l][i][outer] - t;
      }
      for(int l = nk-nq-1; l >= 0; --l){
        gradk[l][i][m] += tv*v[l][i][outer] - t;
      }
    }

  }
}

__global__
void calc_gradv_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a0, float a1, float a2){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(i < bh && outer < d){
    // UNMASKED PART ////////////////////////////
    // calc v 0
    t = 0;
    for(int l = 0; l < nq; ++l){
      t += a0*grad_output[l][i][outer];
    }
    for(int l = 0; l < nk; ++l){
      gradv[l][i][outer] += t;
    }

    // calc v 1
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nq; ++l){
        t += a1*q[l][i][m] * grad_output[l][i][outer];
      }
      for(int l = 0; l < nk; ++l){
        gradv[l][i][outer] += t*k[l][i][m];
      }
    }

  }
}

__global__
void calc_gradv_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a0, float a1, float a2){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(i < bh && outer < d){
    // MASKED PART ////////////////////////////
    // calc v 0    
    t = 0;
    for(int l = nq-1; l >= 0; --l){
      t += a0*grad_output[l][i][outer];
      gradv[l][i][outer] += a0*t;
    }
    for(int l = nk-nq-1; l >= 0; --l){
      gradv[l][i][outer] += a0*t;
    }

    // calc v 1
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = nq-1; l >= 0; --l){
        t += a1*q[l][i][m] * grad_output[l][i][outer];
        gradv[l][i][outer] += t*k[l][i][m];
      }
      for(int l = nk-nq-1; l >= 0; --l){
        gradv[l][i][outer] += t*k[l][i][m];
      }
    }

  }
}


__global__
void div_grad_output(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l) grad_output[l][i][m] /= o[l][i][d];
  }
}


} // namespace

std::vector<torch::Tensor> backward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor grad_output,  
    bool mask,
    float a0,
    float a1,
    float a2){

    // q: (nq,b*h,d)
    // k: (nk,b*h,d)
    // v: (nk,b*h,d)
    // grad_output: (nq,b*h,d)

    // gradq_coeffs0 = lin
    // gradq_coeffs1 = quad
    // gradq_coeffs0v = lin[:,:d,:]
    // gradq_coeffs0o = lin[:,d,:]
    // gradq_coeffs1v = quad[:,:d,:,:]
    // gradq_coeffs1o = quad[:,d,:,:]
    // o[:,d,:] = denum

  const auto nq = q.size(0);
  const auto nk = k.size(0);
  const auto bh = q.size(1);
  const auto d = q.size(2);

  const int threads = d;
  const int blocks = bh;
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  auto gradq = torch::zeros({nq,bh,d},opts);
  auto gradk = torch::zeros({nk,bh,d},opts);
  auto gradv = torch::zeros({nk,bh,d},opts);

  div_grad_output<<<blocks,threads>>>(grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,d);
  cudaDeviceSynchronize();

  if(mask){
    calc_gradq_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2);
    calc_gradk_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2);
    calc_gradv_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a0,a1,a2);
  }
  else{
    calc_gradq_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2);
    calc_gradk_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2);
    calc_gradv_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a0,a1,a2);
  }

  cudaDeviceSynchronize();


  return {gradq,gradk,gradv};
}

