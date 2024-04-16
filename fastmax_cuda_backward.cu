#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// kernel = a0 + a1x + a2x^2
__device__ float a0 = 1.0;
__device__ float a1 = 1.166666;
__device__ float a2 = 0.145833;
__device__ float a22 = 0.145833*2;
// -lim^2 <= q.k <= lim^2
__device__ float lim = 2;

namespace {
  __global__
void calc_gradq_coeffs0v(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq_coeffs0, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //outer iterates through d
  const int outer = blockIdx.y; //i iterates through bh
  if(m < d && outer < d && i < bh){
    gradq_coeffs0[outer][m][i] += a1*k[l][m][i]*v[l][outer][i];
  }
}

  __global__
void calc_gradq_coeffs0o(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq_coeffs0, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //i iterates through bh
  if(m < d && i < bh){
    gradq_coeffs0[d][m][i] += a1*k[l][m][i];
  }
}

__global__
void calc_gradq_coeffs1v(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradq_coeffs1, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int r = blockIdx.y;
  if(m < d && i < bh && r < d){
    for(int outer = 0; outer < d; ++outer){
      gradq_coeffs1[outer][r][m][i] += a22*k[l][m][i]*k[l][r][i]*v[l][outer][i];
    }
  }
}

__global__
void calc_gradq_coeffs1o(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradq_coeffs1, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int r = blockIdx.y;
  if(m < d && r < d && i < bh){
    gradq_coeffs1[d][r][m][i] += a22*k[l][m][i]*k[l][r][i];
  }
}

__global__
void calc_gradk_coeffs0v(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk_coeffs0v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //outer iterates through d
  const int outer = blockIdx.y; //i iterates through bh
  if(m < d && outer < d && i < bh){
    gradk_coeffs0v[outer][m][i] += a1*q[l][m][i]*grad_output[l][outer][i];
  }
}

__global__
void calc_gradk_coeffs0o(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk_coeffs0o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //outer iterates through d
  const int outer = blockIdx.y; //i iterates through bh
  if(m < d && outer < d && i < bh){
    gradk_coeffs0o[outer][m][i] += a1*o[l][outer][i] * q[l][m][i] * grad_output[l][outer][i];
  }
}

__global__
void calc_gradk_coeffs1v(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradk_coeffs1v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int outer = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      gradk_coeffs1v[outer][r][m][i] += a22*q[l][m][i] * q[l][r][i] * grad_output[l][outer][i];
    }
  }
}

__global__
void calc_gradk_coeffs1o(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradk_coeffs1o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int outer = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      gradk_coeffs1o[outer][r][m][i] += a22*o[l][outer][i]*q[l][m][i] * q[l][r][i] * grad_output[l][outer][i];
    }
  }
}

__global__
void calc_gradv_coeffs0(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gradv_coeffs0, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x; //outer iterates through d
  const int outer = blockIdx.x; //i iterates through bh
  if(i < bh && outer < d){
    gradv_coeffs0[outer][i] += a0*grad_output[l][outer][i];
  }
}

__global__
void calc_gradv_coeffs1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv_coeffs1, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x; //outer iterates through d
  const int m = blockIdx.y; //i iterates through bh
  if(i < bh && outer < d && m < d){
    gradv_coeffs1[m][outer][i] += a1*q[l][m][i] * grad_output[l][outer][i];
  }
}

__global__
void calc_gradv_coeffs2(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradv_coeffs2, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x;
  const int m = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      gradv_coeffs2[r][m][outer][i] += q[l][m][i] * q[l][r][i] * grad_output[l][outer][i];
    }
  }
}

__global__
void calc_gradq1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq_coeffs0, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //i iterates through bh
  if(i < bh && m < d){
    for(int outer = 0; outer < d; ++outer){
      gradq[l][m][i] += (gradq_coeffs0[outer][m][i] - gradq_coeffs0[d][m][i] * o[l][outer][i]) * grad_output[l][outer][i];
    }
  }
}

__global__
void calc_gradq21(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradq_coeffs1, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int r = blockIdx.y;
  if(m < d && r < d && i < bh){
    for(int outer = 0; outer < d; ++outer){
      temp[r][m][i] += (gradq_coeffs1[outer][r][m][i] - gradq_coeffs1[d][r][m][i] * o[l][outer][i]) * grad_output[l][outer][i];
    }
  }
}

__global__
void calc_gradq22(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    for(int r = 0; r < d; ++r){
      gradq[l][m][i] += temp[r][m][i] * q[l][r][i];
    }
  }
}

__global__
void calc_gradk1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk_coeffs0v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk_coeffs0o, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x; //i iterates through bh
  if(i < bh && m < d){
    for(int outer = 0; outer < d; ++outer){
      gradk[l][m][i] += gradk_coeffs0v[outer][m][i] * v[l][outer][i] - gradk_coeffs0o[outer][m][i];
    }
  }
}

__global__
void calc_gradk21(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradk_coeffs1v, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradk_coeffs1o, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  const int r = blockIdx.y;
  if(m < d && r < d && i < bh){
    for(int outer = 0; outer < d; ++outer){
      temp[r][m][i] += (gradk_coeffs1v[outer][r][m][i] * v[l][outer][i] - gradk_coeffs1o[outer][r][m][i]);
    }
  }
}

__global__
void calc_gradk22(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    for(int r = 0; r < d; ++r){
      gradk[l][m][i] += a2*temp[r][m][i]*k[l][r][i];
    }
  }
}

__global__
void calc_gradv0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> gradv_coeffs0, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x; 
  if(i < bh && outer < d){
    gradv[l][outer][i] += gradv_coeffs0[outer][i];
  }
}

__global__
void calc_gradv1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv_coeffs1, int bh, int l, int d){

  const int i = threadIdx.x; 
  const int outer = blockIdx.x; 
  if(i < bh && outer < d){
    for(int m = 0; m < d; ++m){
      gradv[l][outer][i] += gradv_coeffs1[m][outer][i]*k[l][m][i];
    }
  }
}

__global__
void calc_gradv21(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> gradv_coeffs2, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x;
  const int m = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      temp[m][outer][i] += gradv_coeffs2[r][m][outer][i]*k[l][m][i]*k[l][r][i];
    }
  }
}

__global__
void calc_gradv22(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.y;
  if(i < bh && outer < d){
    for(int m = 0; m < d; ++m){
      gradv[l][outer][i] += temp[m][outer][i];
    }
  }
}


__global__
void div_grad_output(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int l, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    grad_output[l][m][i] /= o[l][d][i];
  }
}


} // namespace

std::vector<torch::Tensor> backward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor grad_output,  
    bool mask){

    // q: (nq,d,b*h)
    // k: (nk,d,b*h)
    // v: (nk,d,b*h)
    // grad_output: (nq,d,b*h)

    // gradq_coeffs0 = lin
    // gradq_coeffs1 = quad
    // gradq_coeffs0v = lin[:,:d,:]
    // gradq_coeffs0o = lin[:,d,:]
    // gradq_coeffs1v = quad[:,:d,:,:]
    // gradq_coeffs1o = quad[:,d,:,:]
    // o[:,d,:] = denum

    const auto nq = q.size(0);
    const auto nk = k.size(0);
    const auto bh = q.size(2);
    const auto d = q.size(1);

    const int threads = bh;
    const int blocks0 = 1;
    const int blocks1 = d;
    const dim3 blocks2(d,d);
    
    auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
    auto gradq = torch::zeros({nq,d,bh},opts);
    auto gradk = torch::zeros({nk,d,bh},opts);
    auto gradv = torch::zeros({nk,d,bh},opts);
    auto gradq_coeffs0 = torch::zeros({d+1,d,bh},opts);
    auto gradq_coeffs1 = torch::zeros({d+1,d,d,bh},opts);
    auto gradk_coeffs0v = torch::zeros({d,d,bh},opts);
    auto gradk_coeffs0o = torch::zeros({d,d,bh},opts);
    auto gradk_coeffs1v = torch::zeros({d,d,d,bh},opts);
    auto gradk_coeffs1o = torch::zeros({d,d,d,bh},opts);
    auto gradv_coeffs0 = torch::zeros({d,bh},opts);
    auto gradv_coeffs1 = torch::zeros({d,d,bh},opts);
    auto gradv_coeffs2 = torch::zeros({d,d,d,bh},opts);
    auto z = torch::zeros({d,d,bh},opts);
    auto temp = torch::zeros_like(z);

  for(int l = 0; l < nq; l++){
    div_grad_output<<<blocks0,threads>>>(grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
  }
  
  cudaDeviceSynchronize();
  if(mask){
    for(int l = 0; l < nq; l++){
      calc_gradq_coeffs0v<<<blocks2,threads>>>(gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs0o<<<blocks1,threads>>>(gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs1v<<<blocks2,threads>>>(gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs1o<<<blocks2,threads>>>(gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq1<<<blocks1,threads>>>(gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradq21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),  gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq22<<<blocks1,threads>>>(gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
    }

    for(int l = nk-1; l >= 0; l--){
      calc_gradk_coeffs0v<<<blocks2,threads>>>(gradk_coeffs0v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs0o<<<blocks2,threads>>>(gradk_coeffs0o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs1v<<<blocks2,threads>>>(gradk_coeffs1v.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs1o<<<blocks2,threads>>>(gradk_coeffs1o.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs0<<<blocks1,threads>>>(gradv_coeffs0.packed_accessor32<float,2,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs1<<<blocks2,threads>>>(gradv_coeffs1.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs2<<<blocks2,threads>>>(gradv_coeffs2.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk1<<<blocks1,threads>>>(gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs0v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs0o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradk21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs1v.packed_accessor32<float,4,torch::RestrictPtrTraits>(), gradk_coeffs1o.packed_accessor32<float,4,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk22<<<blocks1,threads>>>(gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv0<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs0.packed_accessor32<float,2,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv1<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs1.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradv21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs2.packed_accessor32<float,4,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv22<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
    }
  }

  else{
    for(int l = 0; l < nq; l++){
      calc_gradq_coeffs0v<<<blocks2,threads>>>(gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs0o<<<blocks1,threads>>>(gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs1v<<<blocks2,threads>>>(gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq_coeffs1o<<<blocks2,threads>>>(gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
    }

    for(int l = 0; l < nk; l++){
      calc_gradk_coeffs0v<<<blocks2,threads>>>(gradk_coeffs0v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs0o<<<blocks2,threads>>>(gradk_coeffs0o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs1v<<<blocks2,threads>>>(gradk_coeffs1v.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk_coeffs1o<<<blocks2,threads>>>(gradk_coeffs1o.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs0<<<blocks1,threads>>>(gradv_coeffs0.packed_accessor32<float,2,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs1<<<blocks2,threads>>>(gradv_coeffs1.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv_coeffs2<<<blocks2,threads>>>(gradv_coeffs2.packed_accessor32<float,4,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
    }
    cudaDeviceSynchronize();
    for(int l = 0; l < nq; l++){
      calc_gradq1<<<blocks1,threads>>>(gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq_coeffs0.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradq21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),  gradq_coeffs1.packed_accessor32<float,4,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradq22<<<blocks1,threads>>>(gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
    }
    for(int l = nk-1; l >= 0; l--){
      calc_gradk1<<<blocks1,threads>>>(gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs0v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs0o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradk21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk_coeffs1v.packed_accessor32<float,4,torch::RestrictPtrTraits>(), gradk_coeffs1o.packed_accessor32<float,4,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradk22<<<blocks1,threads>>>(gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv0<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs0.packed_accessor32<float,2,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv1<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs1.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_gradv21<<<blocks2,threads>>>(temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv_coeffs2.packed_accessor32<float,4,torch::RestrictPtrTraits>(), bh,l,d);
      cudaDeviceSynchronize();
      calc_gradv22<<<blocks1,threads>>>(gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,l,d);
    }
  }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory
    // cudaFree(denum);
    // cudaFree(lin);
    // cudaFree(quad);

  return {gradq,gradk,gradv};
}

