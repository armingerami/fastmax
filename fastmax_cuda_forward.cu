#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// kernel = a0 + a1x + a2x^2
__device__ float a0 = 1.0;
__device__ float a1 = 1.166666;
__device__ float a2 = 0.145833;
// -lim^2 <= q.k <= lim^2
__device__ float lim = 2;

// // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.0;
// __device__ float a2 = 0.5;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 1;

namespace {
__global__
void calc_cons_de(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> cons, int bh, int l, int d){

  const int i = threadIdx.x;
  if(i < bh){
    cons[d][i] += a0;
  }
}

__global__
void calc_lin_de(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lin, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    lin[m][d][i] += a1*k[l][m][i];
  }
}

__global__
void calc_quad_de(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> quad, int bh, int l, int d){

    const int i = threadIdx.x;
    const int m = blockIdx.x;
    const int r = blockIdx.y;
  if(m < d && r < d && i < bh){
    quad[r][m][d][i] += k[l][m][i]*k[l][r][i];
  }
}

__global__
void calc_denum_cons(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> cons, int bh, int l, int d){

  const int i = threadIdx.x;
  if(i < bh){
    o[l][d][i] += cons[d][i];
  }
}

__global__
void calc_denum_lin(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lin, int bh, int l, int d){

  const int i = threadIdx.x;
  if(i < bh){
    for(int m = 0; m < d; ++m){
    o[l][d][i] += q[l][m][i]*lin[m][d][i];
    }
  }
}

__global__
void calc_denum_quad1(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> quad, int bh, int l, int d){

  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    for(int r = 0; r < d; ++r){
      temp[m][d][i] += q[l][r][i]*q[l][m][i]*quad[r][m][d][i];
    }
  }
}

__global__
void calc_denum_quad2(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, int bh, int l, int d){

  const int i = threadIdx.x;
  if(i < bh){
    for(int m = 0; m < d; ++m){
      o[l][d][i] += a2*temp[m][d][i];
    }
  }
}

__global__
void calc_cons(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> cons, int bh, int l, int d){

    const int i = threadIdx.x; 
    const int outer = blockIdx.x;
  if(outer < d && i < bh){
    cons[outer][i] += a0*v[l][outer][i];
  }
}

__global__
void calc_lin(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lin, int bh, int l, int d){

    const int i = threadIdx.x;
    const int outer = blockIdx.x; 
    const int m = blockIdx.y;
  if(m < d && outer < d && i < bh){
    lin[m][outer][i] += a1*k[l][m][i]*v[l][outer][i];
  }
}

__global__
void calc_quad(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> quad, int bh, int l, int d){

    const int i = threadIdx.x;
    const int outer = blockIdx.x;
    const int m = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      quad[r][m][outer][i] += k[l][r][i]*k[l][m][i]*v[l][outer][i];
    }
  }
}

__global__
void calc_num_cons(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> cons, int bh, int l, int d){

    const int i = threadIdx.x; 
    const int outer = blockIdx.x;
  if(outer < d && i < bh){
    o[l][outer][i] += cons[outer][i];
  }
}

__global__
void calc_num_lin(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lin, int bh, int l, int d){

  const int i = threadIdx.x; 
  const int outer = blockIdx.x;
  if(outer < d && i < bh){
    for(int m = 0; m < d; ++m){
      o[l][outer][i] += q[l][m][i]*lin[m][outer][i];
    }
  }
}

__global__
void calc_num_quad1(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> quad, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x;
  const int m = blockIdx.y;
  if(m < d && i < bh && outer < d){
    for(int r = 0; r < d; ++r){
      temp[m][outer][i] += q[l][r][i]*q[l][m][i]*quad[r][m][outer][i];
    }
  }
}

__global__
void calc_num_quad2(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> temp, int bh, int l, int d){

  const int i = threadIdx.x;
  const int outer = blockIdx.x;
  if(i < bh && outer < d){
    for(int m = 0; m < d; ++m){
      o[l][outer][i] += a2*temp[m][outer][i];
    }
  }
}

__global__
void apply_division(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int l, int d){
  const int i = threadIdx.x;
  const int outer = blockIdx.x;
  if(outer < d && i < bh){
    o[l][outer][i] /= o[l][d][i];
  }
}

__global__
void apply_rpe_and_temp(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rpe_matrix, int bh, int l, int d, int n, float temperature){
  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    k[l][m][i] /= temperature;
    k[l][m][i] += rpe_matrix[(l+n)%(2*n-1)][m];
  }
}

__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int m){
  const int i = threadIdx.x;
  const int l = blockIdx.x;
  if(l < n && i < bh){
    norms[l][i] += a[l][m][i]*a[l][m][i];
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int l){
  const int i = threadIdx.x;
  if(i < bh){
    maxes[i] = max(maxes[i],norms[l][i]);
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int l, int d){
  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    a[l][m][i] *= lim/maxes[i];
  }
}

__global__
void apply_dropout(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> drop_noise, float dropout, int bh, int l, int d){
  const int i = threadIdx.x;
  const int m = blockIdx.x;
  if(m < d && i < bh){
    o[l][m][i] *= (1+dropout*drop_noise[l][m][i]);
  }
}

} // namespace

// torch::Tensor forward_cuda(
//     torch::Tensor q,
//     torch::Tensor k,
//     torch::Tensor v,
//     torch::Tensor o,
//     torch::Tensor rpe_matrix,
//     torch::Tensor cons,
//     torch::Tensor lin,
//     torch::Tensor quad,
//     bool mask){
torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor drop_noise,
    torch::Tensor rpe_matrix,
    bool mask,
    float dropout,
    bool normalize,
    float temperature){
    // q: (nq,d,b*h)
    // k: (nk,d,b*h)
    // v: (nk,d,b*h)

    const auto nq = q.size(0);
    const auto nk = k.size(0);
    const auto bh = q.size(2);
    const auto d = q.size(1);

    const int threads = bh;
    const int blocks0 = 1;
    const int blocks1 = d;
    const dim3 blocks2(d,d);
    
    auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
    auto o = torch::zeros({nq,d+1,bh},opts);
    auto cons = torch::zeros({d+1,bh},opts);
    auto lin = torch::zeros({d,d+1,bh},opts);
    auto quad = torch::zeros({d,d,d+1,bh},opts);
    auto z = torch::zeros({d,d+1,bh},opts);
    auto temp = torch::zeros_like(z);
    auto qnorms = torch::zeros({nq,bh},opts);
    auto knorms = torch::zeros({nk,bh},opts);
    auto qmaxes = torch::zeros({bh},opts);
    auto kmaxes = torch::zeros({bh},opts);

    for(int l = 0; l < nk; l++){
      apply_rpe_and_temp<<<blocks1,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rpe_matrix.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d,nk,temperature);
    }
  cudaDeviceSynchronize();

  if(normalize){
    for(int m = 0; m < d; ++m){
      calc_norms<<<nq,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,m);
      calc_norms<<<nk,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,m);
    }
    cudaDeviceSynchronize();
    for(int l = 0; l < nq; ++l) find_max<<<blocks0,bh>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,l);
    for(int l = 0; l < nk; ++l) find_max<<<blocks0,bh>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,l);
    cudaDeviceSynchronize();
    for(int l = 0; l < nq; ++l) apply_norm<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,l,d);
    for(int l = 0; l < nk; ++l) apply_norm<<<blocks1,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,l,d);
  }


  if(mask){
    for(int l = 0; l < nk-nq; l++){
      calc_cons_de<<<blocks0,threads>>>(cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin_de<<<blocks1,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad_de<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_cons<<<blocks1,threads>>>(v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
    }
    cudaDeviceSynchronize();
    
    for(int l = 0; l < nk-nq; l++){
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_denum_cons<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_lin<<<blocks0,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad1<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad2<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_cons<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_lin<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad1<<<blocks2,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad2<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      apply_division<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
    }

    cudaDeviceSynchronize();
    
    for(int l = nk-nq; l < nk; l++){
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_cons_de<<<blocks0,threads>>>(cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin_de<<<blocks1,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad_de<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_cons<<<blocks1,threads>>>(v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      
      calc_denum_cons<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_lin<<<blocks0,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad1<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad2<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_cons<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_lin<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad1<<<blocks2,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad2<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      apply_division<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
    }
  }

  else{
    for(int l = 0; l < nk; l++){
      calc_cons_de<<<blocks0,threads>>>(cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin_de<<<blocks1,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad_de<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_cons<<<blocks1,threads>>>(v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_lin<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_quad<<<blocks2,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
    }
    cudaDeviceSynchronize();
    for(int l = 0; l < nq; l++){
      temp = torch::zeros_like(z);
      cudaDeviceSynchronize();
      calc_denum_cons<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_lin<<<blocks0,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad1<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_denum_quad2<<<blocks0,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_cons<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),cons.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_lin<<<blocks1,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),lin.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad1<<<blocks2,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),quad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      calc_num_quad2<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),temp.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
      cudaDeviceSynchronize();
      apply_division<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,l,d);
    }
  }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    if(dropout > 0.0){
      for(int l = 0; l < nq; l++){
        apply_dropout<<<blocks1,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dropout,bh,l,d);
      }
    }

    // Free memory
    // cudaFree(denum);
    // cudaFree(lin);
    // cudaFree(quad);

  return o;
}

