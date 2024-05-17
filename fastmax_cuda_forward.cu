#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.166666;
// __device__ float a2 = 0.145833;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 2;

// // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.0;
// __device__ float a2 = 0.5;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 1;

namespace {
__global__
void calc_unmasked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // UNMASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    for(int rr = 0; rr < d/sz; ++rr){
      for(int r = 0; r < sz; ++r) tr[r]= 0;
      for(int l = 0; l < nk;  ++l){
        s[outer] = k[l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
      }
      for(int l = 0; l < nq;  ++l){
        s[d+outer] = 0;
        s[outer] = q[l][i][outer];
        __syncthreads();
        loc2 = rr*sz;
        for(int r = 0; r < sz; ++r){
          s[d+outer] += tr[r]*s[outer]*s[loc2+r];
        }
        o[l][i][outer] += s[d+outer];
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        t = 0;
        s[outer] = o[l][i][outer];
        __syncthreads();
        if(outer == 0){
          for(int r = 0; r < d; ++r) t += s[r];
          o[l][i][d] += a2*t;
        }
      }
      __syncthreads();
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    for(int m = 0; m < d; ++m){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk;  ++l){
          s[d+outer] = k[l][i][m]*k[l][i][outer];
          tv = v[l][i][outer];
          __syncthreads();
          loc1 = d+rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
          }      
        }
        for(int l = 0; l < nq;  ++l){
          s[outer] = q[l][i][m]*q[l][i][outer];
          __syncthreads();
          t = 0;
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            t += tr[r]*s[loc2+r];
          }      
          o[l][i][outer] += a2*t;
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void calc_unmasked_ryan(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2){
    //auto o = torch::zeros({nq,bh,d+1},opts);
    int i = blockDim.x * blockIdx.x + threadIdx.x; // n index
    int t = blockDim.y * blockIdx.y + threadIdx.y; // bh index
    int j = blockDim.z * blockIdx.z + threadIdx.z; // d index

    //int i_stride = blockDim.x * gridDim.x;
    //int t_stride = blockDim.y * gridDim.y;
    //int j_stride = blockDim.z * gridDim.z;

    float F_ij = 0.0;
    float z_ij_1 = 0.0;
    float z_ij_2 = 0.0;
    float z_ij_3 = 0.0;

    float g_i = 0.0;
    float z_i_g1 = 0.0;
    float z_i_g2 = 0.0;
    float z_i_g3 = 0.0;

    float y_mj_2 = 0.0;
    float y_mlj_2 = 0.0;
    float y_l_g2 = 0.0;
    float y_lp_g3 = 0.0;

    //for (int i = i_init; i < nq; i+= i_stride) {
    //    for (int t = t_init; t < bh; t+= t_stride) {
    //        for (int j = j_init; j < d; j+= j_stride) {
    //F_ij = 0.0;
    //z_ij_1 = 0.0;
    //z_ij_2 = 0.0;
    //z_ij_3 = 0.0;

    //g_i = 0.0;
    //z_i_g1 = 0.0;
    //z_i_g2 = 0.0;
    //z_i_g3 = 0.0;

    if (i < nq && t < bh && j < d) {
        for (int n = 0; n < nk; n++) {
            z_ij_1 += v[n][t][j];
        }
        z_ij_1 = z_ij_1 * a0;

        y_mj_2 = 0.0;
        for (int m = 0; m < d; m++) {
            y_mj_2 = 0.0;
            for (int n = 0; n < nk; n++) {
                y_mj_2 += k[n][t][m]*v[n][t][j];
            }
            z_ij_2 += q[i][t][m]*y_mj_2;
        }
        z_ij_2 = z_ij_2 * a1;

        y_mlj_2 = 0.0;
        for (int m = 0; m < d; m++) {
            for (int l = 0; l < d; l++) {
                y_mlj_2 = 0.0;
                for (int n = 0; n < nk; n++) {
                    y_mlj_2 += k[n][t][m] * k[n][t][l] * v[n][t][j];
                }
                z_ij_3 += q[i][t][m] * q[i][t][l] * y_mlj_2;
            }
        }
        z_ij_3 = z_ij_3 * a2;

        z_i_g1 = a0*nk;

        y_l_g2 = 0.0;
        for (int l = 0; l < d; l++) {
            y_l_g2 = 0.0;
            for (int m = 0; m < nk; m++) {
                y_l_g2 += k[m][t][l];
            }
            z_i_g2 += q[i][t][l] * y_l_g2;
        }
        z_i_g2 = z_i_g2 * a1;

        y_lp_g3 = 0.0;
        for (int l = 0; l < d; l++) {
            for (int p = 0; p < d; p++) {
                y_lp_g3 = 0.0;
                for (int m = 0; m < nk; m++) {
                    y_lp_g3 += k[m][t][l] * k[m][t][p];
                }
                z_i_g3 += q[i][t][l] * q[i][t][p] * y_lp_g3;
            }
        }
        z_i_g3 = z_i_g3 * a2;

        F_ij = z_ij_1 + z_ij_2 + z_ij_3;
        g_i = z_i_g1 + z_i_g2 + z_i_g3;
        o[i][t][j] = F_ij / g_i;
    }
    //        }
    //    }
    //}
}

__global__
void compute_z1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> z1, int bh, int nq, int nk, int d, float a0, float a1, float a2){
}

__global__
void compute_z2(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> z2, int bh, int nq, int nk, int d, float a0, float a1, float a2){
}

__global__
void compute_z3(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> z3, int bh, int nq, int nk, int d, float a0, float a1, float a2){
}

__global__
void compute_g1()

__global__
void calc_masked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // MASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk-nq+l+1);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk-nq; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[d+outer] += a1*k[nk-nq+l][i][outer];
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    for(int rr = 0; rr < d/sz; ++rr){
      for(int r = 0; r < sz; ++r) tr[r]= 0;
      for(int l = 0; l < nk-nq;  ++l){
        s[outer] = k[l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        s[outer] = k[nk-nq+l][i][outer];
        __syncthreads();
        loc1 = rr*sz;
        for(int r = 0; r < sz; ++r){
          tr[r] += s[outer]*s[loc1+r];
        }
        s[d+outer] = 0;
        s[outer] = q[l][i][outer];
        __syncthreads();
        loc2 = rr*sz;
        for(int r = 0; r < sz; ++r){
          s[d+outer] += tr[r]*s[outer]*s[loc2+r];
        }
        o[l][i][outer] += s[d+outer];
      }
      __syncthreads();
      for(int l = 0; l < nq; ++l){
        t = 0;
        s[outer] = o[l][i][outer];
        __syncthreads();
        if(outer == 0){
          for(int r = 0; r < d; ++r) t += s[r];
          o[l][i][d] += a2*t;
        }
      }

    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      t += v[nk-nq+l][i][outer];
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk-nq;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        t += k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    for(int m = 0; m < d; ++m){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk-nq;  ++l){
          s[d+outer] = k[l][i][m]*k[l][i][outer];
          tv = v[l][i][outer];
          __syncthreads();
          loc1 = d+rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
          }      
        }
        for(int l = 0; l < nq;  ++l){
          s[outer] = q[l][i][m]*q[l][i][outer];
          s[d+outer] = k[nk-nq+l][i][m]*k[nk-nq+l][i][outer];
          tv = v[nk-nq+l][i][outer];
          __syncthreads();
          t = 0;
          loc1 = d+rr*sz;
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[loc1+r]*tv;
            t += tr[r]*s[loc2+r];
          }      
          o[l][i][outer] += a2*t;
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void apply_rpe_and_temp(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rpe_matrix, int bh, int nk, int d, float temperature){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nk; ++l){
      k[l][i][m] /= temperature;
      // k[l][i][m] += rpe_matrix[(l+nk)%(2*nk-1)][m];
    }
  }
}

__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int d){
  const int i = threadIdx.x;
  const int l = blockIdx.x;
  float t;
  if(l < n && i < bh){
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[l][i][m]*a[l][i][m];
    }
    norms[l][i] = t;
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n){
  const int i = threadIdx.x;
  float t = 0;
  if(i < bh){
    for(int l = 0; l < n; ++l){
      t = max(t,norms[l][i]);
    }
    maxes[i] = t;
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d, float lim){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float t;
  if(m < d && i < bh){
    t = maxes[i];
    for(int l = 0; l < n; ++l){
      a[l][i][m]*= lim/t;
    }
  }
}

__global__
void apply_dropout(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> drop_noise, float dropout, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l){
      o[l][i][m] *= (1+dropout*drop_noise[l][i][m]);
    }
  }
}

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor drop_noise,
    torch::Tensor rpe_matrix,
    bool mask,
    float dropout,
    bool normalize,
    float temperature,
    float a0,
    float a1,
    float a2,
    float lim){
    // q: (nq,b*h,d)
    // k: (nk,b*h,d)
    // v: (nk,b*h,d)

  const auto nq = q.size(0);
  const auto nk = k.size(0);
  const auto bh = q.size(1);
  const auto d = q.size(2);

  const int threads = d; // threads = 256
  const int blocks = bh;
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  auto o = torch::zeros({nq,bh,d+1},opts);
  auto o_ryan = torch::zeros({nq,bh,d},opts);
  auto qnorms = torch::zeros({nq,bh},opts);
  auto knorms = torch::zeros({nk,bh},opts);
  auto qmaxes = torch::zeros({bh},opts);
  auto kmaxes = torch::zeros({bh},opts);

  apply_rpe_and_temp<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rpe_matrix.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d,temperature);
  cudaDeviceSynchronize();
  if(normalize){
    calc_norms<<<nq,bh>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,d);
    calc_norms<<<nk,bh>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d);
    find_max<<<1,bh>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk);
    find_max<<<1,bh>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq);
    apply_norm<<<blocks,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,d,lim);
    apply_norm<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,d,lim);
  }

  if(mask){
    calc_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2);
  }
  else{
    //calc_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2);
    dim3 threads_per_block(8,8,8); // 512
    dim3 num_blocks(
        nq / threads_per_block.x + (nq % threads_per_block.x != 0),
        bh / threads_per_block.y + (nq % threads_per_block.y != 0),
        d  / threads_per_block.z + (d  % threads_per_block.z != 0)
    );
    calc_unmasked_ryan<<<num_blocks, threads_per_block>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o_ryan.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2);

    auto z1 = torch::zeros({nq, bh, d}, opts);
    auto z2 = torch::zeros({nq, bh, d}, opts);
    auto z3 = torch::zeros({nq, bh, d}, opts);

    auto z1_g = torch::zeros({nq, d}, opts);
    auto z2_g = torch::zeros({nq, d}, opts);
    auto z3_g = torch::zeros({nq, d}, opts);

    const int num_streams = 6;
    cudaStream_t streams[num_streams];

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);
    cudaStreamCreate(&streams[4]);
    cudaStreamCreate(&streams[5]);

    dim3 threads_per_block_z(8,8,8); // 512
    dim3 num_blocks_z(
        nq / threads_per_block_z.x + (nq % threads_per_block_z.x != 0),
        bh / threads_per_block_z.y + (nq % threads_per_block_z.y != 0),
        d  / threads_per_block_z.z + (d  % threads_per_block_z.z != 0)
    );

    dim3 threads_per_block_g(8,8); // 512
    dim3 num_blocks_g(
        nq / threads_per_block_g.x + (nq % threads_per_block_g.x != 0),
        d  / threads_per_block_g.y + (d  % threads_per_block_g.y != 0)
    );

    compute_z1<<<num_blocks_z, threads_per_block_z, 0, streams[0]>>>(v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), z1.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh, nq, nk, d, a0, a1, a2);
    compute_z2<<<num_blocks_z, threads_per_block_z, 0, streams[1]>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), z2.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh, nq, nk, d, a0, a1, a2);
    compute_z3<<<num_blocks_z, threads_per_block_z, 0, streams[2]>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), z3.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh, nq, nk, d, a0, a1, a2);

    compute_g1<<<num_blocks_g, threads_per_block_g, 0, streams[3]>>>
    compute_g2<<<num_blocks_g, threads_per_block_g, 0, streams[4]>>>
    compute_g3<<<num_blocks_g, threads_per_block_g, 0, streams[5]>>>

    return o_ryan;
  }

  //cudaDeviceSynchronize();
  //apply_dropout<<<blocks,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dropout,bh,nq,d);
  //cudaDeviceSynchronize();

  return o;
}
