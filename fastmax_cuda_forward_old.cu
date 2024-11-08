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
void calc_unmasked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(outer < d && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk; ++l){
      s[d+outer] += a1*k[i][l][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[outer] = q[i][l][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[i][l][d] = t + a0*nk;
      }
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[i][l][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[i][l][outer] = a0*t;
    }

    // calc lin
    for(int mm = 0; mm < (d/sz); ++mm){
      for(int m = 0; m < sz; ++m) tr[m] = 0;
      for(int l = 0; l < nk;  ++l){
        tv = v[i][l][outer];
        s[outer+d] = k[i][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          tr[m] += s[d+mm*sz+m]*tv;
        }
      }
      for(int l = 0; l < nq;  ++l){
        s[outer] = q[i][l][outer];
        __syncthreads();
        t = 0;
        for(int m = 0; m < sz; ++m){
          t += a1*tr[m]*s[mm*sz+m];
        }
        o[i][l][outer] += t;
      }
    }

    // // calc lin
     for(int l = 0; l < nk;  ++l){
       t += k[ikv][l][m]*v[ikv][l][outer];
     }
     for(int l = 0; l < nq;  ++l){
       atomicAdd(&o[i][l][outer],t*a1*q[i][l][m]);
     }
     for(int m = 0; m < szr; ++m) tr[m] = 0;
     for(int l = 0; l < nk;  ++l){
       tv = v[i][l][outer];
       s[outer+d] = k[i][l][outer];
       __syncthreads();
       for(int m = 0; m < szr; ++m){
         tr[m] += s[d+mm*szr+m]*tv;
       }
     }
     for(int l = 0; l < nq;  ++l){
       s[outer] = q[i][l][outer];
       __syncthreads();
       t = 0;
       for(int m = 0; m < szr; ++m){
         t += a1*tr[m]*s[mm*szr+m];
       }
       atomicAdd(&o[i][l][outer],t);
     }


    for(int l = 0; l < nq; ++l) o[i][l][outer] /= o[i][l][d];

    // // calc cons denum
    // for(int l = 0; l < nq; ++l){
    //   o[l][i][d] = a0*(nk);
    // }

    // // calc lin denum
    // s[d+outer] = 0;
    // __syncthreads();
    // for(int l = 0; l < nk; ++l){
    //   s[d+outer] += a1*k[l][i][outer];
    // }
    // __syncthreads();
    // for(int l = 0; l < nq; ++l){
    //   s[outer] = q[l][i][outer];
    //   __syncthreads();
    //   if(outer == 0){
    //     t = 0;
    //     for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
    //     o[l][i][d] += t;
    //   }
    // }

    // // calc quad denum
    // if(p == 2){
    //   for(int rr = 0; rr < d/sz; ++rr){
    //     for(int r = 0; r < sz; ++r) tr[r]= 0;
    //     for(int l = 0; l < nk;  ++l){
    //       s[outer] = k[l][i][outer];
    //       __syncthreads();
    //       loc1 = rr*sz;
    //       for(int r = 0; r < sz; ++r){
    //         tr[r] += s[outer]*s[loc1+r];
    //       }
    //     }
    //     for(int l = 0; l < nq;  ++l){
    //       s[d+outer] = 0;
    //       s[outer] = q[l][i][outer];
    //       __syncthreads();
    //       loc2 = rr*sz;
    //       for(int r = 0; r < sz; ++r){
    //         s[d+outer] += tr[r]*s[outer]*s[loc2+r];
    //       }
    //       o[l][i][outer] += s[d+outer];
    //     }
    //     __syncthreads();
    //     for(int l = 0; l < nq; ++l){
    //       t = 0;
    //       s[outer] = o[l][i][outer];
    //       __syncthreads();
    //       if(outer == 0){
    //         for(int r = 0; r < d; ++r) t += s[r];
    //         o[l][i][d] += a2*t;
    //       }
    //     }
    //     __syncthreads();
    //   }
    // }

    // // calc cons
    // t = 0;
    // for(int l = 0; l < nk;  ++l){
    //   t += v[l][i][outer];
    // }
    // for(int l = 0; l < nq;  ++l){
    //   o[l][i][outer] = a0*t;
    // }

    // // calc lin
    // for(int m = 0; m < d; ++m){
    //   t = 0;
    //   for(int l = 0; l < nk;  ++l){
    //     t += k[l][i][m]*v[l][i][outer];
    //   }
    //   for(int l = 0; l < nq;  ++l){
    //     o[l][i][outer] += a1*t*q[l][i][m];
    //   }
    // }

    // // calc quad
    // if(p == 2){
    //   for(int m = 0; m < d; ++m){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk;  ++l){
    //         s[d+outer] = k[l][i][m]*k[l][i][outer];
    //         tv = v[l][i][outer];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }      
    //       }
    //       for(int l = 0; l < nq;  ++l){
    //         s[outer] = q[l][i][m]*q[l][i][outer];
    //         __syncthreads();
    //         t = 0;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }      
    //         o[l][i][outer] += a2*t;
    //       }
    //     }
    //   }
    // }

    // for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void calc_masked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(outer < d && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk-nq; ++l){
      s[d+outer] += a1*k[i][l][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[d+outer] += a1*k[i][nk-nq+l][outer];
      s[outer] = q[i][l][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[i][l][d] = t + a0*(nk-nq+l+1);
      }
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq;  ++l){
      t += v[i][l][outer];
    }
    for(int l = 0; l < nq;  ++l){
      t += v[i][nk-nq+l][outer];
      o[i][l][outer] = a0*t;
    }

    // calc lin
    for(int mm = 0; mm < (d/sz); ++mm){
      for(int m = 0; m < sz; ++m) tr[m] = 0;
      for(int l = 0; l < nk-nq;  ++l){
        tv = v[i][l][outer];
        s[outer+d] = k[i][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          tr[m] += s[d+mm*sz+m]*tv;
        }
      }
      for(int l = 0; l < nq;  ++l){
        tv = v[i][nk-nq+l][outer];
        s[outer] = q[i][l][outer];
        s[outer+d] = k[i][nk-nq+l][outer];
        __syncthreads();
        t = 0;
        for(int m = 0; m < sz; ++m){
          int mmm = mm*sz + m;
          tr[m] += s[d+mmm]*tv;
          t += a1*tr[m]*s[mmm];
        }
        o[i][l][outer] += t;
      }
    }

    for(int l = 0; l < nq;  ++l) o[i][l][outer] /= o[i][l][d];

    // calc cons denum
    // for(int l = 0; l < nq; ++l){
    //   o[l][i][d] = a0*(nk-nq+l+1);
    // }

    // // calc lin denum
    // s[d+outer] = 0;
    // __syncthreads();
    // for(int l = 0; l < nk-nq; ++l){
    //   s[d+outer] += a1*k[l][i][outer];
    // }
    // __syncthreads();
    // for(int l = 0; l < nq; ++l){
    //   s[d+outer] += a1*k[nk-nq+l][i][outer];
    //   s[outer] = q[l][i][outer];
    //   __syncthreads();
    //   if(outer == 0){
    //     t = 0;
    //     for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
    //     o[l][i][d] += t;
    //   }
    // }

    // // calc quad denum
    // if(p == 2){
    //   for(int rr = 0; rr < d/sz; ++rr){
    //     for(int r = 0; r < sz; ++r) tr[r]= 0;
    //     for(int l = 0; l < nk-nq;  ++l){
    //       s[outer] = k[l][i][outer];
    //       __syncthreads();
    //       loc1 = rr*sz;
    //       for(int r = 0; r < sz; ++r){
    //         tr[r] += s[outer]*s[loc1+r];
    //       }
    //     }
    //     __syncthreads();
    //     for(int l = 0; l < nq; ++l){
    //       s[outer] = k[nk-nq+l][i][outer];
    //       __syncthreads();
    //       loc1 = rr*sz;
    //       for(int r = 0; r < sz; ++r){
    //         tr[r] += s[outer]*s[loc1+r];
    //       }
    //       s[d+outer] = 0;
    //       s[outer] = q[l][i][outer];
    //       __syncthreads();
    //       loc2 = rr*sz;
    //       for(int r = 0; r < sz; ++r){
    //         s[d+outer] += tr[r]*s[outer]*s[loc2+r];
    //       }
    //       o[l][i][outer] += s[d+outer];
    //     }
    //     __syncthreads();
    //     for(int l = 0; l < nq; ++l){
    //       t = 0;
    //       s[outer] = o[l][i][outer];
    //       __syncthreads();
    //       if(outer == 0){
    //         for(int r = 0; r < d; ++r) t += s[r];
    //         o[l][i][d] += a2*t;
    //       }
    //     }
    //   }
    // }

    // // calc cons
    // t = 0;
    // for(int l = 0; l < nk-nq;  ++l){
    //   t += v[l][i][outer];
    // }
    // for(int l = 0; l < nq;  ++l){
    //   t += v[nk-nq+l][i][outer];
    //   o[l][i][outer] = a0*t;
    // }

    // // calc lin
    // for(int m = 0; m < d; ++m){
    //   t = 0;
    //   for(int l = 0; l < nk-nq;  ++l){
    //     t += k[l][i][m]*v[l][i][outer];
    //   }
    //   for(int l = 0; l < nq;  ++l){
    //     t += k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
    //     o[l][i][outer] += a1*t*q[l][i][m];
    //   }
    // }

    // // calc quad
    // if(p == 2){
    //   for(int m = 0; m < d; ++m){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk-nq;  ++l){
    //         s[d+outer] = k[l][i][m]*k[l][i][outer];
    //         tv = v[l][i][outer];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }      
    //       }
    //       for(int l = 0; l < nq;  ++l){
    //         s[outer] = q[l][i][m]*q[l][i][outer];
    //         s[d+outer] = k[nk-nq+l][i][m]*k[nk-nq+l][i][outer];
    //         tv = v[nk-nq+l][i][outer];
    //         __syncthreads();
    //         t = 0;
    //         loc1 = d+rr*sz;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //           t += tr[r]*s[loc2+r];
    //         }      
    //         o[l][i][outer] += a2*t;
    //       }
    //     }
    //   }
    // }

    // for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void apply_temp(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, int bh, int nk, int d, float temperature){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nk; ++l){
      k[l][i][m] /= temperature;
    }
  }
}

__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int d, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  const int l = blockIdx.y;
  float t;
  int i;
  if(l < n && ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[i][l][m]*a[i][l][m];
    }
    norms[i][l] = t;
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  float t = 0;
  int i;
  if(ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    for(int l = 0; l < n; ++l){
      t = max(t,norms[i][l]);
    }
    maxes[i] = t;
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d, float lim, int n_seg){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  const int np = int(n/n_seg);
  float mx;
  if(m < d && i < bh){
    mx = maxes[i];
    if(mx < 0.1) mx = 0.1;
    for(int l = j*np; l < min(n,(j+1)*np); ++l){
      a[i][l][m] *= lim/mx;
    }
  }
}

__global__
void apply_dropout(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> drop_noise, float dropout, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l){
      o[i][l][m] *= (1+dropout*drop_noise[i][l][m]);
    }
  }
}

__global__
void apply_permute(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a_p, int b, int h, int n, int d, int dir){
  const int m = threadIdx.x;
  const int j = blockIdx.x;
  const int i = blockIdx.y;
  if(m < d && i < b && j < h){
    for(int l = 0; l < n; ++l){
      if(dir == 0) a_p[l][i*h+j][m] = a[i][l][j][m];
      else a[i][l][j][m] = a_p[l][i*h+j][m];
    }
  }
}

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    // torch::Tensor q_old,
    // torch::Tensor k_old,
    // torch::Tensor v_old,
    // torch::Tensor drop_noise_old,
    torch::Tensor drop_noise,
    bool mask,
    float dropout,
    bool normalize,
    float temperature,
    float a0,
    float a1,
    float a2,
    float lim,
    int p){
    // q: (nq,bh,d)
    // k: (nk,bh,d)
    // v: (nk,bh,d)

  // const auto nq = q_old.size(1);
  // const auto nk = k_old.size(1);
  // const auto b = q_old.size(0);
  // const auto h = q_old.size(2);
  // const auto d = q_old.size(3);
  // const auto bh = b*h;

  // const auto nq = q.size(0);
  // const auto nk = k.size(0);
  // const auto bh = q.size(1);
  // const auto d = q.size(2);

  const auto nq = q.size(1);
  const auto nk = k.size(1);
  const auto bh = q.size(0);
  const auto bhkv = k.size(0);
  const auto d = q.size(2);
  const int bhratio = bh/bhkv;

  const int threads = d; // threads = 256
  const int blocks = bh;

  const int n_seg = 128; // breaks context length into segments of n_seg, which are parallelized; i.e., paralleizes the code n_seg times
  // int szr = int(sqrt(d)); //number of reductions happeing in each thread; should be ~sqrt(d)
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  // auto q = torch::zeros({nq,bh,d},opts);
  // auto k = torch::zeros({nk,bh,d},opts);
  // auto v = torch::zeros({nk,bh,d},opts);
  // auto drop_noise = torch::zeros({nq,bh,d},opts);
  // auto o = torch::zeros({nq,bh,d+1},opts);
  auto o = torch::zeros({bh,nq,d+1},opts);
  // auto out = torch::zeros({b,nq,h,d+1},opts);
  auto qnorms = torch::zeros({bh,nq},opts);
  auto knorms = torch::zeros({bh,nk},opts);
  auto qmaxes = torch::zeros({bh},opts);
  auto kmaxes = torch::zeros({bh},opts);


  // apply_permute<<<dim3(h,b),threads>>>(q_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(k_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(v_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(drop_noise_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);

  if(temperature != 1) apply_temp<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bhkv,nk,d,temperature);
  cudaDeviceSynchronize();
  if(normalize){
    const long th_lim = 1024;
    int th = min(th_lim, bh);
    calc_norms<<<dim3((bh-1)/th + 1, nq),th>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,d,th);
    find_max<<<(bh-1)/th + 1,th>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,th);
    for(int np = 0; np < int(nq/n_seg); ++np){
      apply_norm<<<dim3(blocks,n_seg),threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,d,lim,n_seg);
    }
    th = min(th_lim, bhkv);
    calc_norms<<<dim3((bhkv-1)/th + 1, nk),th>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bhkv,nk,d,th);
    find_max<<<(bhkv-1)/th + 1,th>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bhkv,nq,th);
    for(int np = 0; np < int(nk/n_seg); ++np){
      apply_norm<<<dim3(bhkv,n_seg),threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bhkv,nk,d,lim,n_seg);
    }
  }

  if(mask){
    calc_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2,p,bhratio);
  }
  else{
    calc_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2,p,bhratio);
  }
  

  cudaDeviceSynchronize();
  if(dropout > 0){
    apply_dropout<<<blocks,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dropout,bh,nq,d);
    cudaDeviceSynchronize();
  }

  // apply_permute<<<dim3(h,b),threads+1>>>(out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d+1,1);

  // delete q;
  // delete k;
  // delete v;
  // delete drop_noise;
  // delete o;

  return o;
  // return out;
}
