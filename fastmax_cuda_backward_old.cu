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
void calc_gradq_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22, int p,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(m < d && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
        // calc q 0 
    for(int oo = 0; oo < d/sz; ++oo){
      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nk; l++){
        tv = k[ikv][l][m];
        s[m] = v[ikv][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          tr[outer] += a1*tv*s[oo*sz+outer];
        }
      }
      for(int l = 0; l < nq; l++){
        t = 0;
        s[d+m] = grad_output[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          t += tr[outer]*s[d+oo*sz+outer];
        }
        gradq[i][l][m] += t;
      }

      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nk; l++){
        tv = k[ikv][l][m];
        for(int outer = 0; outer < sz; ++outer){
          tr[outer] += a1*tv;
        }
      }
      for(int l = 0; l < nq; l++){
        t = 0;
        s[m] = o[i][l][m];
        s[d+m] = grad_output[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz+outer;
          t += tr[outer]*s[ooo]*s[d+ooo];
        }
        gradq[i][l][m] -= t;
      }
    }

    // // calc q 0 
    // for(int outer = 0; outer < d; ++outer){
    //   tv = 0;
    //   t = 0;
    //   for(int l = 0; l < nk; l++){
    //     tv += a1*k[l][i][m]*v[l][i][outer];
    //     t += a1*k[l][i][m];
    //   }
    //   for(int l = 0; l < nq; l++){
    //     gradq[l][i][m] += (tv - t*o[l][i][outer]) * grad_output[l][i][outer];
    //   }
    // }

    // // calc q 1
    // if(p == 2){
    //   for(int outer = 0; outer < d; ++outer){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk; l++){
    //         s[d+m] = k[l][i][m];
    //         tv = v[l][i][outer]*k[l][i][m];
    //         __syncthreads();          
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //       }
    //       for(int l = 0; l < nq; l++){
    //         s[m] = q[l][i][m];
    //         __syncthreads();
    //         t = 0;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradq[l][i][m] += a22*t*grad_output[l][i][outer];
    //       }

    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk; l++){
    //         s[d+m] = k[l][i][m];
    //         tv = k[l][i][m];
    //         __syncthreads();          
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //       }
    //       for(int l = 0; l < nq; l++){
    //         s[m] = q[l][i][m];
    //         __syncthreads();
    //         t = 0;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradq[l][i][m] -= a22*t*grad_output[l][i][outer]*o[l][i][outer];
    //       }
    //     }
    //   }
    // }
  }
}

__global__
void calc_gradq_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22, int p,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(m < d && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc q 0 
    for(int oo = 0; oo < d/sz; ++oo){
      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nk-nq; l++){
        tv = k[ikv][l][m];
        s[m] = v[ikv][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          tr[outer] += a1*tv*s[oo*sz+outer];
        }
      }
      for(int l = 0; l < nq; l++){
        t = 0;
        tv = k[ikv][l][m];
        s[m] = v[ikv][l][m];
        s[d+m] = grad_output[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz+outer;
          tr[outer] += a1*tv*s[ooo];
          t += tr[outer]*s[d+ooo];
        }
        gradq[i][l][m] += t;
      }

      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nk-nq; l++){
        tv = k[ikv][l][m];
        for(int outer = 0; outer < sz; ++outer){
          tr[outer] += a1*tv;
        }
      }
      for(int l = 0; l < nq; l++){
        t = 0;
        tv = k[ikv][l][m];
        s[m] = o[i][l][m];
        s[d+m] = grad_output[i][l][m];
        __syncthreads();        
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz+outer;
          tr[outer] += a1*tv;
          t += tr[outer]*s[ooo]*s[d+ooo];
        }
        gradq[i][l][m] -= t;
      }
    }

    // // calc q 0 
    // for(int outer = 0; outer < d; ++outer){
    //   tv = 0;
    //   t = 0;
    //   for(int l = 0; l < nk-nq; l++){
    //     tv += a1*k[l][i][m]*v[l][i][outer];
    //     t += a1*k[l][i][m];
    //   }
    //   for(int l = 0; l < nq; l++){
    //     tv += a1*k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
    //     t += a1*k[nk-nq+l][i][m];
    //     gradq[l][i][m] += (tv - t*o[l][i][outer]) * grad_output[l][i][outer];
    //   }
    // }

    // // calc q 1
    // if(p == 2){
    //   for(int outer = 0; outer < d; ++outer){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk-nq; l++){
    //         s[d+m] = k[l][i][m];
    //         tv = v[l][i][outer]*k[l][i][m];
    //         __syncthreads();          
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //       }
    //       for(int l = 0; l < nq; l++){
    //         s[d+m] = k[nk-nq+l][i][m];
    //         tv = v[nk-nq+l][i][outer]*k[nk-nq+l][i][m];
    //         s[m] = q[l][i][m];
    //         __syncthreads();          
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //         t = 0;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradq[l][i][m] += a22*t*grad_output[l][i][outer];
    //       }

    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nk-nq; l++){
    //         s[d+m] = k[nk-nq+l][i][m];
    //         tv = k[l][i][m];
    //         __syncthreads();          
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //       }
    //       for(int l = 0; l < nq; l++){
    //         s[d+m] = k[nk-nq+l][i][m];
    //         tv = k[nk-nq+l][i][m];
    //         s[m] = q[l][i][m];
    //         __syncthreads();         
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //         t = 0;
    //         loc2 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradq[l][i][m] -= a22*t*grad_output[l][i][outer]*o[l][i][outer];
    //       }
    //     }
    //   }
    // }
  }
}


__global__
void calc_gradk_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22, int p,int bhratio){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(m < d && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    for(int oo = 0; oo < d/sz; ++oo){
      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nq; l++){
        tv = q[i][l][m];
        s[m] = grad_output[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          tr[outer] += a1*tv*s[oo*sz+outer];
        }
      }
      for(int l = 0; l < nk; l++){
        t = 0;
        s[d+m] = v[ikv][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          t += tr[outer]*s[d+oo*sz+outer];
        }
        gradk[ikv][l][m] += t;
      }

      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = 0; l < nq; l++){
        t = 0;
        tv = q[i][l][m];
        s[m] = grad_output[i][l][m];
        s[d+m] = o[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz + outer;
          tr[outer] += a1*s[d+ooo]*tv*s[ooo];
        }
      }

      for(int l = 0; l < nk; l++){
        t = 0;
        for(int outer = 0; outer < sz; ++outer){
          t += tr[outer];
        }
        gradk[ikv][l][m] -= t;
      }
    }

    // // calc k 0 
    // for(int outer = 0; outer < d; ++outer){
    //   tv = 0;
    //   t = 0;
    //   for(int l = 0; l < nq; l++){
    //     tv += a1*q[l][i][m]*grad_output[l][i][outer];
    //     t += a1*o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
    //   }
    //   for(int l = 0; l < nk; l++){
    //     gradk[l][i][m] += tv*v[l][i][outer] - t;
    //   }
    // }

    // // calc k 1
    // if(p == 2){
    //   for(int outer = 0; outer < d; ++outer){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nq; l++){
    //         s[d+m] = q[l][i][m];
    //         tv = q[l][i][m]*grad_output[l][i][outer];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += tv*s[loc1+r];
    //         }
    //       }
    //       for(int l = 0; l < nk; l++){
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] += a22*t*v[l][i][outer];
    //       }

    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nq; l++){
    //         s[d+m] = q[l][i][m];
    //         tv = o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += tv*s[loc1+r];
    //         }
    //       }
    //       for(int l = 0; l < nk; l++){
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] -= a22*t;
    //       }
    //     }
    //   }
    // }
  }
}

__global__
void calc_gradk_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a1, float a22, int p,int bhratio){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(m < d && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc k 0 
    for(int oo = 0; oo < d/sz; ++oo){
      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = nq-1; l >= 0; --l){
        t = 0;
        tv = q[i][l][m];
        s[m] = grad_output[i][l][m];
        s[d+m] = v[ikv][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz + outer;
          tr[outer] += a1*tv*s[ooo];
          t += tr[outer]*s[d+ooo];
        }
        gradk[ikv][l][m] += t;
      }
      for(int l = nk-nq-1; l >= 0; --l){
        t = 0;
        s[d+m] = v[ikv][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          t += tr[outer]*s[d+oo*sz+outer];
        }
        gradk[ikv][l][m] += t;
      }

      for(int outer = 0; outer < sz; ++outer) tr[outer] = 0;
      for(int l = nq-1; l >= 0; --l){
        t = 0;
        tv = q[i][l][m];
        s[m] = grad_output[i][l][m];
        s[d+m] = o[i][l][m];
        __syncthreads();
        for(int outer = 0; outer < sz; ++outer){
          int ooo = oo*sz + outer;
          tr[outer] += a1*s[d+ooo]*tv*s[ooo];
          t += tr[outer];
        }
        gradk[ikv][l][m] -= t;
      }
      for(int l = nk-nq-1; l >= 0; --l){
        t = 0;
        for(int outer = 0; outer < sz; ++outer){
          t += tr[outer];
        }
        gradk[ikv][l][m] -= t;
      }
    }

    // // calc k 0 
    // for(int outer = 0; outer < d; ++outer){
    //   tv = 0;
    //   t = 0;
    //   for(int l = nq-1; l >= 0; --l){
    //     tv += a1*q[l][i][m]*grad_output[l][i][outer];
    //     t += a1*o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
    //     gradk[l][i][m] += tv*v[l][i][outer] - t;
    //   }
    //   for(int l = nk-nq-1; l >= 0; --l){
    //     gradk[l][i][m] += tv*v[l][i][outer] - t;
    //   }
    // }

    // // calc k 1
    // if(p == 2){
    //   for(int outer = 0; outer < d; ++outer){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = nq-1; l >= 0; --l){
    //         s[d+m] = q[l][i][m];
    //         tv = q[l][i][m]*grad_output[l][i][outer];
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += tv*s[loc1+r];
    //         }
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] += a22*t*v[l][i][outer];
    //       }
    //       for(int l = nk-nq-1; l >= 0; --l){
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] += a22*t*v[l][i][outer];
    //       }

    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = nq-1; l >= 0; --l){
    //         s[d+m] = q[l][i][m];
    //         tv = o[l][i][outer]*q[l][i][m]*grad_output[l][i][outer];
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc1 = d+rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += tv*s[loc1+r];
    //         }
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] -= a22*t;
    //       }
    //       for(int l = nk-nq-1; l >= 0; --l){
    //         s[m] = k[l][i][m];
    //         __syncthreads();
    //         loc2 = rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradk[l][i][m] -= a22*t;
    //       }
    //     }
    //   }
    // }
  }
}

__global__
void calc_gradv_unmasked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(i < bh && outer < d){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    // calc v 0    
    t = 0;
    for(int l = 0; l < nq; ++l){
      t += a0*grad_output[i][l][outer];
    }
    for(int l = 0; l < nk; ++l){
      gradv[ikv][l][outer] += t;
    }

    // calc v 1
    
    for(int mm = 0; mm < d/sz; ++mm){
      for(int m = 0; m < sz; ++m) tr[m] = 0;
      for(int l = 0; l < nq; ++l){
        s[outer] = q[i][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          tr[m] += a1*s[mm*sz+m] * grad_output[i][l][outer];
        }
      }
      for(int l = 0; l < nk; ++l){
        t = 0;
        s[d+outer] = k[ikv][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          t += tr[m]*s[d+mm*sz+m];
        }
        gradv[ikv][l][outer] += t;
      }
    }

    // // calc v 0
    // t = 0;
    // for(int l = 0; l < nq; ++l){
    //   t += a0*grad_output[i][l][outer];
    // }
    // for(int l = 0; l < nk; ++l){
    //   gradv[ikv][l][outer] += t;
    // }

    // // calc v 1
    // for(int m = 0; m < d; ++m){
    //   t = 0;
    //   for(int l = 0; l < nq; ++l){
    //     t += a1*q[i][l][m] * grad_output[i][l][outer];
    //   }
    //   for(int l = 0; l < nk; ++l){
    //     gradv[ikv][l][outer] += t*k[ikv][l][m];
    //   }
    // }

    // // calc v 2
    // if(p == 2){
    //   for(int m = 0; m < d; ++m){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = 0; l < nq; l++){
    //         s[outer] = q[l][i][m]*q[l][i][outer];
    //         tv = grad_output[l][i][outer];
    //         __syncthreads();
    //         loc1 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //       }
    //       for(int l = 0; l < nk; l++){
    //         s[d+outer] = k[l][i][m]*k[l][i][outer];
    //         __syncthreads();
    //         loc2 = d+rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradv[l][i][outer] += a2*t;
    //       }
    //     }
    //   }
    // }
  }
}

__global__
void calc_gradv_masked(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  if(i < bh && outer < d){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc v 0    
    t = 0;
    for(int l = nq-1; l >= 0; --l){
      t += a0*grad_output[i][l][outer];
      gradv[ikv][l][outer] += a0*t;
    }
    for(int l = nk-nq-1; l >= 0; --l){
      gradv[ikv][l][outer] += a0*t;
    }

    // calc v 1
    
    for(int mm = 0; mm < d/sz; ++mm){
      for(int m = 0; m < sz; ++m) tr[m] = 0;
      for(int l = nq-1; l >= 0; --l){
        t = 0;
        s[outer] = q[i][l][outer];
        s[d+outer] = k[ikv][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          int mmm = mm*sz + m;
          tr[m] += a1*s[mmm] * grad_output[i][l][outer];
          t += tr[m]*s[d+mmm];
        }
        gradv[ikv][l][outer] += t;
      }
      for(int l = nk-nq-1; l >= 0; --l){
        t = 0;
        s[d+outer] = k[ikv][l][outer];
        __syncthreads();
        for(int m = 0; m < sz; ++m){
          t += tr[m]*s[d+mm*sz+m];
        }
        gradv[ikv][l][outer] += t;
      }
    }
    

    // // calc v 0    
    // t = 0;
    // for(int l = nq-1; l >= 0; --l){
    //   t += a0*grad_output[l][i][outer];
    //   gradv[l][i][outer] += a0*t;
    // }
    // for(int l = nk-nq-1; l >= 0; --l){
    //   gradv[l][i][outer] += a0*t;
    // }

    // // calc v 1
    // for(int m = 0; m < d; ++m){
    //   t = 0;
    //   for(int l = nq-1; l >= 0; --l){
    //     t += a1*q[l][i][m] * grad_output[l][i][outer];
    //     gradv[l][i][outer] += t*k[l][i][m];
    //   }
    //   for(int l = nk-nq-1; l >= 0; --l){
    //     gradv[l][i][outer] += t*k[l][i][m];
    //   }
    // }

    // // calc v 2
    // if(p == 2){
    //   for(int m = 0; m < d; ++m){
    //     for(int rr = 0; rr < d/sz; ++rr){
    //       for(int r = 0; r < sz; ++r) tr[r]= 0;
    //       for(int l = nq-1; l >= 0; --l){
    //         s[outer] = q[l][i][m]*q[l][i][outer];
    //         tv = grad_output[l][i][outer];
    //         s[d+outer] = k[l][i][m]*k[l][i][outer];
    //         __syncthreads();
    //         loc1 = rr*sz;
    //         for(int r = 0; r < sz; ++r){
    //           tr[r] += s[loc1+r]*tv;
    //         }
    //         loc2 = d+rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradv[l][i][outer] += a2*t;
    //       }
    //       for(int l = nk-nq-1; l >= 0; --l){
    //         s[d+outer] = k[l][i][m]*k[l][i][outer];
    //         __syncthreads();
    //         loc2 = d+rr*sz;
    //         t = 0;
    //         for(int r = 0; r < sz; ++r){
    //           t += tr[r]*s[loc2+r];
    //         }
    //         gradv[l][i][outer] += a2*t;
    //       }
    //     }
    //   }
    // }
  }
}


__global__
void div_grad_output(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l) grad_output[i][l][m] /= o[i][l][d];
  }
}

__global__
void apply_permute(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a_p, int b, int h, int n, int d, int dir){
  const int m = threadIdx.x;
  const int j = blockIdx.x;
  const int i = blockIdx.y;
  if(m < d && i < b && j < h){
    for(int l = 0; l < n; ++l){
      // if(dir == 0) a_p[l][i*h+j][m] = a[i][l][j][m];
      // else a[i][l][j][m] = a_p[l][i*h+j][m];
      a[i][l][j][m] = a_p[l][i*h+j][m];
    }
  }
}

} // namespace

std::vector<torch::Tensor> backward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor grad_output,  
    // torch::Tensor q_old,
    // torch::Tensor k_old,
    // torch::Tensor v_old,
    // torch::Tensor o_old,
    // torch::Tensor grad_output_old,  
    bool mask,
    float a0,
    float a1,
    float a2,
    int p){

    // q: (nq,bh,d)
    // k: (nk,bh,d)
    // v: (nk,bh,d)
    // grad_output: (nq,bh,d)

    // gradq_coeffs0 = lin
    // gradq_coeffs1 = quad
    // gradq_coeffs0v = lin[:,:d,:]
    // gradq_coeffs0o = lin[:,d,:]
    // gradq_coeffs1v = quad[:,:d,:,:]
    // gradq_coeffs1o = quad[:,d,:,:]
    // o[:,d,:] = denum

  // const auto nq = q_old.size(1);
  // const auto nk = k_old.size(1);
  // const auto b = q_old.size(0);
  // const auto h = q_old.size(2);
  // const auto d = q_old.size(3);
  // const auto bh = b*h;

  const auto nq = q.size(1);
  const auto nk = k.size(1);
  const auto bh = q.size(0);
  const auto bhkv = k.size(0);
  const auto d = q.size(2);
  const int bhratio = bh/bhkv;

  const int threads = d;
  const int blocks = bh;
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  // auto q = torch::zeros({nq,bh,d},opts);
  // auto k = torch::zeros({nk,bh,d},opts);
  // auto v = torch::zeros({nk,bh,d},opts);
  // auto o = torch::zeros({nq,bh,d+1},opts);
  // auto grad_output = torch::zeros({nq,bh,d},opts);
  auto gradq = torch::zeros({bh,nq,d},opts);
  auto gradk = torch::zeros({bh,nk,d},opts);
  auto gradv = torch::zeros({bh,nk,d},opts);
  // auto gradq_out = torch::zeros({b,nq,h,d},opts);
  // auto gradk_out = torch::zeros({b,nk,h,d},opts);
  // auto gradv_out = torch::zeros({b,nk,h,d},opts);

  // apply_permute<<<dim3(h,b),threads>>>(q_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(k_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(v_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads+1>>>(o_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d+1,0);
  // apply_permute<<<dim3(h,b),threads>>>(grad_output_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);

  div_grad_output<<<blocks,threads>>>(grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,d);
  cudaDeviceSynchronize();

  if(mask){
    calc_gradq_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2,p,bhratio);
    calc_gradk_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2,p,bhratio);
    calc_gradv_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a0,a1,a2,p,bhratio);
  }
  else{
    calc_gradq_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2,p,bhratio);
    calc_gradk_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a1,2*a2,p,bhratio);
    calc_gradv_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,a0,a1,a2,p,bhratio);
  }

  cudaDeviceSynchronize();

  // apply_permute<<<dim3(h,b),threads>>>(gradq_out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,1);
  // apply_permute<<<dim3(h,b),threads>>>(gradk_out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,1);
  // apply_permute<<<dim3(h,b),threads>>>(gradv_out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,1);

  // delete q;
  // delete k;
  // delete v;
  // delete o;
  // delete grad_output;
  // delete gradq;
  // delete gradk;
  // delete gradv;

  return {gradq,gradk,gradv};
}

