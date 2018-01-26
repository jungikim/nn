#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SVDLinear.c"
#else

#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

static real THNN_(get1d)(const THTensor *t, long x0) {
  return THStorage_(get)(t->storage, t->storageOffset + x0*t->stride[0]);
}

static void THNN_(set2d)(THTensor *t, long x0, long x1, real value) {
  THStorage_(set)(t->storage, t->storageOffset + x0*t->stride[0] + x1*t->stride[1], value);
}

//Cn:   N (xB)
//z:    V (xB)
//B:    V x D
//h:    D (xB)
//bias: V
void THNN_(SVDLinear_updateFullView)(
          THNNState *state,
          THLongTensor *indices,
          THTensor *z,
          THTensor *B,
          THTensor *h,
          THTensor *bias)
{
  long N = THLongTensor_size(indices, 0);
  long V = THTensor_(size)(z, 0);
  long D = THTensor_(size)(h, 0);

  if (THLongTensor_nDimension(indices) == 1)
  { // non-batched
    #pragma omp parallel for
    for (long nIdx = 0; nIdx < N; nIdx++) {
      long vIdx = THLongTensor_get1d(indices, nIdx) - 1;
      THNN_(set1d)(z, vIdx,
               THBlas_(dot)(D,
                       ROW_PTR2(B, vIdx), B->stride[1], // B[vIdx]
                       THTensor_(data)(h), h->stride[0] // h
                   )
               + (bias ? THNN_(get1d)(bias, vIdx) : 0)
      );
    }
  }
  else // if (THLongTensor_nDimension(indices) == 2)
  { // batched
    long batchSize = THLongTensor_size(indices, 1);
    #pragma omp parallel for
    for (long nbIdx = 0; nbIdx < N * batchSize; nbIdx++) {
      long nIdx = nbIdx / batchSize;
      long bIdx = nbIdx % batchSize;
      long vIdx = THLongTensor_get2d(indices, nIdx, bIdx) - 1;
      THNN_(set2d)(z, vIdx, bIdx,
               THBlas_(dot)(D,
                       ROW_PTR2(B, vIdx), B->stride[1], // B[vIdx]
                       COL_PTR2(h, bIdx), h->stride[0]  // h[bIdx]
                   )
               + (bias ? THNN_(get1d)(bias, vIdx) : 0)
      );
    }
  }
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
