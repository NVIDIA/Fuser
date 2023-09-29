import jinja2
import pygments
from pygments.lexers import CppLexer, DiffLexer
from pygments.formatters import HtmlFormatter

env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="."))
template = env.get_template("codediff.html")

# some lorem ipsum code and diff
some_code = """
__global__ void kernel1(Tensor<float, 4, 4> T0, Tensor<float, 4, 4> T1) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = ceilDiv(T0.logical_size[3], 32);
  nvfuser_index_t i1;
  i1 = T0.logical_size[2] * i0;
  nvfuser_index_t i2;
  i2 = ((nvfuser_index_t)blockIdx.x) % i1;
  nvfuser_index_t i3;
  i3 = i2 / i0;
  nvfuser_index_t i4;
  i4 = i2 % i0;
  nvfuser_index_t i5;
  i5 = ((nvfuser_index_t)blockIdx.x) / i1;
  nvfuser_index_t i6;
  i6 = ((nvfuser_index_t)threadIdx.x) / 8;
  nvfuser_index_t i7;
  i7 = ((nvfuser_index_t)threadIdx.x) % 8;
  nvfuser_index_t i8;
  i8 = 32 * i4;
  nvfuser_index_t i9;
  i9 = 4 * i7;
  nvfuser_index_t i10;
  i10 = (((i9 + ((T0.logical_size[3] * T0.logical_size[2]) * i6)) + (((32 * T0.logical_size[3]) * T0.logical_size[2]) * i5)) + (T0.logical_size[3] * i3)) + i8;
  nvfuser_index_t i11;
  i11 = (16 * T0.logical_size[3]) * T0.logical_size[2];
  nvfuser_index_t i12;
  i12 = 4 * ((nvfuser_index_t)threadIdx.x);
  nvfuser_index_t i13;
  i13 = ((nvfuser_index_t)threadIdx.x) / 32;
  nvfuser_index_t i14;
  i14 = ((nvfuser_index_t)threadIdx.x) % 32;
  nvfuser_index_t i15;
  i15 = (32 * i14) + i13;
  nvfuser_index_t i16;
  i16 = T0.logical_size[1] * T0.logical_size[3];
  nvfuser_index_t i17;
  i17 = 32 * i5;
  nvfuser_index_t i18;
  i18 = i14 + i17;
  nvfuser_index_t i19;
  i19 = ((((T0.logical_size[1] * i13) + (i16 * i3)) + ((32 * T0.logical_size[1]) * i4)) + ((i16 * T0.logical_size[2]) * (i18 / T0.logical_size[1]))) + (i18 % T0.logical_size[1]);
  nvfuser_index_t i20;
  i20 = 4 * T0.logical_size[1];
  nvfuser_index_t i21;
  i21 = T0.logical_size[0] * T0.logical_size[1];
  bool b22;
  b22 = i18 < i21;
  bool b23;
  b23 = ((3 + i9) + i8) < T0.logical_size[3];
  nvfuser_index_t i24;
  i24 = ((-i21) + i6) + i17;
  nvfuser_index_t i25;
  i25 = ((-T0.logical_size[3]) + i13) + i8;
  float* T2 = reinterpret_cast<float*>(array + smem_offset + 0);
  if (((((((16 + i6) + i17) < i21) && ((((i7 * 4) + 3) + i8) < T0.logical_size[3])) && b22) && (((28 + i13) + i8) < T0.logical_size[3]))) {
    #pragma unroll
    for(nvfuser_index_t i26 = 0; i26 < 2; ++i26) {
      loadGeneric<float, 4>( &T2[(i12 + (512 * i26))],  &T0[(i10 + (i11 * (i26 + nvfuser_zero)))]);
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T3[8];
    __barrier_sync(0);
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 8; ++i27) {
      T3[i27]
         = T2[(i15 + (4 * i27))];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i28 = 0; i28 < 8; ++i28) {
      T1[(i19 + (i20 * (i28 + nvfuser_zero)))]
         = T3[i28];
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  } else {
    #pragma unroll
    for(nvfuser_index_t i26 = 0; i26 < 2; ++i26) {
      nvfuser_index_t i29;
      i29 = i26 + nvfuser_zero;
      if ((b23 && (i24 < (-(16 * i29))))) {
        loadGeneric<float, 4>( &T2[(i12 + (512 * i26))],  &T0[(i10 + (i11 * i29))]);
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    float T3[8];
    __barrier_sync(0);
    #pragma unroll
    for(nvfuser_index_t i27 = 0; i27 < 8; ++i27) {
      if ((b22 && (i25 < (-(4 * (i27 + nvfuser_zero)))))) {
        T3[i27]
           = T2[(i15 + (4 * i27))];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
    #pragma unroll
    for(nvfuser_index_t i28 = 0; i28 < 8; ++i28) {
      nvfuser_index_t i30;
      i30 = i28 + nvfuser_zero;
      if ((b22 && (i25 < (-(4 * i30))))) {
        T1[(i19 + (i20 * i30))]
           = T3[i28];
      }
    }
    NVFUSER_UPDATE_MAGIC_ZERO;
  }
}
        """
some_diff = """
__global__ void kernel15(Tensor<float, 5, 5> T8, nvfuser_index_t i0, nvfuser_index_t i1, nvfuser_index_t i2, Tensor<float, 5, 5> T13, Tensor<float, 5, 5> T6) {
   alignas(16) extern __shared__ char array[];
   void* shared_mem = array;
   NVFUSER_DEFINE_MAGIC_ZERO;
   nvfuser_index_t i3;
   i3 = 4 * ((nvfuser_index_t)threadIdx.x);
   Tensor<float, 5, 5> s4;
   s4.data = T8.data;
   s4.logical_size = T8.logical_size;
   s4.alloc_stride = T8.alloc_stride;
   double d5;
-  d5 = (double)(i1);
+  d5 = (double)(i0);
   double d6;
-  d6 = (double)(i2);
+  d6 = (double)(i1);
   double d7;
   d7 = (double)(0);
   double d8;
   d8 = (double)(0);
   double d9;
   d9 = (double)(0);
   double d10;
-  d10 = (double)(i1);
+  d10 = (double)(i0);
   double d11;
-  d11 = (double)(i2);
+  d11 = (double)(i1);
   Array<nvfuser_index_t, 5, 1> a12;
   a12 = s4.logical_size;
   nvfuser_index_t i13;
   i13 = a12[2];
   nvfuser_index_t i14;
-  i14 = i3 + (((i1 * i2) * i13) * ((nvfuser_index_t)blockIdx.x));
+  i14 = i3 + (((i0 * i1) * i13) * ((nvfuser_index_t)blockIdx.x));
   nvfuser_index_t i15;
-  i15 = (i2 * i1) * i13;
+  i15 = (i1 * i0) * i13;
   nvfuser_index_t i16;
   i16 = 4 * (ceilDiv((ceilDiv(i15, 4)), 7));
   nvfuser_index_t i17;
   i17 = (3 - i15) + i3;
   bool b18;
   b18 = ((nvfuser_index_t)threadIdx.x) == 0;
   double d19;
   d19 = (double)(i13);
   double d20;
   d20 = (double)(i13);
   double d21;
   d21 = 1.00000000000000000e+00 * d20;
   double d22;
   d22 = d21 * d11;
   double d23;
   d23 = d22 * d10;
   double d24;
   d24 = d23 - d9;
   bool b25;
   b25 = d24 >= d8;
                    """

template_vars = {
    "pygments_style_defs": HtmlFormatter().get_style_defs(".highlight"),
    "git1": {
        "abbrev": "8fd1ff44",
        "full_hash": "8fd144083db93d5f954b62b25f1c159947652691",
        "pull_request": {
            "title": "Wrap CompiledKernel in unique_ptr and add a proper destructor.",
            "number": 968,
        },
        "author_name": "Jacob Hinkle",
        "author_email": "jhinkle@nvidia.com",
        "author_datetime": "Wed Sep 27 09:52:34 2023 -0400",
        "title": "Merge remote-tracking branch 'origin/main' into scalar_seg_edges",
    },
    "git2": {
        "abbrev": "877dc636",
        "full_hash": "877dc63606d35d44a0320f927fdb83fd8168eaf9",
        "pull_request": {
            "title": "Visit extent scalars in SegmentCandidateFinder::resolveScalarsInGroup",
            "number": 840,
        },
        "author_name": "Jacob Hinkle",
        "author_email": "jhinkle@nvidia.com",
        "author_datetime": "Wed Sep 27 07:26:54 2023 -0400",
        "title": "Merge remote-tracking branch 'origin/main' into scalar_seg_edges",
    },
    "test_diffs": [
        {
            "name": "NVFuserTestFoo",
            "kernels": [
                {
                    "kernel_num": 3,
                    "highlighted_code1": pygments.highlight(some_code, CppLexer(), HtmlFormatter()),
                    "highlighted_code2": pygments.highlight(some_code, CppLexer(), HtmlFormatter()),
                    "highlighted_diff": pygments.highlight(some_diff, DiffLexer(), HtmlFormatter()),
                },
            ],
        },
    ],
    "new_tests": [
        {
            "name": "bat",
            "highlighted_code": pygments.highlight(some_code, CppLexer(), HtmlFormatter()),
        },
    ],
    "removed_tests": [
        {
            "name": "baz",
            "highlighted_code": pygments.highlight(some_code, CppLexer(), HtmlFormatter()),
        },
    ]
}

print(template.render(template_vars))
