
#include "paddle/phi/kernels/multi_head_attention_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void MultiHeadAttentionKernel(const Context& dev_ctx,
                 const DenseTensor& q,
                 const DenseTensor& k,
                 const DenseTensor& v,
                 const DenseTensor& attn_mask,
                 DenseTensor* out) {
    std::cout << "tanglei in CPU MHA" << std::endl;

}

}

PD_REGISTER_KERNEL(multi_head_attention,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiHeadAttentionKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

