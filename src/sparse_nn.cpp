#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

static inline uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

static inline float from_bf16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

__attribute__((target("avx512f,avx512bf16")))
static float dot_bf16(const uint16_t *a, const uint16_t *b, uint32_t count) {
    __m512 sum = _mm512_setzero_ps();
    uint32_t index = 0;
    for (; index + 32 <= count; index += 32) {
        const __m512bh av = (__m512bh)_mm512_loadu_si512((const void *)(a + index));
        const __m512bh bv = (__m512bh)_mm512_loadu_si512((const void *)(b + index));
        sum = _mm512_dpbf16_ps(sum, av, bv);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; index < count; ++index) result += from_bf16(a[index]) * from_bf16(b[index]);
    return result;
}

extern "C" bool sparse_nn_has_avx512_bf16() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bf16");
}

extern "C" __attribute__((target("avx512f,avx512bw,avx512bf16")))
void sparse_nn_forward_bf16(
    const uint16_t *w1_input_major, const float *b1,
    const float *norm_weight, const float *norm_bias, float prelu,
    const uint16_t *w2_output_major, const float *b2,
    uint32_t hidden, const uint32_t *input_indices,
    const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    alignas(64) float h[1024];
    alignas(64) uint16_t hq[1024];
    if (hidden > 1024 || (hidden & 15u)) __builtin_trap();
    std::memcpy(h, b1, hidden * sizeof(float));
    for (uint32_t row = 0; row < input_count; ++row) {
        const uint16_t *weights = w1_input_major + size_t(input_indices[row]) * hidden;
        const __m512 value = _mm512_set1_ps(from_bf16(to_bf16(input_values[row])));
        for (uint32_t column = 0; column < hidden; column += 16) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(weights + column));
            const __m512 decoded = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
            _mm512_store_ps(h + column, _mm512_fmadd_ps(value, decoded, _mm512_load_ps(h + column)));
        }
    }
    float mean = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) mean += h[i];
    mean /= hidden;
    float variance = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        const float delta = h[i] - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden + 1.0e-5f);
    for (uint32_t i = 0; i < hidden; ++i) {
        float value = (h[i] - mean) * inv_std * norm_weight[i] + norm_bias[i];
        if (value < 0.0f) value *= prelu;
        hq[i] = to_bf16(value);
    }
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t canonical = output_indices[row];
        output[row] = dot_bf16(w2_output_major + size_t(canonical) * hidden, hq, hidden) + b2[canonical];
    }
}

extern "C" __attribute__((target("avx512f,avx512bw,avx512bf16")))
void sparse_nn_forward_bf16_dense_input(
    const uint16_t *w1_input_major, const float *b1,
    const float *norm_weight, const float *norm_bias, float prelu,
    const uint16_t *w2_output_major, const float *b2,
    uint32_t hidden, const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    alignas(64) float h[1024];
    alignas(64) uint16_t hq[1024];
    if (hidden > 1024 || (hidden & 15u)) __builtin_trap();
    std::memcpy(h, b1, hidden * sizeof(float));
    for (uint32_t row = 0; row < input_count; ++row) {
        const float raw = input_values[row];
        if (raw == 0.0f) continue;
        const uint16_t *weights = w1_input_major + size_t(row) * hidden;
        const __m512 value = _mm512_set1_ps(from_bf16(to_bf16(raw)));
        for (uint32_t column = 0; column < hidden; column += 16) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(weights + column));
            const __m512 decoded = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
            _mm512_store_ps(h + column, _mm512_fmadd_ps(value, decoded, _mm512_load_ps(h + column)));
        }
    }
    float mean = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) mean += h[i];
    mean /= hidden;
    float variance = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        const float delta = h[i] - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden + 1.0e-5f);
    for (uint32_t i = 0; i < hidden; ++i) {
        float value = (h[i] - mean) * inv_std * norm_weight[i] + norm_bias[i];
        if (value < 0.0f) value *= prelu;
        hq[i] = to_bf16(value);
    }
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t compact = output_indices[row];
        output[row] = dot_bf16(w2_output_major + size_t(compact) * hidden, hq, hidden) + b2[compact];
    }
}

extern "C" __attribute__((target("avx512f,avx512bw,avx512bf16")))
void sparse_nn_forward_bf16_compact_cfv(
    const uint16_t *w1_input_major, const float *b1,
    const float *norm_weight, const float *norm_bias, float prelu,
    const uint16_t *w2_output_major, const float *b2,
    uint32_t hidden, const float *cfv_values, uint32_t cfv_count,
    float cfv_scale, const uint32_t *extra_indices,
    const float *extra_values, uint32_t extra_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    alignas(64) float h[1024];
    alignas(64) uint16_t hq[1024];
    if (hidden > 1024 || (hidden & 15u) || !(cfv_scale > 0.0f)) __builtin_trap();
    std::memcpy(h, b1, hidden * sizeof(float));

    // Range specialization stores the active CFV rows first, in the same
    // basis-major order as the solver's chance-return buffer. Consume that
    // buffer directly rather than constructing index/value staging arrays.
    for (uint32_t row = 0; row < cfv_count; ++row) {
        const float raw = cfv_values[row];
        if (raw == 0.0f) continue;
        const uint16_t *weights = w1_input_major + size_t(row) * hidden;
        const __m512 value = _mm512_set1_ps(from_bf16(to_bf16(raw / cfv_scale)));
        for (uint32_t column = 0; column < hidden; column += 16) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(weights + column));
            const __m512 decoded = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
            _mm512_store_ps(h + column, _mm512_fmadd_ps(value, decoded, _mm512_load_ps(h + column)));
        }
    }
    for (uint32_t row = 0; row < extra_count; ++row) {
        const uint16_t *weights = w1_input_major + size_t(extra_indices[row]) * hidden;
        const __m512 value = _mm512_set1_ps(from_bf16(to_bf16(extra_values[row])));
        for (uint32_t column = 0; column < hidden; column += 16) {
            const __m256i packed = _mm256_loadu_si256((const __m256i *)(weights + column));
            const __m512 decoded = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
            _mm512_store_ps(h + column, _mm512_fmadd_ps(value, decoded, _mm512_load_ps(h + column)));
        }
    }

    float mean = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) mean += h[i];
    mean /= hidden;
    float variance = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        const float delta = h[i] - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden + 1.0e-5f);
    for (uint32_t i = 0; i < hidden; ++i) {
        float value = (h[i] - mean) * inv_std * norm_weight[i] + norm_bias[i];
        if (value < 0.0f) value *= prelu;
        hq[i] = to_bf16(value);
    }
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t compact = output_indices[row];
        output[row] = dot_bf16(w2_output_major + size_t(compact) * hidden, hq, hidden) + b2[compact];
    }
}

extern "C" __attribute__((target("avx512f")))
void sparse_nn_forward_f32_avx512(
    const float *w1_input_major, const float *b1,
    const float *norm_weight, const float *norm_bias, float prelu,
    const float *w2_output_major, const float *b2,
    uint32_t hidden, const uint32_t *input_indices,
    const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    alignas(64) float h[1024];
    if (hidden > 1024 || (hidden & 15u)) __builtin_trap();
    std::memcpy(h, b1, hidden * sizeof(float));
    for (uint32_t row = 0; row < input_count; ++row) {
        const float *weights = w1_input_major + size_t(input_indices[row]) * hidden;
        const __m512 value = _mm512_set1_ps(input_values[row]);
        for (uint32_t column = 0; column < hidden; column += 16) {
            _mm512_store_ps(h + column, _mm512_fmadd_ps(
                value, _mm512_loadu_ps(weights + column), _mm512_load_ps(h + column)));
        }
    }
    float mean = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) mean += h[i];
    mean /= hidden;
    float variance = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        const float delta = h[i] - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden + 1.0e-5f);
    for (uint32_t i = 0; i < hidden; ++i) {
        float value = (h[i] - mean) * inv_std * norm_weight[i] + norm_bias[i];
        h[i] = value < 0.0f ? value * prelu : value;
    }
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t canonical = output_indices[row];
        const float *weights = w2_output_major + size_t(canonical) * hidden;
        __m512 sum = _mm512_setzero_ps();
        for (uint32_t column = 0; column < hidden; column += 16) {
            sum = _mm512_fmadd_ps(
                _mm512_loadu_ps(weights + column), _mm512_load_ps(h + column), sum);
        }
        output[row] = _mm512_reduce_add_ps(sum) + b2[canonical];
    }
}

// FP32 kernels for older x86 CPUs. Preserve sixteen accumulation lanes and
// the AVX512 horizontal reduction order, even though AVX2 uses two registers.
__attribute__((target("avx2,fma")))
static void sparse_nn_forward_f32_avx2(
    const float *w1, const float *b1, const float *nw, const float *nb, float prelu,
    const float *w2, const float *b2, uint32_t hidden,
    const uint32_t *indices, const float *values, uint32_t inputs,
    const uint32_t *outputs, uint32_t count, float *output) {
    alignas(32) float h[1024];
    if (hidden > 1024 || (hidden & 15u)) __builtin_trap();
    std::memcpy(h,b1,hidden*sizeof(float));
    for (uint32_t r=0;r<inputs;++r) {
        const float *w=w1+size_t(indices[r])*hidden;
        __m256 v=_mm256_set1_ps(values[r]);
        for (uint32_t c=0;c<hidden;c+=8)
            _mm256_store_ps(h+c,_mm256_fmadd_ps(v,_mm256_loadu_ps(w+c),_mm256_load_ps(h+c)));
    }
    float mean=0.f;for(uint32_t c=0;c<hidden;++c)mean+=h[c];mean/=hidden;
    float variance=0.f;for(uint32_t c=0;c<hidden;++c){float d=h[c]-mean;variance+=d*d;}
    const float inv=1.f/std::sqrt(variance/hidden+1.e-5f);
    for(uint32_t c=0;c<hidden;++c){float v=(h[c]-mean)*inv*nw[c]+nb[c];h[c]=v<0.f?v*prelu:v;}
    for(uint32_t r=0;r<count;++r){
        const float *w=w2+size_t(outputs[r])*hidden;
        __m256 lo=_mm256_setzero_ps(),hi=_mm256_setzero_ps();
        for(uint32_t c=0;c<hidden;c+=16){
            lo=_mm256_fmadd_ps(_mm256_loadu_ps(w+c),_mm256_load_ps(h+c),lo);
            hi=_mm256_fmadd_ps(_mm256_loadu_ps(w+c+8),_mm256_load_ps(h+c+8),hi);
        }
        const __m256 sum=_mm256_add_ps(hi,lo);
        const __m128 four=_mm_add_ps(_mm256_extractf128_ps(sum,1),_mm256_castps256_ps128(sum));
        const __m128 pairs=_mm_add_ps(four,_mm_shuffle_ps(four,four,_MM_SHUFFLE(1,0,3,2)));
        const __m128 all=_mm_add_ps(pairs,_mm_shuffle_ps(pairs,pairs,_MM_SHUFFLE(0,1,0,1)));
        output[r]=_mm_cvtss_f32(all)+b2[outputs[r]];
    }
}

static void sparse_nn_forward_f32_scalar(
    const float *w1,const float *b1,const float *nw,const float *nb,float prelu,
    const float *w2,const float *b2,uint32_t hidden,
    const uint32_t *indices,const float *values,uint32_t inputs,
    const uint32_t *outputs,uint32_t count,float *output){
    float h[1024];if(hidden>1024||(hidden&15u))__builtin_trap();
    std::memcpy(h,b1,hidden*sizeof(float));
    for(uint32_t r=0;r<inputs;++r)for(uint32_t c=0;c<hidden;++c)
        h[c]=std::fma(values[r],w1[size_t(indices[r])*hidden+c],h[c]);
    float mean=0.f;for(uint32_t c=0;c<hidden;++c)mean+=h[c];mean/=hidden;
    float variance=0.f;for(uint32_t c=0;c<hidden;++c){float d=h[c]-mean;variance+=d*d;}
    float inv=1.f/std::sqrt(variance/hidden+1.e-5f);
    for(uint32_t c=0;c<hidden;++c){float v=(h[c]-mean)*inv*nw[c]+nb[c];h[c]=v<0.f?v*prelu:v;}
    for(uint32_t r=0;r<count;++r){
        float lanes[16]={};const float *w=w2+size_t(outputs[r])*hidden;
        for(uint32_t c=0;c<hidden;++c)lanes[c%16]=std::fma(w[c],h[c],lanes[c%16]);
        float eight[8],four[4],pairs[2];
        for(int c=0;c<8;++c)eight[c]=lanes[c+8]+lanes[c];
        for(int c=0;c<4;++c)four[c]=eight[c+4]+eight[c];
        for(int c=0;c<2;++c)pairs[c]=four[c]+four[c+2];
        output[r]=pairs[0]+pairs[1]+b2[outputs[r]];
    }
}

extern "C" void sparse_nn_forward_f32(
    const float *w1,const float *b1,const float *nw,const float *nb,float prelu,
    const float *w2,const float *b2,uint32_t hidden,
    const uint32_t *indices,const float *values,uint32_t inputs,
    const uint32_t *outputs,uint32_t count,float *output){
    // Force a backend only for parity diagnostics. Auto keeps the old AVX512
    // path on supported hosts and chooses AVX2/FMA or scalar elsewhere.
    static const int mode=[](){const char *v=std::getenv("SPARSE_NN_F32_KERNEL");
        if(v&&std::strcmp(v,"scalar")==0)return 0;
        if(v&&std::strcmp(v,"avx2")==0)return (__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))?1:0;
        if(__builtin_cpu_supports("avx512f"))return 2;
        return (__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma"))?1:0;}();
    if(mode==2)sparse_nn_forward_f32_avx512(w1,b1,nw,nb,prelu,w2,b2,hidden,indices,values,inputs,outputs,count,output);
    else if(mode==1)sparse_nn_forward_f32_avx2(w1,b1,nw,nb,prelu,w2,b2,hidden,indices,values,inputs,outputs,count,output);
    else sparse_nn_forward_f32_scalar(w1,b1,nw,nb,prelu,w2,b2,hidden,indices,values,inputs,outputs,count,output);
}

// Opt-in static-input experiment: contiguous, board/range-specialized FP32.
extern "C" __attribute__((target("avx512f")))
void sparse_nn_forward_f32_static_dense(
    const float *w1_input_major, const float *b1,
    const float *norm_weight, const float *norm_bias, float prelu,
    const float *w2_output_major, const float *b2,
    uint32_t hidden, const float *input_values, uint32_t input_count,
    uint32_t output_count, float *output) {
    alignas(64) float h[1024];
    if (hidden > 1024 || (hidden & 15u)) __builtin_trap();
    std::memcpy(h, b1, hidden * sizeof(float));
    for (uint32_t row = 0; row < input_count; ++row) {
        const float *weights = w1_input_major + size_t(row) * hidden;
        const __m512 value = _mm512_set1_ps(input_values[row]);
        for (uint32_t column = 0; column < hidden; column += 16) {
            _mm512_store_ps(h + column, _mm512_fmadd_ps(
                value, _mm512_loadu_ps(weights + column), _mm512_load_ps(h + column)));
        }
    }
    float mean = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) mean += h[i];
    mean /= hidden;
    float variance = 0.0f;
    for (uint32_t i = 0; i < hidden; ++i) {
        const float delta = h[i] - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden + 1.0e-5f);
    for (uint32_t i = 0; i < hidden; ++i) {
        float value = (h[i] - mean) * inv_std * norm_weight[i] + norm_bias[i];
        h[i] = value < 0.0f ? value * prelu : value;
    }
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t canonical = row;
        const float *weights = w2_output_major + size_t(canonical) * hidden;
        __m512 sum = _mm512_setzero_ps();
        for (uint32_t column = 0; column < hidden; column += 16) {
            sum = _mm512_fmadd_ps(
                _mm512_loadu_ps(weights + column), _mm512_load_ps(h + column), sum);
        }
        output[row] = _mm512_reduce_add_ps(sum) + b2[canonical];
    }
}


// Experimental child-local, unnormalized first-layer contribution. Bias,
// shared RMS, common features, layer normalization and activation stay at parent.
template<uint32_t H>
__attribute__((target("avx512f"), noinline))
static void partial_first(const float *w1, const float *values,
                          uint32_t inputs, float *out) {
    __m512 accum[H/16];
    for (uint32_t j=0;j<H/16;++j) accum[j]=_mm512_setzero_ps();
    for (uint32_t i=0;i<inputs;++i) {
        const __m512 v=_mm512_set1_ps(values[i]);
        for (uint32_t j=0;j<H/16;++j)
            accum[j]=_mm512_fmadd_ps(v,_mm512_loadu_ps(w1+size_t(i)*H+j*16),accum[j]);
    }
    for (uint32_t j=0;j<H/16;++j) _mm512_storeu_ps(out+j*16,accum[j]);
}
extern "C" __attribute__((target("avx512f")))
void sparse_nn_partial_first_f32(const float *w1, const float *values,
                                uint32_t inputs, uint32_t hidden, float *out) {
    if (hidden==64) partial_first<64>(w1,values,inputs,out);
    else if (hidden==128) partial_first<128>(w1,values,inputs,out);
    else {
        if (hidden>1024 || (hidden&15u)) __builtin_trap();
        for (uint32_t j=0;j<hidden;j+=16) {
            __m512 sum=_mm512_setzero_ps();
            for (uint32_t i=0;i<inputs;++i)
                sum=_mm512_fmadd_ps(_mm512_set1_ps(values[i]),
                    _mm512_loadu_ps(w1+size_t(i)*hidden+j),sum);
            _mm512_storeu_ps(out+j,sum);
        }
    }
}

// Experimental fixed-width kernel. Keep each neuron's input FMA order and
// each output's sixteen-lane reduction identical to static_dense.
template<uint32_t H>
__attribute__((target("avx512f"), noinline))
static void register_dense(
    const float *w1, const float *b1, const float *nw, const float *nb, float prelu,
    const float *w2, const float *b2, const float *values, uint32_t inputs,
    uint32_t outputs, float *out) {
    __m512 accum[H/16];
    for (uint32_t j=0;j<H/16;++j) accum[j]=_mm512_loadu_ps(b1+j*16);
    for (uint32_t i=0;i<inputs;++i) {
        const __m512 v=_mm512_set1_ps(values[i]);
        for (uint32_t j=0;j<H/16;++j)
            accum[j]=_mm512_fmadd_ps(v,_mm512_loadu_ps(w1+size_t(i)*H+j*16),accum[j]);
    }
    alignas(64) float h[H];
    for (uint32_t j=0;j<H/16;++j) _mm512_store_ps(h+j*16,accum[j]);
    float mean=0.f;
    for (uint32_t j=0;j<H;++j) mean+=h[j];
    mean/=H;
    float variance=0.f;
    for (uint32_t j=0;j<H;++j) { const float d=h[j]-mean;variance+=d*d; }
    const float inv=1.f/std::sqrt(variance/H+1.e-5f);
    for (uint32_t j=0;j<H;++j) {
        const float v=(h[j]-mean)*inv*nw[j]+nb[j];
        h[j]=v<0.f?v*prelu:v;
    }
    uint32_t r=0;
    for (;r+4<=outputs;r+=4) {
        __m512 a=_mm512_setzero_ps(),b=a,c=a,d=a;
        for (uint32_t j=0;j<H;j+=16) {
            const __m512 v=_mm512_load_ps(h+j);
            a=_mm512_fmadd_ps(_mm512_loadu_ps(w2+size_t(r)*H+j),v,a);
            b=_mm512_fmadd_ps(_mm512_loadu_ps(w2+size_t(r+1)*H+j),v,b);
            c=_mm512_fmadd_ps(_mm512_loadu_ps(w2+size_t(r+2)*H+j),v,c);
            d=_mm512_fmadd_ps(_mm512_loadu_ps(w2+size_t(r+3)*H+j),v,d);
        }
        out[r]=_mm512_reduce_add_ps(a)+b2[r];
        out[r+1]=_mm512_reduce_add_ps(b)+b2[r+1];
        out[r+2]=_mm512_reduce_add_ps(c)+b2[r+2];
        out[r+3]=_mm512_reduce_add_ps(d)+b2[r+3];
    }
    for (;r<outputs;++r) {
        __m512 a=_mm512_setzero_ps();
        for (uint32_t j=0;j<H;j+=16)
            a=_mm512_fmadd_ps(_mm512_loadu_ps(w2+size_t(r)*H+j),_mm512_load_ps(h+j),a);
        out[r]=_mm512_reduce_add_ps(a)+b2[r];
    }
}
extern "C" __attribute__((target("avx512f")))
void sparse_nn_forward_f32_register_dense(
    const float *w1,const float *b1,const float *nw,const float *nb,float prelu,
    const float *w2,const float *b2,uint32_t hidden,const float *values,
    uint32_t inputs,uint32_t outputs,float *out) {
    if(hidden==64) register_dense<64>(w1,b1,nw,nb,prelu,w2,b2,values,inputs,outputs,out);
    else if(hidden==128) register_dense<128>(w1,b1,nw,nb,prelu,w2,b2,values,inputs,outputs,out);
    else sparse_nn_forward_f32_static_dense(w1,b1,nw,nb,prelu,w2,b2,hidden,values,inputs,outputs,out);
}
