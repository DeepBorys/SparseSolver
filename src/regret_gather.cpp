#include <cstddef>
#include <cstdint>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

static void scalar_copy(const float* base, float* dst, const uint32_t* indices, size_t count) {
    for (size_t i = 0; i < count; ++i)
        dst[i] = indices[i] == UINT32_MAX ? 0.0f : base[indices[i]];
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f")))
static void avx512_copy(const float* base, float* dst, const uint32_t* indices, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const auto ids = _mm512_loadu_si512(indices + i);
        const auto valid = _mm512_cmpneq_epi32_mask(ids, _mm512_set1_epi32(-1));
        const auto value = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), valid, ids, base, 4);
        _mm512_storeu_ps(dst + i, value);
    }
    scalar_copy(base, dst + i, indices + i, count - i);
}

__attribute__((target("avx2")))
static void avx2_copy(const float* base, float* dst, const uint32_t* indices, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto ids = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
        const auto valid = _mm256_xor_si256(_mm256_cmpeq_epi32(ids, _mm256_set1_epi32(-1)), _mm256_set1_epi32(-1));
        const auto value = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), base, ids, _mm256_castsi256_ps(valid), 4);
        _mm256_storeu_ps(dst + i, value);
    }
    scalar_copy(base, dst + i, indices + i, count - i);
}
#endif

using Copy = void (*)(const float*, float*, const uint32_t*, size_t);
static Copy select_copy() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx512f")) return avx512_copy;
    if (__builtin_cpu_supports("avx2")) return avx2_copy;
#endif
    return scalar_copy;
}

extern "C" void pfs_k1_regret_gather(const float* base, float* dst, const uint32_t* indices, size_t count) {
    static Copy copy = select_copy();
    copy(base, dst, indices, count);
}

extern "C" const char* pfs_k1_regret_gather_backend() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx512f")) return "avx512";
    if (__builtin_cpu_supports("avx2")) return "avx2";
#endif
    return "scalar";
}

static void shared_scalar(const float* base, float* dst, const uint32_t* sources,
                          const uint16_t* hands, const uint8_t* slots, size_t n, size_t actions) {
    for (size_t a = 0; a < actions; ++a)
        for (size_t h = 0; h < n; ++h)
            dst[a*n+h] = hands[h] == UINT16_MAX ? 0.0f : base[sources[slots ? slots[h] : 0] + a*n + hands[h]];
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f")))
static void shared_avx512(const float* base, float* dst, const uint32_t* sources,
                         const uint16_t* hands, const uint8_t* slots, size_t n, size_t actions) {
    const auto offsets = _mm512_setr_epi32(sources[0], sources[1], sources[2], sources[3], 0,0,0,0,0,0,0,0,0,0,0,0);
    for (size_t a = 0; a < actions; ++a) {
        size_t h = 0;
        for (; h + 16 <= n; h += 16) {
            const auto hand = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hands+h)));
            auto source = _mm512_set1_epi32(sources[0]);
            if (slots) source = _mm512_permutexvar_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(slots+h))), offsets);
            const auto ids = _mm512_add_epi32(hand, _mm512_add_epi32(source, _mm512_set1_epi32(a*n)));
            const auto valid = _mm512_cmpneq_epi32_mask(hand, _mm512_set1_epi32(UINT16_MAX));
            const auto value = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), valid, ids, base, 4);
            _mm512_storeu_ps(dst+a*n+h, value);
        }
        for (; h < n; ++h)
            dst[a*n+h] = hands[h] == UINT16_MAX ? 0.0f : base[sources[slots ? slots[h] : 0]+a*n+hands[h]];
    }
}

__attribute__((target("avx2")))
static void shared_avx2(const float* base, float* dst, const uint32_t* sources,
                       const uint16_t* hands, const uint8_t* slots, size_t n, size_t actions) {
    const auto offsets = _mm256_setr_epi32(sources[0],sources[1],sources[2],sources[3],0,0,0,0);
    for (size_t a = 0; a < actions; ++a) {
        size_t h = 0;
        for (; h + 8 <= n; h += 8) {
            const auto hand = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(hands+h)));
            auto source = _mm256_set1_epi32(sources[0]);
            if (slots) source = _mm256_permutevar8x32_epi32(offsets,_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(slots+h))));
            const auto ids = _mm256_add_epi32(hand,_mm256_add_epi32(source,_mm256_set1_epi32(a*n)));
            const auto valid = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(hand,_mm256_set1_epi32(UINT16_MAX)),_mm256_set1_epi32(-1)));
            const auto value = _mm256_mask_i32gather_ps(_mm256_setzero_ps(),base,ids,valid,4);
            _mm256_storeu_ps(dst+a*n+h,value);
        }
        for (; h < n; ++h)
            dst[a*n+h] = hands[h] == UINT16_MAX ? 0.0f : base[sources[slots ? slots[h] : 0]+a*n+hands[h]];
    }
}
#endif

using SharedCopy = void (*)(const float*,float*,const uint32_t*,const uint16_t*,const uint8_t*,size_t,size_t);
static SharedCopy select_shared() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx512f")) return shared_avx512;
    if (__builtin_cpu_supports("avx2")) return shared_avx2;
#endif
    return shared_scalar;
}

extern "C" void pfs_k1_shared_regret_gather(const float* base,float* dst,const uint32_t* sources,
                                           const uint16_t* hands,const uint8_t* slots,size_t n,size_t actions) {
    static SharedCopy copy = select_shared();
    copy(base,dst,sources,hands,slots,n,actions);
}

// Source pools above four boards use a second vector gather for source offsets.
// The existing one-to-four-board kernels remain untouched.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f")))
static void shared_many_avx512(const float* base,float* dst,const uint32_t* sources,
 const uint16_t* hands,const uint8_t* slots,size_t n,size_t actions) {
 for(size_t a=0;a<actions;++a){size_t h=0;
  for(;h+16<=n;h+=16){
   const auto hand=_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(hands+h)));
   const auto slot=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(slots+h)));
   const auto source=_mm512_i32gather_epi32(slot,sources,4);
   const auto ids=_mm512_add_epi32(hand,_mm512_add_epi32(source,_mm512_set1_epi32(a*n)));
   const auto valid=_mm512_cmpneq_epi32_mask(hand,_mm512_set1_epi32(UINT16_MAX));
   _mm512_storeu_ps(dst+a*n+h,_mm512_mask_i32gather_ps(_mm512_setzero_ps(),valid,ids,base,4));
  }
  for(;h<n;++h)dst[a*n+h]=hands[h]==UINT16_MAX?0.0f:base[sources[slots[h]]+a*n+hands[h]];
 }
}
__attribute__((target("avx2")))
static void shared_many_avx2(const float* base,float* dst,const uint32_t* sources,
 const uint16_t* hands,const uint8_t* slots,size_t n,size_t actions) {
 for(size_t a=0;a<actions;++a){size_t h=0;
  for(;h+8<=n;h+=8){
   const auto hand=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(hands+h)));
   const auto slot=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(slots+h)));
   const auto source=_mm256_i32gather_epi32(reinterpret_cast<const int*>(sources),slot,4);
   const auto ids=_mm256_add_epi32(hand,_mm256_add_epi32(source,_mm256_set1_epi32(a*n)));
   const auto valid=_mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(hand,_mm256_set1_epi32(UINT16_MAX)),_mm256_set1_epi32(-1)));
   _mm256_storeu_ps(dst+a*n+h,_mm256_mask_i32gather_ps(_mm256_setzero_ps(),base,ids,valid,4));
  }
  for(;h<n;++h)dst[a*n+h]=hands[h]==UINT16_MAX?0.0f:base[sources[slots[h]]+a*n+hands[h]];
 }
}
#endif
static SharedCopy select_shared_many(){
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
 if(__builtin_cpu_supports("avx512f"))return shared_many_avx512;
 if(__builtin_cpu_supports("avx2"))return shared_many_avx2;
#endif
 return shared_scalar;
}
extern "C" void pfs_k1_shared_regret_gather_many(const float* base,float* dst,const uint32_t* sources,
 const uint16_t* hands,const uint8_t* slots,size_t n,size_t actions,size_t source_count){
 if(source_count<=4){pfs_k1_shared_regret_gather(base,dst,sources,hands,slots,n,actions);return;}
 static SharedCopy copy=select_shared_many();copy(base,dst,sources,hands,slots,n,actions);
}
