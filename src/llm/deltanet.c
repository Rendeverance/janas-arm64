/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deltanet.c - Gated DeltaNet step (see deltanet.h).
 */
#include "deltanet.h"

#include <math.h>

#include "quant.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

void janas_gdn_step_ref(float *M, const float *q, const float *k,
                        const float *v, float decay, float beta, float scale,
                        float *o, size_t dim, size_t rows)
{
    for (size_t j = 0; j < rows; j++) {
        float *r = M + j * dim, sk = 0.0f, sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            r[i] *= decay;
            sk += r[i] * k[i];
        }
        float d = (v[j] - sk) * beta;
        for (size_t i = 0; i < dim; i++) {
            r[i] += d * k[i];
            sq += r[i] * q[i];
        }
        o[j] = sq * scale;
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static inline float hsum(__m256 a)
{
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/*
 * Row by row: the row is decayed and dotted with k in the same loop, then
 * updated and dotted with q; it is read and written once per step.
 */
__attribute__((target("avx2,fma"))) static void
gdn_step_avx2(float *M, const float *q, const float *k, const float *v,
              float decay, float beta, float scale, float *o, size_t dim,
              size_t rows)
{
    __m256 vd = _mm256_set1_ps(decay);
    for (size_t j = 0; j < rows; j++) {
        float *r = M + j * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(r + i), vd);
            __m256 x1 = _mm256_mul_ps(_mm256_loadu_ps(r + i + 8), vd);
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(k + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(r + i), vd);
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
        }
        float d = (v[j] - hsum(_mm256_add_ps(a0, a1))) * beta;
        __m256 vdl = _mm256_set1_ps(d);
        a0 = _mm256_setzero_ps();
        a1 = _mm256_setzero_ps();
        for (i = 0; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            __m256 x1 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i + 8),
                                        _mm256_loadu_ps(r + i + 8));
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(q + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
        }
        o[j] = hsum(_mm256_add_ps(a0, a1)) * scale;
    }
}
#endif

static void gdn_step_h_ref(uint16_t *M, const float *q, const float *k,
                           const float *v, float decay, float beta, float scale,
                           float *o, size_t dim, size_t rows)
{
    float r[256];
    for (size_t j = 0; j < rows; j++) {
        uint16_t *m = M + j * dim;
        float sk = 0.0f, sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            r[i] = janas_fp16_to_fp32(m[i]) * decay;
            sk += r[i] * k[i];
        }
        float d = (v[j] - sk) * beta;
        for (size_t i = 0; i < dim; i++) {
            r[i] += d * k[i];
            sq += r[i] * q[i];
            m[i] = janas_fp32_to_fp16(r[i]);
        }
        o[j] = sq * scale;
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma,f16c"))) static void
gdn_step_h_avx2(uint16_t *M, const float *q, const float *k, const float *v,
                float decay, float beta, float scale, float *o, size_t dim,
                size_t rows)
{
    __m256 vd = _mm256_set1_ps(decay);
    float r[256];
    for (size_t j = 0; j < rows; j++) {
        uint16_t *m = M + j * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i))), vd);
            __m256 x1 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i + 8))),
                vd);
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(k + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i))), vd);
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
        }
        float d = (v[j] - hsum(_mm256_add_ps(a0, a1))) * beta;
        __m256 vdl = _mm256_set1_ps(d);
        a0 = _mm256_setzero_ps();
        a1 = _mm256_setzero_ps();
        for (i = 0; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            __m256 x1 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i + 8),
                                        _mm256_loadu_ps(r + i + 8));
            _mm_storeu_si128((__m128i *)(m + i), _mm256_cvtps_ph(x0, 0));
            _mm_storeu_si128((__m128i *)(m + i + 8), _mm256_cvtps_ph(x1, 0));
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(q + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            _mm_storeu_si128((__m128i *)(m + i), _mm256_cvtps_ph(x0, 0));
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
        }
        o[j] = hsum(_mm256_add_ps(a0, a1)) * scale;
    }
}
#endif

#if defined(__aarch64__)
#include <arm_neon.h>

#include "vmath.h"

/*
 * ARM64: the AVX2 kernels with two registers of four lanes for each of
 * eight (a[0] a[1] are their a0, a[2] a[3] their a1): the same 16- then
 * 8-wide loops, the same products in the same lanes, the same sums.
 */
static void gdn_step_neon(float *M, const float *q, const float *k,
                          const float *v, float decay, float beta, float scale,
                          float *o, size_t dim, size_t rows)
{
    float32x4_t vd = vdupq_n_f32(decay);
    for (size_t j = 0; j < rows; j++) {
        float *r = M + j * dim;
        float32x4_t a[4];
        for (int l = 0; l < 4; l++)
            a[l] = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16)
            for (int l = 0; l < 4; l++) {
                float32x4_t x = vmulq_f32(vld1q_f32(r + i + 4 * l), vd);
                vst1q_f32(r + i + 4 * l, x);
                a[l] = vfmaq_f32(a[l], x, vld1q_f32(k + i + 4 * l));
            }
        for (; i < dim; i += 8)
            for (int l = 0; l < 2; l++) {
                float32x4_t x = vmulq_f32(vld1q_f32(r + i + 4 * l), vd);
                vst1q_f32(r + i + 4 * l, x);
                a[l] = vfmaq_f32(a[l], x, vld1q_f32(k + i + 4 * l));
            }
        float sk = janas_hsum4x2(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3]));
        float d = (v[j] - sk) * beta;
        float32x4_t vdl = vdupq_n_f32(d);
        for (int l = 0; l < 4; l++)
            a[l] = vdupq_n_f32(0.0f);
        for (i = 0; i + 16 <= dim; i += 16)
            for (int l = 0; l < 4; l++) {
                float32x4_t x = vfmaq_f32(vld1q_f32(r + i + 4 * l), vdl,
                                          vld1q_f32(k + i + 4 * l));
                vst1q_f32(r + i + 4 * l, x);
                a[l] = vfmaq_f32(a[l], x, vld1q_f32(q + i + 4 * l));
            }
        for (; i < dim; i += 8)
            for (int l = 0; l < 2; l++) {
                float32x4_t x = vfmaq_f32(vld1q_f32(r + i + 4 * l), vdl,
                                          vld1q_f32(k + i + 4 * l));
                vst1q_f32(r + i + 4 * l, x);
                a[l] = vfmaq_f32(a[l], x, vld1q_f32(q + i + 4 * l));
            }
        o[j] =
            janas_hsum4x2(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3])) * scale;
    }
}

/* The state in fp16: NEON converts both ways in hardware (round to nearest
   even, as F16C and janas_fp32_to_fp16). */
static void gdn_step_h_neon(uint16_t *M, const float *q, const float *k,
                            const float *v, float decay, float beta,
                            float scale, float *o, size_t dim, size_t rows)
{
    float r[256];
    float32x4_t vd = vdupq_n_f32(decay);
    for (size_t j = 0; j < rows; j++) {
        uint16_t *m = M + j * dim;
        float32x4_t a[4];
        for (int l = 0; l < 4; l++)
            a[l] = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= dim; i += 16)
            for (int h = 0; h < 2; h++) {
                float16x8_t f = vreinterpretq_f16_u16(vld1q_u16(m + i + 8 * h));
                float32x4_t x0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f)), vd);
                float32x4_t x1 = vmulq_f32(vcvt_high_f32_f16(f), vd);
                vst1q_f32(r + i + 8 * h, x0);
                vst1q_f32(r + i + 8 * h + 4, x1);
                a[2 * h] = vfmaq_f32(a[2 * h], x0, vld1q_f32(k + i + 8 * h));
                a[2 * h + 1] =
                    vfmaq_f32(a[2 * h + 1], x1, vld1q_f32(k + i + 8 * h + 4));
            }
        for (; i < dim; i += 8) {
            float16x8_t f = vreinterpretq_f16_u16(vld1q_u16(m + i));
            float32x4_t x0 = vmulq_f32(vcvt_f32_f16(vget_low_f16(f)), vd);
            float32x4_t x1 = vmulq_f32(vcvt_high_f32_f16(f), vd);
            vst1q_f32(r + i, x0);
            vst1q_f32(r + i + 4, x1);
            a[0] = vfmaq_f32(a[0], x0, vld1q_f32(k + i));
            a[1] = vfmaq_f32(a[1], x1, vld1q_f32(k + i + 4));
        }
        float sk = janas_hsum4x2(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3]));
        float d = (v[j] - sk) * beta;
        float32x4_t vdl = vdupq_n_f32(d);
        for (int l = 0; l < 4; l++)
            a[l] = vdupq_n_f32(0.0f);
        for (i = 0; i + 16 <= dim; i += 16)
            for (int h = 0; h < 2; h++) {
                float32x4_t x0 = vfmaq_f32(vld1q_f32(r + i + 8 * h), vdl,
                                           vld1q_f32(k + i + 8 * h));
                float32x4_t x1 = vfmaq_f32(vld1q_f32(r + i + 8 * h + 4), vdl,
                                           vld1q_f32(k + i + 8 * h + 4));
                vst1q_u16(m + i + 8 * h,
                          vreinterpretq_u16_f16(
                              vcvt_high_f16_f32(vcvt_f16_f32(x0), x1)));
                a[2 * h] = vfmaq_f32(a[2 * h], x0, vld1q_f32(q + i + 8 * h));
                a[2 * h + 1] =
                    vfmaq_f32(a[2 * h + 1], x1, vld1q_f32(q + i + 8 * h + 4));
            }
        for (; i < dim; i += 8) {
            float32x4_t x0 = vfmaq_f32(vld1q_f32(r + i), vdl, vld1q_f32(k + i));
            float32x4_t x1 =
                vfmaq_f32(vld1q_f32(r + i + 4), vdl, vld1q_f32(k + i + 4));
            vst1q_u16(m + i, vreinterpretq_u16_f16(
                                 vcvt_high_f16_f32(vcvt_f16_f32(x0), x1)));
            a[0] = vfmaq_f32(a[0], x0, vld1q_f32(q + i));
            a[1] = vfmaq_f32(a[1], x1, vld1q_f32(q + i + 4));
        }
        o[j] =
            janas_hsum4x2(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3])) * scale;
    }
}
#endif

static int use_avx2;
#if defined(__x86_64__)
static int use_f16c;
#endif

__attribute__((constructor)) static void gdn_init(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
    use_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    use_f16c = __builtin_cpu_supports("f16c");
#endif
}

void janas_gdn_step(float *M, const float *q, const float *k, const float *v,
                    float decay, float beta, float scale, float *o, size_t dim,
                    size_t rows)
{
#if defined(__aarch64__)
    if (!janas_kernels_ref()) {
        gdn_step_neon(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
#if defined(__x86_64__)
    if (use_avx2) {
        gdn_step_avx2(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
    janas_gdn_step_ref(M, q, k, v, decay, beta, scale, o, dim, rows);
}

void janas_gdn_step_h(uint16_t *M, const float *q, const float *k,
                      const float *v, float decay, float beta, float scale,
                      float *o, size_t dim, size_t rows)
{
#if defined(__aarch64__)
    if (!janas_kernels_ref()) {
        gdn_step_h_neon(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
#if defined(__x86_64__)
    if (use_avx2 && use_f16c) {
        gdn_step_h_avx2(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
    gdn_step_h_ref(M, q, k, v, decay, beta, scale, o, dim, rows);
}

const char *janas_gdn_kernel_name(void)
{
#if defined(__aarch64__)
    return janas_kernels_ref() ? "ref" : "neon";
#endif
    return use_avx2 ? "avx2" : "ref";
}
