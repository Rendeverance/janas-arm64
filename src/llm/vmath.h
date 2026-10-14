/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vmath.h - vector math helpers shared by the kernels (private).
 */
#ifndef JANAS_LLM_VMATH_H
#define JANAS_LLM_VMATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * Deterministic functions: made only of operations that every IEEE-754
 * implementation rounds the same way (add, multiply, fused multiply-add,
 * round to nearest even, bit moves), so that the scalar code, the AVX2 code
 * and the GPU shaders give the same bits. Used where a GPU must reproduce
 * the CPU exactly (the experts' activation). No division (GPUs may round
 * it differently), no library exp.
 */
static inline uint32_t janas_f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static inline float janas_u2f(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* e^x as janas_exp8 below, one lane: x clamped to [-88, 88]. */
static inline float janas_exp_det(float x)
{
    x = fminf(fmaxf(x, -88.0f), 88.0f);
    float n = nearbyintf(x * 1.44269504088896341f);
    float r = fmaf(-n, 0.693359375f, x);
    r = fmaf(-n, -2.12194440e-4f, r);
    float y = 1.9875691500e-4f;
    y = fmaf(y, r, 1.3981999507e-3f);
    y = fmaf(y, r, 8.3334519073e-3f);
    y = fmaf(y, r, 4.1665795894e-2f);
    y = fmaf(y, r, 1.6666665459e-1f);
    y = fmaf(y, r, 5.0000001201e-1f);
    y = fmaf(y, r * r, r);
    y = y + 1.0f;
    return y * janas_u2f((uint32_t)((int32_t)n + 127) << 23);
}

/* 1/d for normal positive d: a bit-level estimate, three Newton steps. */
static inline float janas_recip_det(float d)
{
    float y = janas_u2f(0x7EF311C3u - janas_f2u(d));
    for (int i = 0; i < 3; i++)
        y = fmaf(y, fmaf(-d, y, 1.0f), y);
    return y;
}

/* SiLU(g) * u = g / (1 + e^-g) * u; e^-g clamped to e^30 so that the
   reciprocal stays a normal number. */
static inline float janas_silu_mul_det(float g, float u)
{
    float e = janas_exp_det(fminf(fmaxf(-g, -88.0f), 30.0f));
    return g * janas_recip_det(1.0f + e) * u;
}

/*
 * GELU(g) * u, the tanh form: 0.5 g (1 + tanh(z)), z = sqrt(2/pi) (g +
 * 0.044715 g^3), which is g / (1 + e^-2z) - SiLU's shape with 2z for g;
 * e^-2z clamped the same way.
 */
static inline float janas_gelu_mul_det(float g, float u)
{
    float z2 = 1.5957691216057308f * fmaf(0.044715f * g, g * g, g);
    float e = janas_exp_det(fminf(fmaxf(-z2, -88.0f), 30.0f));
    return g * janas_recip_det(1.0f + e) * u;
}

#if defined(__x86_64__)
#include <immintrin.h>

/*
 * e^x, x clamped to [-88, 88]: x = n ln2 + r, e^r by a degree-6 polynomial
 * (Cephes coefficients), 2^n put into the exponent; about 1 ulp.
 */
__attribute__((target("avx2,fma"))) static inline __m256 janas_exp8(__m256 x)
{
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.0f)),
                      _mm256_set1_ps(88.0f));
    __m256 n =
        _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

/* janas_silu_mul_det on 8 lanes: the same operations, the same order. */
__attribute__((target("avx2,fma"))) static inline __m256
janas_silu_mul_det8(__m256 g, __m256 u)
{
    __m256 a =
        _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), g),
                                    _mm256_set1_ps(-88.0f)),
                      _mm256_set1_ps(30.0f));
    __m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f), janas_exp8(a));
    __m256 y = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(0x7EF311C3), _mm256_castps_si256(d)));
    for (int i = 0; i < 3; i++)
        y = _mm256_fmadd_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(1.0f)), y);
    return _mm256_mul_ps(_mm256_mul_ps(g, y), u);
}

/* janas_gelu_mul_det on 8 lanes: the same operations, the same order. */
__attribute__((target("avx2,fma"))) static inline __m256
janas_gelu_mul_det8(__m256 g, __m256 u)
{
    __m256 z2 = _mm256_mul_ps(
        _mm256_set1_ps(1.5957691216057308f),
        _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715f), g),
                        _mm256_mul_ps(g, g), g));
    __m256 a =
        _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), z2),
                                    _mm256_set1_ps(-88.0f)),
                      _mm256_set1_ps(30.0f));
    __m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f), janas_exp8(a));
    __m256 y = _mm256_castsi256_ps(_mm256_sub_epi32(
        _mm256_set1_epi32(0x7EF311C3), _mm256_castps_si256(d)));
    for (int i = 0; i < 3; i++)
        y = _mm256_fmadd_ps(y, _mm256_fnmadd_ps(d, y, _mm256_set1_ps(1.0f)), y);
    return _mm256_mul_ps(_mm256_mul_ps(g, y), u);
}

/* x / (1 + e^-x) */
__attribute__((target("avx2,fma"))) static inline __m256 janas_silu8(__m256 x)
{
    __m256 e = janas_exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

/* Sum of the 8 lanes, in a fixed order. */
__attribute__((target("avx2,fma"))) static inline float janas_hsum8(__m256 a)
{
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
/* janas_exp_det on 4 lanes: the same operations, the same order (fmaxnm/
   fminnm are fmaxf/fminf, round-to-nearest-even is nearbyintf). */
static inline float32x4_t janas_exp4(float32x4_t x)
{
    x = vminnmq_f32(vmaxnmq_f32(x, vdupq_n_f32(-88.0f)), vdupq_n_f32(88.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, r);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, r);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, r);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, r);
    y = vfmaq_f32(r, y, vmulq_f32(r, r));
    y = vaddq_f32(y, vdupq_n_f32(1.0f));
    int32x4_t e =
        vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

/* x / (1 + e^-x) on 4 lanes, as janas_silu8 (a real division). */
static inline float32x4_t janas_silu4(float32x4_t x)
{
    float32x4_t e = janas_exp4(vnegq_f32(x));
    return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e));
}

/* janas_recip_det on 4 lanes. */
static inline float32x4_t janas_recip4(float32x4_t d)
{
    float32x4_t y = vreinterpretq_f32_u32(
        vsubq_u32(vdupq_n_u32(0x7EF311C3u), vreinterpretq_u32_f32(d)));
    for (int i = 0; i < 3; i++)
        y = vfmaq_f32(y, y, vfmsq_f32(vdupq_n_f32(1.0f), d, y));
    return y;
}

/* janas_silu_mul_det on 4 lanes: the same operations, the same order. */
static inline float32x4_t janas_silu_mul_det4(float32x4_t g, float32x4_t u)
{
    float32x4_t a = vminnmq_f32(vmaxnmq_f32(vnegq_f32(g), vdupq_n_f32(-88.0f)),
                                vdupq_n_f32(30.0f));
    float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f), janas_exp4(a));
    return vmulq_f32(vmulq_f32(g, janas_recip4(d)), u);
}

/* janas_gelu_mul_det on 4 lanes: the same operations, the same order. */
static inline float32x4_t janas_gelu_mul_det4(float32x4_t g, float32x4_t u)
{
    float32x4_t z2 = vmulq_f32(
        vdupq_n_f32(1.5957691216057308f),
        vfmaq_f32(g, vmulq_f32(vdupq_n_f32(0.044715f), g), vmulq_f32(g, g)));
    float32x4_t a = vminnmq_f32(vmaxnmq_f32(vnegq_f32(z2), vdupq_n_f32(-88.0f)),
                                vdupq_n_f32(30.0f));
    float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f), janas_exp4(a));
    return vmulq_f32(vmulq_f32(g, janas_recip4(d)), u);
}
#endif

#endif
