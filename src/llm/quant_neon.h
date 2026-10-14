/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Rendeverance */
/*
 * quant_neon.h - the quantized dot products for ARM64 (NEON + the dot-product
 * extension), private to quant.c, which includes it once.
 *
 * Each kernel mirrors its scalar reference in quant.c: the integer sums are
 * exact, so any order gives the same int32, and every float operation is the
 * reference's, in the reference's order (fused multiply-adds lane for lane,
 * the same final pairing of the eight lanes), so the results match it bit
 * for bit; tests/test_quant.c checks this.
 */
#include <arm_neon.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#define NEON_DP __attribute__((target("dotprod")))

/* The exact sum of 16 + 16 int8 products. */
static inline NEON_DP int32_t neon_dot32(int8x16_t w0, int8x16_t w1,
                                         const int8_t *a)
{
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(a));
    s = vdotq_s32(s, w1, vld1q_s8(a + 16));
    return vaddvq_s32(s);
}

/* The 32 weights of a 4-bit block as int8 levels: low nibbles are weights
   0-15, high nibbles 16-31 (the GGUF layout). */
static inline NEON_DP void neon_nib_unpack(const uint8_t *q, int8x16_t levels,
                                           int8x16_t *w0, int8x16_t *w1)
{
    uint8x16_t b = vld1q_u8(q);
    *w0 = vqtbl1q_s8(levels, vandq_u8(b, vdupq_n_u8(15)));
    *w1 = vqtbl1q_s8(levels, vshrq_n_u8(b, 4));
}

/* The eight lanes of acc (lanes 0-3 in a, 4-7 in b) added as the reference
   does: ((0 + 4) + (2 + 6)) + ((1 + 5) + (3 + 7)). */
static inline float neon_sum8(float32x4_t a, float32x4_t b)
{
    float32x4_t s = vaddq_f32(a, b);                       /* 0+4 1+5 2+6 3+7 */
    float32x2_t t = vadd_f32(vget_low_f32(s), vget_high_f32(s)); /* (0+4)+(2+6), (1+5)+(3+7) */
    return vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
}

/* IQ4_NL, Q4_0 and MXFP4 (scales passed per block): per block position k a
   lane, acc[k] = fma(scale_k * d32[k], dot_k, acc[k]). */
static NEON_DP float neon_nib_dot(const struct janas_block_iq4nl *w,
                                  const struct janas_block_q8k *x, size_t nb,
                                  const int8_t *lv)
{
    int8x16_t levels = vld1q_s8(lv);
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int32_t dot[8];
        float sc[8];
        for (int k = 0; k < 8; k++) {
            const struct janas_block_iq4nl *wk = w + 8 * b + k;
            int8x16_t w0, w1;
            neon_nib_unpack(wk->qs, levels, &w0, &w1);
            dot[k] = neon_dot32(w0, w1, x[b].qs32 + 32 * k);
            sc[k] = janas_fp16_to_fp32(wk->d) * x[b].d32[k];
        }
        acc0 = vfmaq_f32(acc0, vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(sc + 4),
                         vcvtq_f32_s32(vld1q_s32(dot + 4)));
    }
    return neon_sum8(acc0, acc1);
}

static NEON_DP float q4_0_dot_neon(const struct janas_block_q4_0 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_nib_dot((const struct janas_block_iq4nl *)w, x, nb,
                        janas_q4_0_values);
}

static NEON_DP float iq4nl_dot_neon(const struct janas_block_iq4nl *w,
                                    const struct janas_block_q8k *x, size_t nb)
{
    return neon_nib_dot(w, x, nb, janas_iq4_values);
}

static NEON_DP float q8_0_dot_neon(const struct janas_block_q8_0 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int32_t dot[8];
        float sc[8];
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q8_0 *wk = w + 8 * b + k;
            dot[k] = neon_dot32(vld1q_s8(wk->qs), vld1q_s8(wk->qs + 16),
                                x[b].qs32 + 32 * k);
            sc[k] = janas_fp16_to_fp32(wk->d) * x[b].d32[k];
        }
        acc0 = vfmaq_f32(acc0, vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
        acc1 = vfmaq_f32(acc1, vld1q_f32(sc + 4),
                         vcvtq_f32_s32(vld1q_s32(dot + 4)));
    }
    return neon_sum8(acc0, acc1);
}

/* Q4_1 (five = 0) and Q5_1 (five = 1): d x the exact weight sum and m x the
   activations' sum, per block position, as dm_dot_ref. */
static NEON_DP float neon_dm_dot(const uint8_t *w, size_t size, int five,
                                 const struct janas_block_q8k *x, size_t nb)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t m0 = vdupq_n_f32(0.0f), m1 = vdupq_n_f32(0.0f);
    const uint8x16_t lo4 = vdupq_n_u8(15);
    /* bit i of a 16-bit half of qh, spread to byte i */
    static const uint8_t bit_of[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                       1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t bits = vld1q_u8(bit_of);
    for (size_t b = 0; b < nb; b++) {
        int32_t dot[8], asum[8];
        float sc[8], sm[8];
        for (int k = 0; k < 8; k++) {
            const uint8_t *wk = w + (8 * b + k) * size;
            const uint8_t *qs = wk + (five ? 8 : 4);
            uint8x16_t q = vld1q_u8(qs);
            uint8x16_t v0 = vandq_u8(q, lo4), v1 = vshrq_n_u8(q, 4);
            if (five) {
                uint32_t qh;
                memcpy(&qh, wk + 4, 4);
                /* bytes 0-7 take qh bits 0-7, 8-15 bits 8-15 (and 16-31 for
                   the second half) */
                uint8x16_t h0 = vcombine_u8(vdup_n_u8((uint8_t)qh),
                                            vdup_n_u8((uint8_t)(qh >> 8)));
                uint8x16_t h1 = vcombine_u8(vdup_n_u8((uint8_t)(qh >> 16)),
                                            vdup_n_u8((uint8_t)(qh >> 24)));
                v0 = vorrq_u8(v0, vandq_u8(vtstq_u8(h0, bits), vdupq_n_u8(16)));
                v1 = vorrq_u8(v1, vandq_u8(vtstq_u8(h1, bits), vdupq_n_u8(16)));
            }
            /* unsigned weights 0-31 times signed activations: as int8, exact
               (both under 128) */
            dot[k] = neon_dot32(vreinterpretq_s8_u8(v0),
                                vreinterpretq_s8_u8(v1), x[b].qs32 + 32 * k);
            asum[k] = x[b].bsums32[2 * k] + x[b].bsums32[2 * k + 1];
            uint16_t d, m;
            memcpy(&d, wk, 2);
            memcpy(&m, wk + 2, 2);
            sc[k] = janas_fp16_to_fp32(d) * x[b].d32[k];
            sm[k] = janas_fp16_to_fp32(m) * x[b].d32[k];
        }
        a0 = vfmaq_f32(a0, vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
        a1 = vfmaq_f32(a1, vld1q_f32(sc + 4), vcvtq_f32_s32(vld1q_s32(dot + 4)));
        m0 = vfmaq_f32(m0, vld1q_f32(sm), vcvtq_f32_s32(vld1q_s32(asum)));
        m1 = vfmaq_f32(m1, vld1q_f32(sm + 4), vcvtq_f32_s32(vld1q_s32(asum + 4)));
    }
    return neon_sum8(a0, a1) + neon_sum8(m0, m1);
}

static NEON_DP float q4_1_dot_neon(const struct janas_block_q4_1 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_dm_dot((const uint8_t *)w, sizeof(*w), 0, x, nb);
}

static NEON_DP float q5_1_dot_neon(const struct janas_block_q5_1 *w,
                                   const struct janas_block_q8k *x, size_t nb)
{
    return neon_dm_dot((const uint8_t *)w, sizeof(*w), 1, x, nb);
}

/* Q6_K: per group of 16 the exact sum of (q - 32) x a, times the group's
   scale, all in one int32 per block; then one fma per block, as the
   reference. */
static NEON_DP float q6k_dot_neon(const struct janas_block_q6k *w,
                                  const struct janas_block_q8k *x, size_t nb)
{
    float acc = 0.0f;
    const uint8x16_t m4 = vdupq_n_u8(15), m2 = vdupq_n_u8(3);
    const int8x16_t c32 = vdupq_n_s8(32);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_q6k *wb = &w[b];
        const int8_t *a = x[b].qs;
        int32_t sumi = 0;
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = wb->ql + 64 * n, *qh = wb->qh + 32 * n;
            for (int h = 0; h < 2; h++) { /* 16 of the 32 l at a time */
                uint8x16_t l0 = vld1q_u8(ql + 16 * h);      /* part 0 / 2 */
                uint8x16_t l1 = vld1q_u8(ql + 32 + 16 * h); /* part 1 / 3 */
                uint8x16_t hh = vld1q_u8(qh + 16 * h);
                int8x16_t q[4];
                q[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4),
                           vshlq_n_u8(vandq_u8(hh, m2), 4))), c32);
                q[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m2), 4))), c32);
                q[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4),
                           vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m2), 4))), c32);
                q[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4),
                           vshlq_n_u8(vshrq_n_u8(hh, 6), 4))), c32);
                for (int p = 0; p < 4; p++) {
                    /* weights v = 128n + 32p + 16h .. +15: group v / 16 */
                    int v = 128 * n + 32 * p + 16 * h;
                    int32x4_t s = vdotq_s32(vdupq_n_s32(0), q[p], vld1q_s8(a + v));
                    sumi += wb->scales[v / 16] * vaddvq_s32(s);
                }
            }
        }
        acc = fmaf(janas_fp16_to_fp32(wb->d) * x[b].d, (float)sumi, acc);
    }
    return acc;
}

/* Whether the CPU has the dot-product instructions (SDOT/UDOT). */
static int neon_has_dotprod(void)
{
#if defined(_WIN32)
    return IsProcessorFeaturePresent(43 /* PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE */) != 0;
#elif defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return 0;
#endif
}

/*
 * The same products against nv activation vectors at once (nv 1 to 8), as
 * janas_dot_multi asks: each weight block is unpacked once and used for all
 * of them. Per vector the integer sums and the float operations are those of
 * the single-vector kernel above, in the same order, so every result is that
 * kernel's (and the reference's), bit for bit.
 */
static NEON_DP void neon_nib_dotn(const struct janas_block_iq4nl *w,
                                  const struct janas_block_q8k *x,
                                  size_t xs, int nv, size_t nb, float *out,
                                  const int8_t *lv)
{
    int8x16_t levels = vld1q_s8(lv);
    float32x4_t a0[8], a1[8];
    for (int v = 0; v < nv; v++)
        a0[v] = a1[v] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int8x16_t w0[8], w1[8];
        float d[8];
        for (int k = 0; k < 8; k++) {
            const struct janas_block_iq4nl *wk = w + 8 * b + k;
            neon_nib_unpack(wk->qs, levels, &w0[k], &w1[k]);
            d[k] = janas_fp16_to_fp32(wk->d);
        }
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            int32_t dot[8];
            float sc[8];
            for (int k = 0; k < 8; k++) {
                dot[k] = neon_dot32(w0[k], w1[k], xb->qs32 + 32 * k);
                sc[k] = d[k] * xb->d32[k];
            }
            a0[v] = vfmaq_f32(a0[v], vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
            a1[v] = vfmaq_f32(a1[v], vld1q_f32(sc + 4),
                              vcvtq_f32_s32(vld1q_s32(dot + 4)));
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = neon_sum8(a0[v], a1[v]);
}

static NEON_DP void q8_0_dotn_neon(const struct janas_block_q8_0 *w,
                                   const struct janas_block_q8k *x, size_t xs,
                                   int nv, size_t nb, float *out)
{
    float32x4_t a0[8], a1[8];
    for (int v = 0; v < nv; v++)
        a0[v] = a1[v] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        int8x16_t w0[8], w1[8];
        float d[8];
        for (int k = 0; k < 8; k++) {
            const struct janas_block_q8_0 *wk = w + 8 * b + k;
            w0[k] = vld1q_s8(wk->qs);
            w1[k] = vld1q_s8(wk->qs + 16);
            d[k] = janas_fp16_to_fp32(wk->d);
        }
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            int32_t dot[8];
            float sc[8];
            for (int k = 0; k < 8; k++) {
                dot[k] = neon_dot32(w0[k], w1[k], xb->qs32 + 32 * k);
                sc[k] = d[k] * xb->d32[k];
            }
            a0[v] = vfmaq_f32(a0[v], vld1q_f32(sc), vcvtq_f32_s32(vld1q_s32(dot)));
            a1[v] = vfmaq_f32(a1[v], vld1q_f32(sc + 4),
                              vcvtq_f32_s32(vld1q_s32(dot + 4)));
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = neon_sum8(a0[v], a1[v]);
}

/* Q6_K: the 256 weights of a block unpacked once (as int8, q - 32). */
static NEON_DP void q6k_dotn_neon(const struct janas_block_q6k *w,
                                  const struct janas_block_q8k *x, size_t xs,
                                  int nv, size_t nb, float *out)
{
    float acc[8] = {0};
    const uint8x16_t m4 = vdupq_n_u8(15), m2 = vdupq_n_u8(3);
    const int8x16_t c32 = vdupq_n_s8(32);
    for (size_t b = 0; b < nb; b++) {
        const struct janas_block_q6k *wb = &w[b];
        int8x16_t q[16]; /* q[i] holds weights 16i .. 16i + 15 */
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = wb->ql + 64 * n, *qh = wb->qh + 32 * n;
            for (int h = 0; h < 2; h++) {
                uint8x16_t l0 = vld1q_u8(ql + 16 * h), l1 = vld1q_u8(ql + 32 + 16 * h);
                uint8x16_t hh = vld1q_u8(qh + 16 * h);
                int base = 8 * n + h; /* v = 128n + 32p + 16h -> index 8n + 2p + h */
                q[base + 0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4),
                                  vshlq_n_u8(vandq_u8(hh, m2), 4))), c32);
                q[base + 2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m2), 4))), c32);
                q[base + 4] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4),
                                  vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m2), 4))), c32);
                q[base + 6] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4),
                                  vshlq_n_u8(vshrq_n_u8(hh, 6), 4))), c32);
            }
        }
        float d = janas_fp16_to_fp32(wb->d);
        for (int v = 0; v < nv; v++) {
            const struct janas_block_q8k *xb = x + v * xs + b;
            int32_t sumi = 0;
            for (int g = 0; g < 16; g++) {
                int32x4_t s = vdotq_s32(vdupq_n_s32(0), q[g], vld1q_s8(xb->qs + 16 * g));
                sumi += wb->scales[g] * vaddvq_s32(s);
            }
            acc[v] = fmaf(d * xb->d, (float)sumi, acc[v]);
        }
    }
    for (int v = 0; v < nv; v++)
        out[v] = acc[v];
}
