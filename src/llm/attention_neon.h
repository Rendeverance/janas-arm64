/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Rendeverance */
/*
 * attention_neon.h - the attention kernels for ARM64 (NEON), private to
 * attention.c, which includes it once.
 *
 * The sixteen-bit paths (int16 queries times int8 keys; int16 weights times
 * int8 values) sum exact integers, so they give the reference's numbers bit
 * for bit. The float paths sum each row in vector lanes, the weights with
 * janas_exp4 (four lanes of janas_exp8). head_dim is a multiple of 8 on
 * every model Janas runs; a tail is handled too.
 */
#include <arm_neon.h>

/* scores: int16 query, int8 keys, exact int32 sums */
static void att_scores_i16_neon(const int16_t *const *q, const int8_t *K,
                                const float *ks, uint32_t nt, uint32_t hd,
                                const float *qsc, float scale,
                                float *const *sc, int ns)
{
    for (uint32_t t = 0; t < nt; t++) {
        const int8_t *k = K + (size_t)t * hd;
        for (int s = 0; s < ns; s++) {
            const int16_t *qs = q[s];
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            uint32_t i = 0;
            for (; i + 8 <= hd; i += 8) {
                int16x8_t kk = vmovl_s8(vld1_s8(k + i));
                int16x8_t qq = vld1q_s16(qs + i);
                a0 = vmlal_s16(a0, vget_low_s16(qq), vget_low_s16(kk));
                a1 = vmlal_high_s16(a1, qq, kk);
            }
            int32_t sum = vaddvq_s32(vaddq_s32(a0, a1));
            for (; i < hd; i++)
                sum += (int32_t)qs[i] * k[i];
            sc[s][t] = (float)sum * qsc[s] * scale * ks[t];
        }
    }
}

/* scores: float query, int8 (kv16 = 0) or int16 (kv16 = 1) keys */
static void att_scores_f32_neon(const float *const *q, const void *Kv,
                                const float *ks, uint32_t nt, uint32_t hd,
                                float scale, float *const *sc, int ns,
                                int kv16)
{
    for (uint32_t t = 0; t < nt; t++)
        for (int s = 0; s < ns; s++) {
            const float *qs = q[s];
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            uint32_t i = 0;
            if (kv16) {
                const int16_t *k = (const int16_t *)Kv + (size_t)t * hd;
                for (; i + 8 <= hd; i += 8) {
                    int16x8_t kk = vld1q_s16(k + i);
                    a0 = vfmaq_f32(a0, vld1q_f32(qs + i),
                                   vcvtq_f32_s32(vmovl_s16(vget_low_s16(kk))));
                    a1 = vfmaq_f32(a1, vld1q_f32(qs + i + 4),
                                   vcvtq_f32_s32(vmovl_high_s16(kk)));
                }
                float sum = vaddvq_f32(vaddq_f32(a0, a1));
                for (; i < hd; i++)
                    sum += qs[i] * (float)k[i];
                sc[s][t] = sum * scale * ks[t];
            } else {
                const int8_t *k = (const int8_t *)Kv + (size_t)t * hd;
                for (; i + 8 <= hd; i += 8) {
                    int16x8_t kk = vmovl_s8(vld1_s8(k + i));
                    a0 = vfmaq_f32(a0, vld1q_f32(qs + i),
                                   vcvtq_f32_s32(vmovl_s16(vget_low_s16(kk))));
                    a1 = vfmaq_f32(a1, vld1q_f32(qs + i + 4),
                                   vcvtq_f32_s32(vmovl_high_s16(kk)));
                }
                float sum = vaddvq_f32(vaddq_f32(a0, a1));
                for (; i < hd; i++)
                    sum += qs[i] * (float)k[i];
                sc[s][t] = sum * scale * ks[t];
            }
        }
}

/* values: float weights, int8 or int16 values; out[s][i] = sum_t w_t V[t][i] */
static void att_values_f32_neon(const float *const *w, const void *Vv,
                                uint32_t nt, uint32_t hd, float *const *out,
                                int ns, int kv16)
{
    for (int s = 0; s < ns; s++) {
        uint32_t i = 0;
        for (; i + 8 <= hd; i += 8) {
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            for (uint32_t t = 0; t < nt; t++) {
                int16x8_t vv =
                    kv16 ? vld1q_s16((const int16_t *)Vv + (size_t)t * hd + i)
                         : vmovl_s8(vld1_s8((const int8_t *)Vv + (size_t)t * hd + i));
                float32x4_t wt = vdupq_n_f32(w[s][t]);
                a0 = vfmaq_f32(a0, wt, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vv))));
                a1 = vfmaq_f32(a1, wt, vcvtq_f32_s32(vmovl_high_s16(vv)));
            }
            vst1q_f32(out[s] + i, a0);
            vst1q_f32(out[s] + i + 4, a1);
        }
        for (; i < hd; i++) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < nt; t++)
                acc += w[s][t] * (kv16 ? (float)((const int16_t *)Vv)[(size_t)t * hd + i]
                                       : (float)((const int8_t *)Vv)[(size_t)t * hd + i]);
            out[s][i] = acc;
        }
    }
}

/* values: int16 weights (two a word), int8 values, exact int32 sums */
static void att_values_i16_neon(const int32_t *const *wp, const float *wsc,
                                const int8_t *V, uint32_t nt, uint32_t hd,
                                float *const *out, int ns)
{
    for (int s = 0; s < ns; s++) {
        const int16_t *w = (const int16_t *)wp[s]; /* little-endian: t in order */
        uint32_t i = 0;
        for (; i + 8 <= hd; i += 8) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            for (uint32_t t = 0; t < nt; t++) {
                int16x8_t vv = vmovl_s8(vld1_s8(V + (size_t)t * hd + i));
                a0 = vmlal_n_s16(a0, vget_low_s16(vv), w[t]);
                a1 = vmlal_n_s16(a1, vget_high_s16(vv), w[t]);
            }
            vst1q_f32(out[s] + i, vmulq_n_f32(vcvtq_f32_s32(a0), wsc[s]));
            vst1q_f32(out[s] + i + 4, vmulq_n_f32(vcvtq_f32_s32(a1), wsc[s]));
        }
        for (; i < hd; i++) {
            int32_t acc = 0;
            for (uint32_t t = 0; t < nt; t++)
                acc += (int32_t)w[t] * V[(size_t)t * hd + i];
            out[s][i] = (float)acc * wsc[s];
        }
    }
}

/* att_wquant_ref on NEON: the largest weight (exact), then to 0 .. 32767
   rounded to nearest even (vcvtnq, as lrintf), packed two a word. */
static void att_wquant_neon(float *row, uint32_t nt)
{
    float32x4_t vm = vdupq_n_f32(0.0f);
    uint32_t t = 0;
    for (; t + 4 <= nt; t += 4)
        vm = vmaxq_f32(vm, vld1q_f32(row + t));
    float mx = vmaxvq_f32(vm);
    for (; t < nt; t++)
        if (row[t] > mx)
            mx = row[t];
    float inv = mx > 0.0f ? 32767.0f / mx : 0.0f;
    int32_t *wp = (int32_t *)row;
    float32x4_t vi = vdupq_n_f32(inv);
    for (t = 0; t + 8 <= nt; t += 8) {
        int32x4_t a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + t), vi));
        int32x4_t b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(row + t + 4), vi));
        /* positions t..t+7 as eight int16, written over the first four words */
        int16x8_t p = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
        vst1q_s32(wp + t / 2, vreinterpretq_s32_s16(p));
    }
    att_wquant_tail(row, t, nt, inv);
    row[ATT_CHUNK - 1] = mx / 32767.0f;
}

/* The weights of a row and their sum, as att_weights_avx2 does (janas_exp4,
   the same final pairing of eight lanes). */
static float att_weights_neon(float *row, uint32_t nt, uint32_t vis, float *mx)
{
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    uint32_t t = 0;
    for (; t + 4 <= vis; t += 4)
        vm = vmaxq_f32(vm, vld1q_f32(row + t));
    float m = vmaxvq_f32(vm);
    for (; t < vis; t++)
        if (row[t] > m)
            m = row[t];
    float32x4_t vmx = vdupq_n_f32(m), lim = vdupq_n_f32(-80.0f);
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    static const uint32_t idx_[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32x4_t i0 = vld1q_u32(idx_), i1 = vld1q_u32(idx_ + 4);
    uint32x4_t vv = vdupq_n_u32(vis);
    for (t = 0; t < nt; t += 8) { /* row has room for a multiple of 8 */
        uint32x4_t tt = vdupq_n_u32(t);
        float32x4_t d0 = vsubq_f32(vld1q_f32(row + t), vmx);
        float32x4_t d1 = vsubq_f32(vld1q_f32(row + t + 4), vmx);
        uint32x4_t k0 = vandq_u32(vcgeq_f32(d0, lim), vcgtq_u32(vv, vaddq_u32(i0, tt)));
        uint32x4_t k1 = vandq_u32(vcgeq_f32(d1, lim), vcgtq_u32(vv, vaddq_u32(i1, tt)));
        float32x4_t w0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(janas_exp4(d0)), k0));
        float32x4_t w1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(janas_exp4(d1)), k1));
        vst1q_f32(row + t, w0);
        vst1q_f32(row + t + 4, w1);
        acc0 = vaddq_f32(acc0, w0);
        acc1 = vaddq_f32(acc1, w1);
    }
    *mx = m;
    float32x4_t s4 = vaddq_f32(acc0, acc1); /* 0+4 1+5 2+6 3+7 */
    float32x2_t s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
    return vget_lane_f32(s2, 0) + vget_lane_f32(s2, 1);
}
