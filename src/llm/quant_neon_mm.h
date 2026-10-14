/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Rendeverance */
/*
 * quant_neon_mm.h - two weight rows against two vectors at once with the
 * 8-bit matrix multiply of ARM64 (SMMLA, the I8MM extension), for the
 * 4-bit formats in blocks of 32 (Q4_0, IQ4_NL); private to quant.c, after
 * quant_neon.h. SMMLA takes 8 weights of each of two rows and the same 8
 * activations of each of two vectors and adds the four products' sums into
 * one int32 vector, [row 0 x vector 0, r0 x v1, r1 x v0, r1 x v1]: a block's
 * four exact integer dots in four instructions where SDOT needs eight. The
 * activations of the two vectors are interleaved once (neon_mm_interleave)
 * and serve every pair of rows. The integers are exact and the float steps
 * are the single-vector kernel's, lane for lane: every result is the
 * reference's, bit for bit. Q8_0 stays on SDOT: its weights need no
 * unpacking to share, so the interleave and transposes cost more than the
 * halved multiply count saves.
 */

/* Rows of at most this many blocks of 256 (8192 columns): the interleaved
   activations of two vectors, 512 bytes a block, on the stack. */
#define NEON_MM_MAX_NB 32

/* The activations of vectors a and b for SMMLA: per block of 32, four
   groups of 16 bytes, 8 of a then the same 8 of b, for the weights 0-7,
   16-23, 8-15 and 24-31 (the order the 4-bit blocks unpack in). */
static void neon_mm_interleave(const struct janas_block_q8k *a,
                               const struct janas_block_q8k *b, size_t nb,
                               int8_t *xi)
{
    for (size_t s = 0; s < nb; s++)
        for (int k = 0; k < 8; k++) {
            const int8_t *p = a[s].qs32 + 32 * k, *q = b[s].qs32 + 32 * k;
            int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(p));
            int64x2_t a1 = vreinterpretq_s64_s8(vld1q_s8(p + 16));
            int64x2_t b0 = vreinterpretq_s64_s8(vld1q_s8(q));
            int64x2_t b1 = vreinterpretq_s64_s8(vld1q_s8(q + 16));
            int8_t *o = xi + (s * 8 + (size_t)k) * 64;
            vst1q_s8(o, vreinterpretq_s8_s64(vzip1q_s64(a0, b0)));
            vst1q_s8(o + 16, vreinterpretq_s8_s64(vzip1q_s64(a1, b1)));
            vst1q_s8(o + 32, vreinterpretq_s8_s64(vzip2q_s64(a0, b0)));
            vst1q_s8(o + 48, vreinterpretq_s8_s64(vzip2q_s64(a1, b1)));
        }
}

/* One block's four dots: a0-a3 the two rows' weights in the order above. */
static inline NEON_MM int32x4_t neon_mm_block(int8x16_t a0, int8x16_t a1,
                                              int8x16_t a2, int8x16_t a3,
                                              const int8_t *g)
{
    int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), a0, vld1q_s8(g));
    s = vmmlaq_s32(s, a1, vld1q_s8(g + 16));
    s = vmmlaq_s32(s, a2, vld1q_s8(g + 32));
    return vmmlaq_s32(s, a3, vld1q_s8(g + 48));
}

/* The eight blocks' dots of a group of 256, s[k] = [r0v0 r0v1 r1v0 r1v1],
   into acc[pair][half] as the single-vector kernel does per pair: lane k of
   a half is block k (0-3) or 4 + k, fma(d_w * d32, dot, acc). Pairs 0-3:
   r0v0, r0v1, r1v0, r1v1. */
static inline NEON_MM void neon_mm_acc(const int32x4_t s[8], float32x4_t d0[2],
                                       float32x4_t d1[2], float32x4_t xa[2],
                                       float32x4_t xb[2], float32x4_t acc[4][2])
{
    for (int h = 0; h < 2; h++) {
        const int32x4_t *q = s + 4 * h;
        /* the 4x4 transpose: pair p's dots of blocks 4h .. 4h + 3 */
        int64x2_t t0 = vreinterpretq_s64_s32(vtrn1q_s32(q[0], q[1]));
        int64x2_t t1 = vreinterpretq_s64_s32(vtrn2q_s32(q[0], q[1]));
        int64x2_t t2 = vreinterpretq_s64_s32(vtrn1q_s32(q[2], q[3]));
        int64x2_t t3 = vreinterpretq_s64_s32(vtrn2q_s32(q[2], q[3]));
        int32x4_t p0 = vreinterpretq_s32_s64(vzip1q_s64(t0, t2));
        int32x4_t p1 = vreinterpretq_s32_s64(vzip1q_s64(t1, t3));
        int32x4_t p2 = vreinterpretq_s32_s64(vzip2q_s64(t0, t2));
        int32x4_t p3 = vreinterpretq_s32_s64(vzip2q_s64(t1, t3));
        acc[0][h] =
            vfmaq_f32(acc[0][h], vmulq_f32(d0[h], xa[h]), vcvtq_f32_s32(p0));
        acc[1][h] =
            vfmaq_f32(acc[1][h], vmulq_f32(d0[h], xb[h]), vcvtq_f32_s32(p1));
        acc[2][h] =
            vfmaq_f32(acc[2][h], vmulq_f32(d1[h], xa[h]), vcvtq_f32_s32(p2));
        acc[3][h] =
            vfmaq_f32(acc[3][h], vmulq_f32(d1[h], xb[h]), vcvtq_f32_s32(p3));
    }
}

/* Q4_0 and IQ4_NL (levels lv): rows w0, w1 against vectors a, b. */
static NEON_MM void neon_mm_nib(const struct janas_block_iq4nl *w0,
                                const struct janas_block_iq4nl *w1,
                                const struct janas_block_q8k *a,
                                const struct janas_block_q8k *b,
                                const int8_t *xi, size_t nb, const int8_t *lv,
                                float out[4])
{
    const int8x16_t levels = vld1q_s8(lv);
    const uint8x16_t m4 = vdupq_n_u8(15);
    float32x4_t acc[4][2];
    for (int p = 0; p < 4; p++)
        acc[p][0] = acc[p][1] = vdupq_n_f32(0.0f);
    for (size_t s = 0; s < nb; s++) {
        const struct janas_block_iq4nl *r0 = w0 + 8 * s, *r1 = w1 + 8 * s;
        int32x4_t dots[8];
        for (int k = 0; k < 8; k++) {
            uint64x2_t q0 = vreinterpretq_u64_u8(vld1q_u8(r0[k].qs));
            uint64x2_t q1 = vreinterpretq_u64_u8(vld1q_u8(r1[k].qs));
            /* bytes 0-7 of both rows hold weights 0-7 (low nibbles) and
               16-23 (high), bytes 8-15 weights 8-15 and 24-31 */
            uint8x16_t lo = vreinterpretq_u8_u64(vzip1q_u64(q0, q1));
            uint8x16_t hi = vreinterpretq_u8_u64(vzip2q_u64(q0, q1));
            dots[k] = neon_mm_block(vqtbl1q_s8(levels, vandq_u8(lo, m4)),
                                    vqtbl1q_s8(levels, vshrq_n_u8(lo, 4)),
                                    vqtbl1q_s8(levels, vandq_u8(hi, m4)),
                                    vqtbl1q_s8(levels, vshrq_n_u8(hi, 4)),
                                    xi + (s * 8 + (size_t)k) * 64);
        }
        float32x4_t d0[2] = {neon_h2f4(&r0[0].d, sizeof(*r0)),
                             neon_h2f4(&r0[4].d, sizeof(*r0))};
        float32x4_t d1[2] = {neon_h2f4(&r1[0].d, sizeof(*r1)),
                             neon_h2f4(&r1[4].d, sizeof(*r1))};
        float32x4_t xa[2] = {vld1q_f32(a[s].d32), vld1q_f32(a[s].d32 + 4)};
        float32x4_t xb[2] = {vld1q_f32(b[s].d32), vld1q_f32(b[s].d32 + 4)};
        neon_mm_acc(dots, d0, d1, xa, xb, acc);
    }
    for (int p = 0; p < 4; p++)
        out[p] = neon_sum8(acc[p][0], acc[p][1]);
}

/* A pair of rows unpacked once for several pairs of vectors: per block of
   32 the four operands neon_mm_nib makes, 64 bytes, into wi. */
static NEON_MM void neon_mm_unpack(const struct janas_block_iq4nl *w0,
                                   const struct janas_block_iq4nl *w1,
                                   size_t nb, const int8_t *lv, int8_t *wi)
{
    const int8x16_t levels = vld1q_s8(lv);
    const uint8x16_t m4 = vdupq_n_u8(15);
    for (size_t k = 0; k < 8 * nb; k++) {
        uint64x2_t q0 = vreinterpretq_u64_u8(vld1q_u8(w0[k].qs));
        uint64x2_t q1 = vreinterpretq_u64_u8(vld1q_u8(w1[k].qs));
        uint8x16_t lo = vreinterpretq_u8_u64(vzip1q_u64(q0, q1));
        uint8x16_t hi = vreinterpretq_u8_u64(vzip2q_u64(q0, q1));
        int8_t *o = wi + 64 * k;
        vst1q_s8(o, vqtbl1q_s8(levels, vandq_u8(lo, m4)));
        vst1q_s8(o + 16, vqtbl1q_s8(levels, vshrq_n_u8(lo, 4)));
        vst1q_s8(o + 32, vqtbl1q_s8(levels, vandq_u8(hi, m4)));
        vst1q_s8(o + 48, vqtbl1q_s8(levels, vshrq_n_u8(hi, 4)));
    }
}

/* neon_mm_nib on rows already unpacked into wi (neon_mm_unpack); w0, w1
   give the scales. The same steps in the same order: the same results. */
static NEON_MM void neon_mm_nib_pre(const struct janas_block_iq4nl *w0,
                                    const struct janas_block_iq4nl *w1,
                                    const int8_t *wi,
                                    const struct janas_block_q8k *a,
                                    const struct janas_block_q8k *b,
                                    const int8_t *xi, size_t nb, float out[4])
{
    float32x4_t acc[4][2];
    for (int p = 0; p < 4; p++)
        acc[p][0] = acc[p][1] = vdupq_n_f32(0.0f);
    for (size_t s = 0; s < nb; s++) {
        const struct janas_block_iq4nl *r0 = w0 + 8 * s, *r1 = w1 + 8 * s;
        int32x4_t dots[8];
        for (int k = 0; k < 8; k++) {
            const int8_t *u = wi + (s * 8 + (size_t)k) * 64;
            dots[k] =
                neon_mm_block(vld1q_s8(u), vld1q_s8(u + 16), vld1q_s8(u + 32),
                              vld1q_s8(u + 48), xi + (s * 8 + (size_t)k) * 64);
        }
        float32x4_t d0[2] = {neon_h2f4(&r0[0].d, sizeof(*r0)),
                             neon_h2f4(&r0[4].d, sizeof(*r0))};
        float32x4_t d1[2] = {neon_h2f4(&r1[0].d, sizeof(*r1)),
                             neon_h2f4(&r1[4].d, sizeof(*r1))};
        float32x4_t xa[2] = {vld1q_f32(a[s].d32), vld1q_f32(a[s].d32 + 4)};
        float32x4_t xb[2] = {vld1q_f32(b[s].d32), vld1q_f32(b[s].d32 + 4)};
        neon_mm_acc(dots, d0, d1, xa, xb, acc);
    }
    for (int p = 0; p < 4; p++)
        out[p] = neon_sum8(acc[p][0], acc[p][1]);
}
