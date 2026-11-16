#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

extern "C" bool sparse_nn_has_avx512_bf16() { return true; }

static void normalize_prelu(std::vector<float> &hidden, const float *weight,
                            const float *bias, float prelu) {
    float mean = 0.0f;
    for (float value : hidden) mean += value;
    mean /= hidden.size();
    float variance = 0.0f;
    for (float value : hidden) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    const float inv_std = 1.0f / std::sqrt(variance / hidden.size() + 1.0e-5f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        float value = (hidden[i] - mean) * inv_std * weight[i] + bias[i];
        hidden[i] = value < 0.0f ? value * prelu : value;
    }
}

extern "C" void sparse_nn_forward_f32(
    const float *w1, const float *b1, const float *norm_weight,
    const float *norm_bias, float prelu, const float *w2, const float *b2,
    uint32_t hidden_count, const uint32_t *input_indices,
    const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    std::vector<float> hidden(b1, b1 + hidden_count);
    for (uint32_t row = 0; row < input_count; ++row) {
        const float *weights = w1 + size_t(input_indices[row]) * hidden_count;
        for (uint32_t column = 0; column < hidden_count; ++column)
            hidden[column] += input_values[row] * weights[column];
    }
    normalize_prelu(hidden, norm_weight, norm_bias, prelu);
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t canonical = output_indices[row];
        const float *weights = w2 + size_t(canonical) * hidden_count;
        float sum = b2[canonical];
        for (uint32_t column = 0; column < hidden_count; ++column)
            sum += weights[column] * hidden[column];
        output[row] = sum;
    }
}

extern "C" void sparse_nn_forward_bf16(
    const uint16_t *w1, const float *b1, const float *norm_weight,
    const float *norm_bias, float prelu, const uint16_t *w2, const float *b2,
    uint32_t hidden_count, const uint32_t *input_indices,
    const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    std::vector<float> hidden(b1, b1 + hidden_count);
    for (uint32_t row = 0; row < input_count; ++row) {
        const uint16_t *weights = w1 + size_t(input_indices[row]) * hidden_count;
        const float value = from_bf16(to_bf16(input_values[row]));
        for (uint32_t column = 0; column < hidden_count; ++column)
            hidden[column] += value * from_bf16(weights[column]);
    }
    normalize_prelu(hidden, norm_weight, norm_bias, prelu);
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t canonical = output_indices[row];
        const uint16_t *weights = w2 + size_t(canonical) * hidden_count;
        float sum = b2[canonical];
        for (uint32_t column = 0; column < hidden_count; ++column)
            sum += from_bf16(weights[column]) * from_bf16(to_bf16(hidden[column]));
        output[row] = sum;
    }
}

extern "C" void sparse_nn_forward_bf16_dense_input(
    const uint16_t *w1, const float *b1, const float *norm_weight,
    const float *norm_bias, float prelu, const uint16_t *w2, const float *b2,
    uint32_t hidden_count, const float *input_values, uint32_t input_count,
    const uint32_t *output_indices, uint32_t output_count, float *output) {
    std::vector<float> hidden(b1, b1 + hidden_count);
    for (uint32_t row = 0; row < input_count; ++row) {
        const uint16_t *weights = w1 + size_t(row) * hidden_count;
        const float value = from_bf16(to_bf16(input_values[row]));
        for (uint32_t column = 0; column < hidden_count; ++column)
            hidden[column] += value * from_bf16(weights[column]);
    }
    normalize_prelu(hidden, norm_weight, norm_bias, prelu);
    for (uint32_t row = 0; row < output_count; ++row) {
        const uint32_t compact = output_indices[row];
        const uint16_t *weights = w2 + size_t(compact) * hidden_count;
        float sum = b2[compact];
        for (uint32_t column = 0; column < hidden_count; ++column)
            sum += from_bf16(weights[column]) * from_bf16(to_bf16(hidden[column]));
        output[row] = sum;
    }
}
