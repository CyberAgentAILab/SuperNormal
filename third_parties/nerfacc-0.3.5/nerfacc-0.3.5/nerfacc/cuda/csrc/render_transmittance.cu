/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void transmittance_from_sigma_forward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info,
    const float *starts,
    const float *ends,
    const float *sigmas,
    // outputs
    float *transmittance)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    starts += base;
    ends += base;
    sigmas += base;
    transmittance += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = 0; j < steps; ++j)
    {
        transmittance[j] = __expf(-cumsum);
        cumsum += sigmas[j] * (ends[j] - starts[j]);
    }

    // // another way to impl:
    // float T = 1.f;
    // for (int j = 0; j < steps; ++j)
    // {
    //     const float delta = ends[j] - starts[j];
    //     const float alpha = 1.f - __expf(-sigmas[j] * delta);
    //     transmittance[j] = T;
    //     T *= (1.f - alpha);
    // }
    return;
}

__global__ void transmittance_from_sigma_backward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info,
    const float *starts,
    const float *ends,
    const float *transmittance,
    const float *transmittance_grad,
    // outputs
    float *sigmas_grad)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    transmittance += base;
    transmittance_grad += base;
    starts += base;
    ends += base;
    sigmas_grad += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = steps - 1; j >= 0; --j)
    {
        sigmas_grad[j] = cumsum * (ends[j] - starts[j]);
        cumsum += -transmittance_grad[j] * transmittance[j];
    }
    return;
}

__global__ void transmittance_from_alpha_forward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info,
    const float *alphas,
    // outputs
    float *transmittance)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base;
    transmittance += base;

    // accumulation
    float T = 1.0f;
    for (int j = 0; j < steps; ++j)
    {
        transmittance[j] = T;
        T *= (1.0f - alphas[j]);
    }
    return;
}

__global__ void transmittance_from_alpha_backward_kernel(
    const uint32_t n_rays,
    // inputs
    const int *packed_info,
    const float *alphas,
    const float *transmittance,
    const float *transmittance_grad,
    // outputs
    float *alphas_grad)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base;
    transmittance += base;
    transmittance_grad += base;
    alphas_grad += base;

    // accumulation
    float cumsum = 0.0f;
    for (int j = steps - 1; j >= 0; --j)
    {
        alphas_grad[j] = cumsum / fmax(1.0f - alphas[j], 1e-10f);
        cumsum += -transmittance_grad[j] * transmittance[j];
    }
    return;
}

__global__ void transmittance_from_alpha_patch_based_forward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    // inputs
    const int *packed_info,
    const float *alphas,
    // outputs
    float *transmittance)
{
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch

    // locate
    const int base = packed_info[i * 2 + 0];  // get the base of the patch
    const int steps = packed_info[i * 2 + 1]; // get the steps of the patch
    if (steps == 0)
        return;

    alphas += base * patch_size;  // move the pointer to the base
    transmittance += base * patch_size;  // move the pointer to the base

    // accumulation
    float T = 1.0f;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t ray_id = j * patch_size + k;
        transmittance[ray_id] = T;
        T *= (1.0f - alphas[j]);
    }
    return;
}

__global__ void transmittance_from_alpha_patch_based_backward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    // inputs
    const int *packed_info,
    const float *alphas,
    const float *transmittance,
    const float *transmittance_grad,
    // outputs
    float *alphas_grad)
{
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base * patch_size;
    transmittance += base * patch_size;
    transmittance_grad += base * patch_size;
    alphas_grad += base * patch_size;

    // accumulation
    float cumsum = 0.0f;
    for (int j = steps - 1; j >= 0; --j)
    {
        const uint32_t sample_idx = j * patch_size + k;
        alphas_grad[sample_idx] = cumsum / fmax(1.0f - alphas[sample_idx], 1e-10f);
        cumsum += -transmittance_grad[sample_idx] * transmittance[sample_idx];
    }
    return;
}

torch::Tensor transmittance_from_sigma_forward_naive(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);

    const uint32_t n_samples = sigmas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor transmittance = torch::empty_like(sigmas);

    // parallel across rays
    transmittance_from_sigma_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        // outputs
        transmittance.data_ptr<float>());
    return transmittance;
}

torch::Tensor transmittance_from_sigma_backward_naive(
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor sigmas_grad = torch::empty_like(transmittance);

    // parallel across rays
    transmittance_from_sigma_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        transmittance_grad.data_ptr<float>(),
        // outputs
        sigmas_grad.data_ptr<float>());
    return sigmas_grad;
}

torch::Tensor transmittance_from_alpha_forward_naive(
    torch::Tensor packed_info, torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(packed_info.ndimension() == 2);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor transmittance = torch::empty_like(alphas);

    // parallel across rays
    transmittance_from_alpha_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        transmittance.data_ptr<float>());
    return transmittance;
}

torch::Tensor transmittance_from_alpha_backward_naive(
    torch::Tensor packed_info,
    torch::Tensor alphas,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(transmittance.ndimension() == 2 & transmittance.size(1) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 2 & transmittance_grad.size(1) == 1);

    const uint32_t n_samples = transmittance.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor alphas_grad = torch::empty_like(alphas);

    // parallel across rays
    transmittance_from_alpha_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        transmittance_grad.data_ptr<float>(),
        // outputs
        alphas_grad.data_ptr<float>());
    return alphas_grad;
}

torch::Tensor transmittance_from_alpha_patch_based_forward_naive(
    torch::Tensor packed_info, torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 3 & alphas.size(2) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_patches = packed_info.size(0);
    const uint32_t patch_size  = alphas.size(1);

    // compute the required number of thread.y from patch size
    // take the log2 of patch size and round up to the next power of 2
    const uint32_t thread_for_a_patch = pow(2, ceil(log2(patch_size)));
    const uint32_t thread_for_n_samples = 256 / thread_for_a_patch;

    const dim3 threads(thread_for_n_samples, thread_for_a_patch);
    const dim3 blocks((n_patches+threads.x-1)/threads.x, (patch_size+threads.y-1)/threads.y);

    // outputs
    torch::Tensor transmittance = torch::empty_like(alphas);

    // parallel across rays
    transmittance_from_alpha_patch_based_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        transmittance.data_ptr<float>());
    return transmittance;
}

torch::Tensor transmittance_from_alpha_patch_based_backward_naive(
    torch::Tensor packed_info,
    torch::Tensor alphas,
    torch::Tensor transmittance,
    torch::Tensor transmittance_grad)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(transmittance);
    CHECK_INPUT(transmittance_grad);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(transmittance.ndimension() == 3 & transmittance.size(2) == 1);
    TORCH_CHECK(transmittance_grad.ndimension() == 3 & transmittance_grad.size(2) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_patches = packed_info.size(0);
    const uint32_t patch_size = alphas.size(1);

    // compute the required number of thread.y from patch size
    // take the log2 of patch size and round up to the next power of 2
    const uint32_t thread_for_a_patch = pow(2, ceil(log2(patch_size)));
    const uint32_t thread_for_n_samples = 256 / thread_for_a_patch;

    const dim3 threads(thread_for_n_samples, thread_for_a_patch);
    const dim3 blocks((n_patches+threads.x-1)/threads.x, (patch_size+threads.y-1)/threads.y);


    // outputs
    torch::Tensor alphas_grad = torch::empty_like(alphas);

    // parallel across rays
    transmittance_from_alpha_patch_based_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        transmittance.data_ptr<float>(),
        transmittance_grad.data_ptr<float>(),
        // outputs
        alphas_grad.data_ptr<float>());
    return alphas_grad;
}