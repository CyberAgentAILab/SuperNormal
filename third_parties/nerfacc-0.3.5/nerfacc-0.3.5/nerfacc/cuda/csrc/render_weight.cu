/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#include "include/helpers_cuda.h"

__global__ void weight_from_sigma_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *starts,
    const float *ends,
    const float *sigmas,
    // outputs
    float *weights)
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
    weights += base;

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float delta = ends[j] - starts[j];
        const float alpha = 1.f - __expf(-sigmas[j] * delta);
        weights[j] = alpha * T;
        T *= (1.f - alpha);
    }
    return;
}

__global__ void weight_from_sigma_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info, 
    const float *starts, 
    const float *ends,   
    const float *sigmas, 
    const float *weights, 
    const float *grad_weights, 
    // outputs
    float *grad_sigmas)
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
    weights += base;
    grad_weights += base;
    grad_sigmas += base;

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float delta = ends[j] - starts[j];
        const float alpha = 1.f - __expf(-sigmas[j] * delta);
        grad_sigmas[j] = (grad_weights[j] * T - accum) * delta;
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
    return;
}

// template <typename scalar_t>
__global__ void weight_from_alpha_patch_based_forward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    const int *packed_info, // (n_patches, 2)
    const float *alphas,  // (n_samples, patch_size, 1)
    // outputs
    float *weights// ()
    ){
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch

    // locate
    const int base = packed_info[i * 2 + 0];  // get the base of the patch
    const int steps = packed_info[i * 2 + 1]; // get the steps of the patch
    if (steps == 0)
        return;

    alphas += base * patch_size;  // move the pointer to the base
    weights += base * patch_size;  // move the pointer to the base
//     transmittance += base * patch_size;  // move the pointer to the base

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t ray_id = j * patch_size + k;
        const float alpha = alphas[ray_id];  // get the alpha value
//         transmittance[ray_id] = T;
        weights[ray_id] = alpha * T;  // calculate the weight
        T *= (1.f - alpha);  // update the T value
    }
    return;
}

__global__ void weight_and_transmittance_from_alpha_patch_based_forward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    const int *packed_info, // (n_patches, 2)
    const float *alphas,  // (n_samples, patch_size, 1)
    // outputs
    float *weights,
    float *transmittance// ()
    ){
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch

    // locate
    const int base = packed_info[i * 2 + 0];  // get the base of the patch
    const int steps = packed_info[i * 2 + 1]; // get the steps of the patch
    if (steps == 0)
        return;

    alphas += base * patch_size;  // move the pointer to the base
    weights += base * patch_size;  // move the pointer to the base
    transmittance += base * patch_size;  // move the pointer to the base

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t ray_id = j * patch_size + k;
        const float alpha = alphas[ray_id];  // get the alpha value
        transmittance[ray_id] = T;
        weights[ray_id] = alpha * T;  // calculate the weight
        T *= (1.f - alpha);  // update the T value
    }
    return;
}

__global__ void weight_from_alpha_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *alphas,   
    // outputs
    float *weights)
{
    CUDA_GET_THREAD_ID(i, n_rays);  // i is the thread id

    // locate
    const int base = packed_info[i * 2 + 0];  // get the base
    const int steps = packed_info[i * 2 + 1]; // get the steps
    if (steps == 0)
        return;

    alphas += base;  // move the pointer to the base
    weights += base;  // move the pointer to the base

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];  // get the alpha value
        weights[j] = alpha * T;  // calculate the weight
        T *= (1.f - alpha);  // update the T value
    }
    return;
}

__global__ void weight_from_alpha_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info,  
    const float *alphas,     
    const float *weights,    
    const float *grad_weights,
    // outputs
    float *grad_alphas)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0]; 
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base;
    weights += base;
    grad_weights += base;
    grad_alphas += base;

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];
        grad_alphas[j] = (grad_weights[j] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
    return;
}


__global__ void weight_from_alpha_importance_sampling_forward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *alphas,
    const float *importance,
    // outputs
    float *weights)
{
    CUDA_GET_THREAD_ID(i, n_rays);  // i is the thread id

    // locate
    const int base = packed_info[i * 2 + 0];  // get the base
    const int steps = packed_info[i * 2 + 1]; // get the steps
    if (steps == 0)
        return;

    alphas += base;  // move the pointer to the base
    weights += base;  // move the pointer to the base
    importance += base;  // move the pointer to the base

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];  // get the alpha value
        weights[j] = alpha * T / importance[j];  // calculate the weight
        T *= (1.f - alpha);  // update the T value
    }
    return;
}

__global__ void weight_from_alpha_importance_sampling_backward_kernel(
    const uint32_t n_rays,
    const int *packed_info,
    const float *alphas,
    const float *weights,
    const float *grad_weights,
    const float *importance,
    // outputs
    float *grad_alphas)
{
    CUDA_GET_THREAD_ID(i, n_rays);

    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base;
    weights += base;
    grad_weights += base;
    grad_alphas += base;
    importance += base;

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        accum += grad_weights[j] * weights[j];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const float alpha = alphas[j];
        grad_alphas[j] = (grad_weights[j] * T - importance[j] * accum) / (importance[j] * fmaxf(1.f - alpha, 1e-10f));
        accum -= grad_weights[j] * weights[j];
        T *= (1.f - alpha);
    }
    return;
}


__global__ void weight_from_alpha_patch_based_backward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    const int *packed_info,
    const float *alphas,
    const float *weights,
    const float *grad_weights,
    // outputs
    float *grad_alphas)
{
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch


    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base * patch_size;  // move the pointer to the base
    weights += base * patch_size;  // move the pointer to the base
    grad_weights += base * patch_size;  // move the pointer to the base
    grad_alphas += base * patch_size;  // move the pointer to the base

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t sample_idx = j * patch_size + k;
        accum += grad_weights[sample_idx] * weights[sample_idx];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t sample_idx = j * patch_size + k;
        const float alpha = alphas[sample_idx];
        grad_alphas[sample_idx] = (grad_weights[sample_idx] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
        accum -= grad_weights[sample_idx] * weights[sample_idx];
        T *= (1.f - alpha);
    }
    return;
}

__global__ void weight_and_transmittance_from_alpha_patch_based_backward_kernel(
    const uint32_t n_patches,
    const uint32_t patch_size,
    const int *packed_info,
    const float *alphas,
    const float *weights,
    const float *grad_weights,
    // outputs
    float *grad_alphas)
{
    CUDA_GET_THREAD_ID_2D(i, k, n_patches, patch_size);  // i is the patch id, k is the ray id within the patch


    // locate
    const int base = packed_info[i * 2 + 0];
    const int steps = packed_info[i * 2 + 1];
    if (steps == 0)
        return;

    alphas += base * patch_size;  // move the pointer to the base
    weights += base * patch_size;  // move the pointer to the base
    grad_weights += base * patch_size;  // move the pointer to the base
    grad_alphas += base * patch_size;  // move the pointer to the base

    float accum = 0;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t sample_idx = j * patch_size + k;
        accum += grad_weights[sample_idx] * weights[sample_idx];
    }

    // accumulation
    float T = 1.f;
    for (int j = 0; j < steps; ++j)
    {
        const uint32_t sample_idx = j * patch_size + k;
        const float alpha = alphas[sample_idx];
        grad_alphas[sample_idx] = (grad_weights[sample_idx] * T - accum) / fmaxf(1.f - alpha, 1e-10f);
        accum -= grad_weights[sample_idx] * weights[sample_idx];
        T *= (1.f - alpha);
    }
    return;
}

torch::Tensor weight_from_sigma_forward_naive(
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
    torch::Tensor weights = torch::empty_like(sigmas);

    weight_from_sigma_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>());
    return weights;
}

torch::Tensor weight_from_sigma_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor starts,
    torch::Tensor ends,
    torch::Tensor sigmas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(starts);
    CHECK_INPUT(ends);
    CHECK_INPUT(sigmas);

    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(starts.ndimension() == 2 & starts.size(1) == 1);
    TORCH_CHECK(ends.ndimension() == 2 & ends.size(1) == 1);
    TORCH_CHECK(sigmas.ndimension() == 2 & sigmas.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 2 & grad_weights.size(1) == 1);

    const uint32_t n_samples = sigmas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_sigmas = torch::empty_like(sigmas);

    weight_from_sigma_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        starts.data_ptr<float>(),
        ends.data_ptr<float>(),
        sigmas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_sigmas.data_ptr<float>());

    return grad_sigmas;
}

torch::Tensor weight_from_alpha_forward_naive(
    torch::Tensor packed_info, torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor weights = torch::empty_like(alphas);

    weight_from_alpha_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>());
    return weights;
}

torch::Tensor weight_from_alpha_patch_based_forward_naive(
    torch::Tensor packed_info, // (n_patches, 2)
    torch::Tensor alphas // (n_samples, patches_size, 1)
    )
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
    // convert to uint
//     thread_for_a_patch = static_cast<uint32_t>(thread_for_a_patch);
//     thread_for_n_samples = static_cast<uint32_t>(thread_for_n_samples);

    const dim3 threads(thread_for_n_samples, thread_for_a_patch);
//     const dim3 blocks = CUDA_N_BLOCKS_NEEDED(n_samples, threads);
    const dim3 blocks((n_patches+threads.x-1)/threads.x, (patch_size+threads.y-1)/threads.y);

    // outputs
    torch::Tensor weights = torch::empty_like(alphas);
    torch::Tensor transmittance = torch::empty_like(alphas);

    weight_from_alpha_patch_based_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>());
    return weights;
}

torch::Tensor weight_from_alpha_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,
    torch::Tensor packed_info,
    torch::Tensor alphas)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
    TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 2 & grad_weights.size(1) == 1);

    const uint32_t n_samples = alphas.size(0);
    const uint32_t n_rays = packed_info.size(0);

    const int threads = 256;
    const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);

    // outputs
    torch::Tensor grad_alphas = torch::empty_like(alphas);

    weight_from_alpha_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_rays,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_alphas.data_ptr<float>());
    return grad_alphas;
}

torch::Tensor weight_from_alpha_patch_based_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,  // (n_samples, patches_size, 1)
    torch::Tensor packed_info,
    torch::Tensor alphas)  // (n_samples, patches_size, 1)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 3 & alphas.size(2) == 1);
    TORCH_CHECK(weights.ndimension() == 3 & weights.size(2) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 3 & grad_weights.size(2) == 1);

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
    torch::Tensor grad_alphas = torch::empty_like(alphas);

    weight_from_alpha_patch_based_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_alphas.data_ptr<float>());
    return grad_alphas;
}


std::vector<torch::Tensor> weight_and_transmittance_from_alpha_patch_based_forward_naive(
    torch::Tensor packed_info, // (n_patches, 2)
    torch::Tensor alphas // (n_samples, patches_size, 1)
    )
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
    torch::Tensor weights = torch::empty_like(alphas);
    torch::Tensor transmittance = torch::empty_like(alphas);

    weight_and_transmittance_from_alpha_patch_based_forward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        // outputs
        weights.data_ptr<float>(),
        transmittance.data_ptr<float>());
    return {weights, transmittance};
}

torch::Tensor weight_and_transmittance_from_alpha_patch_based_backward_naive(
    torch::Tensor weights,
    torch::Tensor grad_weights,  // (n_samples, patches_size, 1)
    torch::Tensor packed_info,
    torch::Tensor alphas)  // (n_samples, patches_size, 1)
{
    DEVICE_GUARD(packed_info);
    CHECK_INPUT(packed_info);
    CHECK_INPUT(alphas);
    CHECK_INPUT(weights);
    CHECK_INPUT(grad_weights);
    TORCH_CHECK(packed_info.ndimension() == 2);
    TORCH_CHECK(alphas.ndimension() == 3 & alphas.size(2) == 1);
    TORCH_CHECK(weights.ndimension() == 3 & weights.size(2) == 1);
    TORCH_CHECK(grad_weights.ndimension() == 3 & grad_weights.size(2) == 1);

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
    torch::Tensor grad_alphas = torch::empty_like(alphas);

    weight_and_transmittance_from_alpha_patch_based_backward_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_patches,
        patch_size,
        // inputs
        packed_info.data_ptr<int>(),
        alphas.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_weights.data_ptr<float>(),
        // outputs
        grad_alphas.data_ptr<float>());
    return grad_alphas;
}

// torch::Tensor weight_from_alpha_importance_sampling_forward_naive(
//     torch::Tensor packed_info, torch::Tensor alphas, torch::Tensor importance_pdfs)
// {
//     DEVICE_GUARD(packed_info);
//     CHECK_INPUT(packed_info);
//     CHECK_INPUT(alphas);
//     CHECK_INPUT(importance_pdfs);
//     TORCH_CHECK(packed_info.ndimension() == 2);
//     TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
//     TORCH_CHECK(importance_pdfs.ndimension() == 2 & importance_pdfs.size(1) == 1);
//
//     const uint32_t n_samples = alphas.size(0);
//     const uint32_t n_rays = packed_info.size(0);
//
//     const int threads = 256;
//     const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);
//
//     // outputs
//     torch::Tensor weights = torch::empty_like(alphas);
//
//     weight_from_alpha_forward_kernel<<<
//         blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
//         n_rays,
//         // inputs
//         packed_info.data_ptr<int>(),
//         alphas.data_ptr<float>(),
//         importance_pdfs.data_ptr<float>(),
//         // outputs
//         weights.data_ptr<float>());
//     return weights;
// }
//
// torch::Tensor weight_from_alpha_importance_sampling_backward_naive(
//     torch::Tensor weights,
//     torch::Tensor grad_weights,
//     torch::Tensor packed_info,
//     torch::Tensor alphas,
//     torch::Tensor importance_pdfs)
// {
//     DEVICE_GUARD(packed_info);
//     CHECK_INPUT(packed_info);
//     CHECK_INPUT(alphas);
//     CHECK_INPUT(weights);
//     CHECK_INPUT(grad_weights);
//     CHECK_INPUT(importance_pdfs);
//     TORCH_CHECK(packed_info.ndimension() == 2);
//     TORCH_CHECK(alphas.ndimension() == 2 & alphas.size(1) == 1);
//     TORCH_CHECK(weights.ndimension() == 2 & weights.size(1) == 1);
//     TORCH_CHECK(importance_pdfs.ndimension() == 2 & importance_pdfs.size(1) == 1);
//     TORCH_CHECK(grad_weights.ndimension() == 2 & grad_weights.size(1) == 1);
//
//
//     const uint32_t n_samples = alphas.size(0);
//     const uint32_t n_rays = packed_info.size(0);
//
//     const int threads = 256;
//     const int blocks = CUDA_N_BLOCKS_NEEDED(n_rays, threads);
//
//     // outputs
//     torch::Tensor grad_alphas = torch::empty_like(alphas);
//
//     weight_from_alpha_backward_kernel<<<
//         blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
//         n_rays,
//         // inputs
//         packed_info.data_ptr<int>(),
//         alphas.data_ptr<float>(),
//         weights.data_ptr<float>(),
//         grad_weights.data_ptr<float>(),
//         importance_pdfs.data_ptr<float>(),
//         // outputs
//         grad_alphas.data_ptr<float>());
//     return grad_alphas;
// }