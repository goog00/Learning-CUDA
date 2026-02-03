#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <type_traits>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <iostream>


/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // 检查输入大小是否足够
  if(h_input.size() < rows * cols){
    throw std::invalid_argument("trace: h_input.size() < rows * cols");
  }

  T sum = T(0);
  size_t n = std::min(rows, cols);
  for(size_t i = 0; i < n; ++i) {
    //行优先索引：第i行第i列的元素下标为 i * cols + i
    sum += h_input[i *cols + i];
  }
  return sum;
}

// ============================================================================
// Flash Attention 实现
// 针对 NVIDIA A100 GPU 优化
// 支持因果掩码(Causal Masking)和分组查询注意力(GQA)
// ============================================================================

// 线程块大小常量
constexpr int FA_BLOCK_SIZE = 256;  // 每个 block 的线程数

/**
 * @brief 类型转换辅助函数：将输入类型转换为 float
 * @param x 输入值
 * @return float 类型的值
 */
__device__ __forceinline__ float toFloat(float x) { return x; }
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

/**
 * @brief 类型转换辅助函数：将 float 转换为目标类型
 * @param x 输入的 float 值
 * @return 目标类型的值
 */
template <typename T>
__device__ __forceinline__ T fromFloat(float x);

template <>
__device__ __forceinline__ float fromFloat<float>(float x) { return x; }

template <>
__device__ __forceinline__ half fromFloat<half>(float x) { return __float2half(x); }

/**
 * @brief Flash Attention 核函数
 * 
 * 每个 block 处理一个 (batch, head, query_pos) 位置的注意力计算
 * 
 * 算法流程：
 * 1. 加载当前 Query 向量到共享内存
 * 2. 计算与所有 Key 的点积，得到注意力分数
 * 3. 应用因果掩码（如果需要）
 * 4. 使用稳定的 softmax 算法计算注意力权重
 * 5. 计算 attention * V 得到输出
 * 
 数值精度优化：
  * - 使用 double 类型进行所有中间计算
 * - 使用 Kahan 求和算法进行累加，减少浮点误差
 * @tparam T 数据类型（float 或 half）
 * @param q Query 张量，设备端指针
 * @param k Key 张量，设备端指针
 * @param v Value 张量，设备端指针
 * @param o Output 张量，设备端指针
 * @param batch_size 批次大小
 * @param target_seq_len 目标序列长度
 * @param src_seq_len 源序列长度
 * @param query_heads Query 头数
 * @param kv_heads Key/Value 头数（用于 GQA）
 * @param head_dim 每个头的维度
 * @param heads_per_group 每组中 Query 头的数量（用于 GQA）
 * @param scale 缩放因子（1/sqrt(head_dim)）
 * @param is_causal 是否应用因果掩码
 */
template <typename T>
__global__ void flashAttentionSimpleKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    int heads_per_group,
    double scale,
    bool is_causal) {
    
    // 每个 block 处理一个 (batch, head, query_pos) 位置的组合
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int query_pos = blockIdx.z;
    int tid = threadIdx.x;
    
    // 边界检查
    if (batch_idx >= batch_size || head_idx >= query_heads || query_pos >= target_seq_len) {
        return;
    }
    
    // GQA 支持：计算对应的 KV 头索引
    // heads_per_group = query_heads / kv_heads，已在主机端预先计算
    int kv_head_idx = head_idx / heads_per_group;
    
    // 计算输入张量的基地址偏移
    // Q: [batch_size, target_seq_len, query_heads, head_dim]
    int q_base = batch_idx * target_seq_len * query_heads * head_dim
               + query_pos * query_heads * head_dim
               + head_idx * head_dim;
    
    // K/V: [batch_size, src_seq_len, kv_heads, head_dim]
    int kv_base = batch_idx * src_seq_len * kv_heads * head_dim
                + kv_head_idx * head_dim;
    
    // O: [batch_size, target_seq_len, query_heads, head_dim]
    int o_base = batch_idx * target_seq_len * query_heads * head_dim
               + query_pos * query_heads * head_dim
               + head_idx * head_dim;
    
    // 分配共享内存
    // s_q: [head_dim] - 当前 Query 向量（使用 double 提高精度）
    // s_scores: [src_seq_len] - 注意力分数（使用 double 提高精度）
    extern __shared__ char shared_mem[];
    double* s_q = (double*)shared_mem;
    double* s_scores = s_q + head_dim;
    
    // 加载 Query 向量到共享内存
    for (int i = tid; i < head_dim; i += blockDim.x) {
          s_q[i] = static_cast<double>(toFloat(q[q_base + i]));
    }
    __syncthreads();
    
    // ========================================
    // 第一阶段：计算所有注意力分数 Q*K^T
    // ========================================
    double local_max = -DBL_MAX;
    
    for (int kv_pos = tid; kv_pos < src_seq_len; kv_pos += blockDim.x) {
        // 因果掩码检查：只关注当前位置及之前的位置
        if (is_causal && kv_pos > query_pos) {
            s_scores[kv_pos] = -DBL_MAX;
        } else {
            // 计算 Q * K^T 的点积
            double score = 0.0;
            double comp = 0.0;  // Kahan 补偿项
            int k_offset = kv_base + kv_pos * kv_heads * head_dim;
            
            for (int d = 0; d < head_dim; d++) {
                double prod = s_q[d] * static_cast<double>(toFloat(k[k_offset + d]));
                volatile double y = prod - comp;
                volatile double t = score + y;
                volatile double c = (t - score) - y;
                comp = c;
                score = t;
            }
            
            // 应用缩放因子 scale = 1/sqrt(head_dim)
            s_scores[kv_pos] = score * scale;
        }
        
        // 更新局部最大值（用于数值稳定的 softmax）
        if (s_scores[kv_pos] > local_max) {
            local_max = s_scores[kv_pos];
        }
    }
    __syncthreads();
    
    // ========================================
    // 规约找全局最大值
    // ========================================
    __shared__ double s_max[FA_BLOCK_SIZE];
    if (tid < FA_BLOCK_SIZE) {
        s_max[tid] = local_max;
    }
    __syncthreads();
    
    // 并行规约操作
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            if (s_max[tid + stride] > s_max[tid]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    
    double global_max = s_max[0];
    __syncthreads();
    
    // ========================================
    // 第二阶段：计算 softmax 分母（指数和）
    // ========================================
    double local_sum = 0.0;
    double comp_sum = 0.0;  // Kahan 补偿项

    for (int kv_pos = tid; kv_pos < src_seq_len; kv_pos += blockDim.x) {
        // 只处理有效的分数（非掩码位置）
        if (s_scores[kv_pos] > -DBL_MAX / 2) {
            s_scores[kv_pos] = exp(s_scores[kv_pos] - global_max);
            // Kahan 求和
            volatile double y = s_scores[kv_pos] - comp_sum;
            volatile double t = local_sum + y;
            volatile double c = (t - local_sum) - y;
            comp_sum = c;
            local_sum = t;
        } else {
            s_scores[kv_pos] = 0.0;
        }
    }
    __syncthreads();
    
    // 规约求和
    __shared__ double s_sum[FA_BLOCK_SIZE];
    if (tid < FA_BLOCK_SIZE) {
        s_sum[tid] = local_sum;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < blockDim.x) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    double global_sum = s_sum[0];
    __syncthreads();
    
    // 归一化注意力权重
    if (global_sum > 0) {
        for (int kv_pos = tid; kv_pos < src_seq_len; kv_pos += blockDim.x) {
            s_scores[kv_pos] /= global_sum;
        }
    }
    __syncthreads();
    
    // ========================================
    // 第三阶段：计算 Attention * V
    //使用 Kahan 求和算法累加提高精度
    // ========================================
    // 每个线程计算输出的一部分维度
    for (int d = tid; d < head_dim; d += blockDim.x) {
        double acc = 0.0;
        double comp = 0.0;  // Kahan 补偿项
        
        // 累加所有 KV 位置的加权 Value
        for (int kv_pos = 0; kv_pos < src_seq_len; kv_pos++) {
            int v_offset = kv_base + kv_pos * kv_heads * head_dim + d;
            double prod = s_scores[kv_pos] * static_cast<double>(toFloat(v[v_offset]));
            volatile double y = prod - comp;
            volatile double t = acc + y;
            volatile double c = (t - acc) - y;
            comp = c;
            acc = t;
        }
        
        // 写入输出
       o[o_base + d] = fromFloat<T>(static_cast<float>(acc));
    }
}





/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // ========================================
    // 参数验证
    // ========================================
    // 检查所有维度参数是否为正数
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return;  // 无效的维度参数
    }
    
    // 检查 GQA 配置是否有效（query_heads 必须是 kv_heads 的整数倍）
    if (query_heads % kv_heads != 0) {
        return;  // 无效的 GQA 配置
    }
    
    // 计算张量大小（使用 size_t 避免整数溢出）
    size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
    size_t kv_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
    size_t o_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
    
    // 验证输入大小
    if (h_q.size() != q_size || h_k.size() != kv_size || h_v.size() != kv_size) {
        return;  // 输入大小不匹配
    }
    
    // 调整输出大小
    h_o.resize(o_size);
    
    // ========================================
    // 分配设备内存
    // ========================================
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    
    // 使用 do-while 结构确保错误时能正确清理资源
    cudaError_t err;
    
    err = cudaMalloc(&d_q, q_size * sizeof(T));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_k, kv_size * sizeof(T));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_v, kv_size * sizeof(T));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_o, o_size * sizeof(T));
    if (err != cudaSuccess) goto cleanup;
    
    // ========================================
    // 拷贝数据到设备
    // ========================================
    err = cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    {
        // ========================================
        // 计算核函数参数
        // ========================================
        // scale = 1 / sqrt(head_dim)
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        
        // 计算每组中 Query 头的数量（用于 GQA）
        int heads_per_group = query_heads / kv_heads;
        // ========================================
        // 配置核函数启动参数
        // ========================================
        // Grid: (batch_size, query_heads, target_seq_len)
        // 检查 grid 维度是否超过 CUDA 限制
        // CUDA 限制: x,y <= 65535, z <= 2^31-1 (compute capability 3.0+)
        if (batch_size > 65535 || query_heads > 65535) {
            goto cleanup;  // 超过 CUDA grid 维度限制
        }
        
        dim3 grid(batch_size, query_heads, target_seq_len);
        int block_size = FA_BLOCK_SIZE;
        
        // 计算共享内存大小 使用 double 类型提高精度）
        // s_q: head_dim * sizeof(double)
        // s_scores: src_seq_len * sizeof(double)
        size_t shared_mem_size = (head_dim + src_seq_len) * sizeof(double);
        
        // 检查共享内存是否超过设备限制（A100 每个 SM 最大 164KB）
        // 这里使用保守的 48KB 限制，适用于大多数 GPU
        if (shared_mem_size > 48 * 1024) {
            // 共享内存过大，可能导致核函数启动失败
            // 但我们仍然尝试启动，让 CUDA 运行时处理
        }
        
        // ========================================
        // 启动核函数
        // ========================================
        flashAttentionSimpleKernel<T><<<grid, block_size, shared_mem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            heads_per_group, scale, is_causal
        );
        
        // 检查核函数执行错误
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
        
        // ========================================
        // 拷贝结果回主机
        // ========================================
        err = cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
cleanup:
    // ========================================
    // 释放设备内存
    // ========================================
    if (d_q) cudaFree(d_q);
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_o) cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
