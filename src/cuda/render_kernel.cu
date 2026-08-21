#include <cuda_runtime.h>
#include <iostream>

// Kernel to copy and convert float normalized (0..1) to uchar3 (BGR)
// grid is 2048x1024
__global__ void render_bg_kernel(float* d_frame_f, float* d_map_f, uchar3* d_out, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    
    int out_idx = y * cols + x;
    float val;
    if (x < 1024) {
        val = d_frame_f[y * 1024 + x];
    } else {
        val = d_map_f[y * 1024 + (x - 1024)];
    }
    
    unsigned char c = (unsigned char)(fminf(fmaxf(val * 255.0f, 0.0f), 255.0f));
    d_out[out_idx] = make_uchar3(c, c, c); // grayscale as BGR
}

__device__ void draw_circle(uchar3* d_out, int cols, int rows, int cx, int cy, int radius, uchar3 color) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < cols && py >= 0 && py < rows) {
                    d_out[py * cols + px] = color;
                }
            }
        }
    }
}

__device__ void draw_line(uchar3* d_out, int cols, int rows, int x0, int y0, int x1, int y1, uchar3 color) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;
    
    while (true) {
        if (x0 >= 0 && x0 < cols && y0 >= 0 && y0 < rows) {
            d_out[y0 * cols + x0] = color;
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = err;
        if (e2 > -dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
}

__global__ void render_matches_kernel(uchar3* d_out, int cols, int rows, float2* pts1, float2* pts2, int num_matches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_matches) return;
    
    int cx1 = (int)pts1[idx].x;
    int cy1 = (int)pts1[idx].y;
    int cx2 = (int)pts2[idx].x + 1024;
    int cy2 = (int)pts2[idx].y;
    
    uchar3 color = make_uchar3(0, 255, 0); // Green
    
    draw_line(d_out, cols, rows, cx1, cy1, cx2, cy2, color);
    draw_circle(d_out, cols, rows, cx1, cy1, 3, color);
    draw_circle(d_out, cols, rows, cx2, cy2, 3, color);
}

void cuda_render_visualization(float* d_frame_f, float* d_map_f, uchar3* d_out, float2* d_pts1, float2* d_pts2, int num_matches, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((2048 + block.x - 1) / block.x, (1024 + block.y - 1) / block.y);
    render_bg_kernel<<<grid, block, 0, stream>>>(d_frame_f, d_map_f, d_out, 2048, 1024);
    
    if (num_matches > 0) {
        int m_blocks = (num_matches + 255) / 256;
        render_matches_kernel<<<m_blocks, 256, 0, stream>>>(d_out, 2048, 1024, d_pts1, d_pts2, num_matches);
    }
}
