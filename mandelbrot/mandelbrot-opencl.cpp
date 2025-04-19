#include <CL/cl.h>
#include <SDL.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static const unsigned Xres = 1706, Yres = 960;
static const unsigned MAXITER = 8000;
static const double ESC_RADIUS_SQ = 4.0;

const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define XRES 1706
#define YRES 960
#define MAXITER 8000
#define ESC_RADIUS_SQ 4.0
__constant uchar R[12] = {0x00,0x40,0x7E,0x11,0x16,0x38,0xFC,0xD0,0x5F,0xDC,0xFF,0x6B};
__constant uchar G[12] = {0x00,0x40,0x9F,0x90,0x68,0xCF,0xFF,0x99,0x00,0x37,0x8E,0x14};
__constant uchar B[12] = {0x00,0xE0,0xFF,0x9F,0x18,0x3F,0x00,0x24,0x09,0x0A,0xFE,0xBC};

double mylog2c(double v) { return log2(v); }

double iterate(double zr, double zi) {
    double cr = zr, ci = zi, dist;
    int iter = MAXITER;
    while (iter > 0) {
        double r2 = cr*cr, i2 = ci*ci;
        dist = r2 + i2;
        if (dist >= ESC_RADIUS_SQ) break;
        double ri = cr*ci;
        ci = zi + 2.0*ri;
        cr = zr + r2 - i2;
        --iter;
    }
    if (iter == 0) return mylog2c(MAXITER - iter + 1 - log2(log2(dist)/2.0));
    return 0.0;
}

uint colorFun(int iter, double v) {
    int i = iter % 12, j = (iter + 1) % 12;
    double f = v - floor(v);
    uint cr = (uint)((R[i] + (R[j]-R[i])*f)) & 0xFF;
    uint cg = (uint)((G[i] + (G[j]-G[i])*f)) & 0xFF;
    uint cb = (uint)((B[i] + (B[j]-B[i])*f)) & 0xFF;
    return (cr<<16)|(cg<<8)|cb;
}

__kernel void mandelbrot(__global uint* pixels, double time) {
    int x = get_global_id(0), y = get_global_id(1);
    double scale = 4.0 * pow(2.0, -min(time,53.0)*0.7);
    double xs = scale / YRES, ys = scale / YRES;
    double zr = -0.743639266077433 + xs*(x - XRES/2);
    double zi =  0.131824786875559 + ys*(y - YRES/2);
    double v = iterate(zr, zi);
    int it = (v == 0.0) ? MAXITER : (int)v;
    pixels[y*XRES + x] = colorFun(it, v);
}
)CLC";

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* win = SDL_CreateWindow("zoom", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Xres, Yres, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, 0);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, Xres, Yres);

    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, nullptr);
    cl_device_id dev;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, nullptr);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, nullptr);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    cl_kernel kern = clCreateKernel(prog, "mandelbrot", nullptr);

    size_t global[2] = { Xres, Yres };
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(uint)*Xres*Yres, nullptr, nullptr);
    clSetKernelArg(kern, 0, sizeof(cl_mem), &buf);

    std::vector<uint> pixels(Xres*Yres);
    SDL_Event ev;
    auto t0 = std::chrono::system_clock::now();

    while (true) {
        double t = std::chrono::duration<double>(std::chrono::system_clock::now() - t0).count();
        clSetKernelArg(kern, 1, sizeof(double), &t);
        clEnqueueNDRangeKernel(q, kern, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, buf, CL_TRUE, 0, sizeof(uint)*pixels.size(), pixels.data(), 0, nullptr, nullptr);

        SDL_UpdateTexture(tex, nullptr, pixels.data(), Xres * 4);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);

        if (SDL_PollEvent(&ev) && ev.type == SDL_KEYDOWN && ev.key.keysym.sym == SDLK_ESCAPE) break;
    }

    clReleaseMemObject(buf);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

