#define _Alignof alignof
#include <iostream>
//static const unsigned Xres = 3840, Yres = 2160;
//static const unsigned Xres = 1920, Yres = 1080;
static const unsigned Xres = 1706, Yres = 960;
//static const unsigned Xres = 848, Yres = 480;
//static const unsigned Xres = 424, Yres = 240;
//static const unsigned Xres = 212, Yres = 120;

static unsigned Color(unsigned x,unsigned y, double iter)
{
    static const unsigned char r[]{0x00,0x40,0x7E,0x11,0x16,0x38,0xFC,0xD0,0x5F,0xDC,0xFF,0x6B};
    static const unsigned char g[]{0x00,0x40,0x9F,0x90,0x68,0xCF,0xFF,0x99,0x00,0x37,0x8E,0x14};
    static const unsigned char b[]{0x00,0xE0,0xFF,0x9F,0x18,0x3F,0x00,0x24,0x09,0x0A,0xFE,0xBC};
    constexpr int k = 1, m = 0x3F;
    //constexpr int k = 150, m = 0x10;
    double d = ((((x&4)/4u + (x&2)*2u + (x&1)*16u) + (((x^y)&4)/2u + ((x^y)&2)*4u + ((x^y)&1)*32u))&m)/64.;
    auto lerp = [d,k](int a,int b,double p) -> unsigned { return int(a/k + (b/k-a/k) * p + d)*255/(255/k); };
    return lerp(r[int(iter)%sizeof r], r[int(iter+1)%sizeof r], iter-int(iter))*0x10000u
         + lerp(g[int(iter)%sizeof r], g[int(iter+1)%sizeof r], iter-int(iter))*0x100u
         + lerp(b[int(iter)%sizeof r], b[int(iter+1)%sizeof r], iter-int(iter))*0x1u;
}



#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cstring>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <array>
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <SDL.h>

unsigned frame = 0;
SDL_Window *w = nullptr;
SDL_Renderer *r = nullptr;
SDL_Texture *t = nullptr;
 
SDL_Event event;
volatile bool Terminated = false;

double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin ).count();
}

void Put(const std::vector<unsigned>& pixels)
{
    SDL_UpdateTexture(t, nullptr, &pixels[0], 4*Xres);
    SDL_RenderCopy(r, t, nullptr, nullptr);
    SDL_RenderPresent(r);
    ++frame;
    std::fprintf(stderr, "Frame%6u, %.2f fps, time = %d s...\r", frame, frame / GetTime(), (int)GetTime());
    std::fflush(stderr);
}

#define CHECK_TERMINATE() do {SDL_PollEvent(&event); \
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym==SDLK_ESCAPE) Terminated=true; \
        } while(0)

#define MAINLOOP_START(n)

#define MAINLOOP_GET_CONDITION()     GetTime() < 53 && !Terminated

#define MAINLOOP_SET_COORDINATES()   do { \
            zr = -0.743639266077433; \
            zi = +0.131824786875559; \
            double scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7); \
            xscale = scale / Yres; \
            yscale = scale / Yres; } while(0)

#define MAINLOOP_PUT_RESULT(pixels) Put(pixels)
#define MAINLOOP_FINISH()           std::printf("\n%u frames rendered\n", frame)

double mylog2(double value)
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    const double  half         = 0.5;
    std::uint64_t half_bits    = reinterpret_cast<const std::uint64_t&>(half);
    int e,lt;
    uint64_t m;
    double x, dbl_e, z, y, u, t;
    m = reinterpret_cast<const std::uint64_t&>(value);
    e = m >> mantissa_bits;
    m &= std::uint64_t((1ull << mantissa_bits)-1);
    m |= half_bits;
    x = reinterpret_cast<const double&>(m);
    lt = (x < 1/std::sqrt(2.)) ? -1 : 0;
    dbl_e = e + lt - exponent_bias;
    z = x - (half + (lt ? 0. : half));
    y = half * (x - (lt ? half : 0.)) + half;
    x = z/y;
    z = x*x;
    u = z   + -3.56722798512324312549E1;
    t =       -7.89580278884799154124E-1;
    u = u*z +  3.12093766372244180303E2;
    t = t*z +  1.63866645699558079767E1;
    u = u*z + -7.69691943550460008604E2;
    t = t*z + -6.41409952958715622951E1;
    y = z* (t/u) + (half+half);
    return x*(y*std::log2(std::exp(1.))) + dbl_e;
}

template<bool WithMoment>
double Iterate(double zr, double zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    double cr = zr, sr = cr;
    double ci = zi, si = ci;
    double dist;
    int iter = maxiter, notescaped = -1;

    if(zr*(1+zr*(8*zr*zr+(16*zi*zi-3)))+zi*zi*(8*zi*zi-3) < 3./32 || ((zr+1)*(zr+1)+zi*zi)<1./16) { iter=0; }

    while(notescaped)
    {
        double r2 = cr * cr;
        double i2 = ci * ci;
        dist = r2 + i2;

        notescaped &= ((iter != 0) & (dist < escape_radius_squared)) ? -1 : 0;
        iter += notescaped;

        double ri = cr * ci;
        ci = zi + (ri * 2);
        cr = zr + (r2 - i2);

        if(WithMoment)
        {
            bool notmoment = iter & (iter-1);
            iter = (cr == sr && ci == si) ? 0 : iter;
            sr = notmoment ? sr : cr;
            si = notmoment ? si : ci;
        }
    }
    return iter ? mylog2( maxiter-iter + 1 - mylog2(mylog2(dist) / 2)) * (4/std::log2(std::exp(1.))) : 0;
}

int main(int argc, char *argv[])
{
    bool NeedMoment = true;
    w = SDL_CreateWindow("zoom", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Xres, Yres, SDL_WINDOW_RESIZABLE);
    r = SDL_CreateRenderer(w, -1, 0);
    t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, Xres, Yres);

    MAINLOOP_START(1);
    while(MAINLOOP_GET_CONDITION())
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        std::atomic<unsigned>    y_done{0}, n_inside{0};
        std::vector<std::thread> threads;
        for(unsigned n=0; n<std::thread::hardware_concurrency(); ++n)
            threads.emplace_back([&](){
                unsigned count_inside = 0;
                for(unsigned y; (y = y_done++) < Yres; )
                {
                    double i = zi+yscale*int(y-Yres/2);
                    if(NeedMoment)
                        for(unsigned x=0; x<Xres; ++x)
                        {
                            double v = Iterate<true>( zr+xscale*int(x-Xres/2), i );
                            if(v == 0.) ++count_inside;
                            pixels[y*Xres + x] = Color(x,y,v);
                        }
                    else
                        for(unsigned x=0; x<Xres; ++x)
                        {
                            double v = Iterate<false>( zr+xscale*int(x-Xres/2), i );
                            if(v == 0.) ++count_inside;
                            pixels[y*Xres + x] = Color(x,y,v);
                        }
                }
                n_inside += count_inside;
            });

        for(auto& t: threads) t.join();

        NeedMoment = n_inside >= (Xres*Yres)/1024;
        CHECK_TERMINATE();
        MAINLOOP_PUT_RESULT(pixels);
    }
    MAINLOOP_FINISH();
}
