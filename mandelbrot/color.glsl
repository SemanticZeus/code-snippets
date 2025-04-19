uint part1 = ((pixel.x & 4u) >> 2u)        // weight = 1
           + ((pixel.x & 2u) << 1u)        // weight = 4
           + ((pixel.x & 1u) << 4u);       // weight = 16

uint xorxy = pixel.x ^ pixel.y;
uint part2 = ((xorxy & 4u) >> 2u)          // weight = 1
           + ((xorxy & 2u) << 2u)          // weight = 8
           + ((xorxy & 1u) << 5u);         // weight = 32

uint d_uint = (part1 + part2) & 0x3Fu;
float d = float(d_uint) / 64.0;



const char* computeShaderSrc = R"(#version 410
layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) uniform image2D destImage;

uniform float u_centerX;
uniform float u_centerY;
uniform float u_scale;
uniform int   u_maxIter;
uniform int   u_width;
uniform int   u_height;

vec3 palette[12] = vec3[12](
    vec3(0.00, 0.00, 0.00),
    vec3(64.0/255.0, 64.0/255.0, 126.0/255.0),
    vec3(126.0/255.0,159.0/255.0,1.0),
    vec3(17.0/255.0,144.0/255.0,159.0/255.0),
    vec3(22.0/255.0,104.0/255.0,24.0/255.0),
    vec3(56.0/255.0,207.0/255.0,63.0/255.0),
    vec3(252.0/255.0,255.0/255.0,0.0),
    vec3(208.0/255.0,153.0/255.0,36.0/255.0),
    vec3(95.0/255.0,0.0,9.0/255.0),
    vec3(220.0/255.0,55.0/255.0,10.0/255.0),
    vec3(1.0,142.0/255.0,254.0/255.0),
    vec3(107.0/255.0,20.0/255.0,188.0/255.0)
);

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    // Map to Mandelbrot plane with center at (u_centerX, u_centerY)
    float x0 = u_centerX + (float(pixel.x) - float(u_width) * 0.5) * (u_scale / float(u_height));
    float y0 = u_centerY + (float(pixel.y) - float(u_height) * 0.5) * (u_scale / float(u_height));

    float x = 0.0, y = 0.0;
    float x2 = 0.0, y2 = 0.0;
    int iter;
    const float escape = 4.0;

    for (iter = 0; iter < u_maxIter; iter++) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        if (x2 + y2 > escape) break;
    }

    float smoothIter = iter < u_maxIter ? float(iter) - log2(log2(sqrt(x2 + y2))) : 0.0;

    // Centered coordinate system for dithering
    int cx = pixel.x - (u_width / 2);
    int cy = pixel.y - (u_height / 2);

    uint part1 = ((uint(cx) & 4u) >> 2u)
               + ((uint(cx) & 2u) << 1u)
               + ((uint(cx) & 1u) << 4u);

    uint xorxy = uint(cx ^ cy);
    uint part2 = ((xorxy & 4u) >> 2u)
               + ((xorxy & 2u) << 2u)
               + ((xorxy & 1u) << 5u);

    uint d_uint = (part1 + part2) & 0x3Fu;
    float d = float(d_uint) / 64.0;

    // Palette interpolation
    int idx = int(floor(smoothIter)) % 12;
    int idx2 = (idx + 1) % 12;
    float f = smoothIter - floor(smoothIter);
    vec3 col = mix(palette[idx], palette[idx2], f);
    col = clamp(col + d / 255.0, 0.0, 1.0);

    imageStore(destImage, pixel, vec4(col, 1.0));
}
)";



float r2 = x2 + y2;
float safe_r2 = max(r2, 1e-12);
float base = float(u_maxIter - iter + 1);
float smoothIter = iter < u_maxIter
    ? log2(base - log2(log2(safe_r2) * 0.5)) * (4.0 / log2(exp(1.0)))
    : 0.0;




const char* computeShaderSrc = R"(#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) uniform image2D destImage;

uniform float u_centerX;
uniform float u_centerY;
uniform float u_scale;
uniform int   u_maxIter;
uniform int   u_width;
uniform int   u_height;

vec3 palette[12] = vec3[12](
    vec3(0.00, 0.00, 0.00),
    vec3(64.0/255.0, 64.0/255.0, 126.0/255.0),
    vec3(126.0/255.0,159.0/255.0,1.0),
    vec3(17.0/255.0,144.0/255.0,159.0/255.0),
    vec3(22.0/255.0,104.0/255.0,24.0/255.0),
    vec3(56.0/255.0,207.0/255.0,63.0/255.0),
    vec3(252.0/255.0,255.0/255.0,0.0),
    vec3(208.0/255.0,153.0/255.0,36.0/255.0),
    vec3(95.0/255.0,0.0,9.0/255.0),
    vec3(220.0/255.0,55.0/255.0,10.0/255.0),
    vec3(1.0,142.0/255.0,254.0/255.0),
    vec3(107.0/255.0,20.0/255.0,188.0/255.0)
);

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    // Map pixel to Mandelbrot complex plane
    float scale = u_scale / float(u_height);
    float x0 = u_centerX + (float(pixel.x) - float(u_width) * 0.5) * scale;
    float y0 = u_centerY + (float(pixel.y) - float(u_height) * 0.5) * scale;

    float x = 0.0, y = 0.0;
    float x2 = 0.0, y2 = 0.0;
    int iter = 0;
    const float escape = 4.0;

    for (; iter < u_maxIter; ++iter) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        if (x2 + y2 > escape) break;
    }

    float r2 = x2 + y2;
    float smoothIter = 0.0;
    if (iter > 0 && iter < u_maxIter) {
        float log_term = log(log(max(r2, 1e-12)) / 2.0);
        float base = float(u_maxIter - iter + 1) - log_term;
        smoothIter = log2(base) * (4.0 / log2(exp(1.0)));
    }

    // Dithering using centered coordinates
    int cx = pixel.x - (u_width / 2);
    int cy = pixel.y - (u_height / 2);

    uint part1 = ((uint(cx) & 4u) >> 2u)
               + ((uint(cx) & 2u) << 1u)
               + ((uint(cx) & 1u) << 4u);

    uint xorxy = uint(cx ^ cy);
    uint part2 = ((xorxy & 4u) >> 2u)
               + ((xorxy & 2u) << 2u)
               + ((xorxy & 1u) << 5u);

    float d = float((part1 + part2) & 0x3Fu) / 64.0;

    // Palette interpolation
    int idx = int(floor(smoothIter)) % 12;
    int idx2 = (idx + 1) % 12;
    float f = smoothIter - floor(smoothIter);
    vec3 col = mix(palette[idx], palette[idx2], f);
    col = clamp(col + d / 255.0, 0.0, 1.0);`

    imageStore(destImage, pixel, vec4(col, 1.0));
}
)";







