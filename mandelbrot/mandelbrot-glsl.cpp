
#define _Alignof alignof  // Preserve _Alignof definition.
#include <GL/glew.h>      // Include GLEW before any OpenGL headers.
#include <SDL.h>
#include <SDL_opengl.h>


#include <cstdio>
#include <chrono>
#include <cmath>
#include <algorithm>

// Screen dimensions and fractal parameters.
static const unsigned int XRES = 1706, YRES = 960;
static const int MAX_ITER = 1000;

// Compute shader source – this shader maps each invocation to a pixel,
// iterates the Mandelbrot function with smoothing, and writes an RGBA color.
const char* computeShaderSrc = R"(#version 410
layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) uniform image2D destImage;

uniform float u_centerX;
uniform float u_centerY;
uniform float u_scale;
uniform int   u_maxIter;
uniform int   u_width;
uniform int   u_height;
/*
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
*/
vec3 palette[12] = vec3[12](
    vec3(0.00, 0.00, 0.00),
    vec3(64.0, 64.0, 126.0),
    vec3(126.0,159.0,1.0),
    vec3(17.0,144.0,159.0),
    vec3(22.0,104.0,24.0),
    vec3(56.0,207.0,63.0),
    vec3(252.0,255.0,0.0),
    vec3(208.0,153.0,36.0),
    vec3(95.0,0.0,9.0),
    vec3(220.0,55.0,10.0),
    vec3(1.0,142.0,254.0),
    vec3(107,20.0,188.0)
);

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    float x0 = u_centerX + (float(pixel.x) - float(u_width) * 0.5) * (u_scale / float(u_height));
    float y0 = u_centerY + (float(pixel.y) - float(u_height) * 0.5) * (u_scale / float(u_height));

    float x = 0.0, y = 0.0;
    int iter = 0;
    float x2 = 0.0, y2 = 0.0;
    const float escape = 4.0;
    for (iter = 0; iter < u_maxIter; iter++) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        if (x2 + y2 > escape)
            break;
    }
    
    float smoothIter = iter < u_maxIter ? float(iter) - log2(log2(sqrt(x2+y2))) : 0.0;

    uint m = 0x3Fu;
    uint part1 = ((uint(pixel.x) & 4u) / 4u) + ((uint(pixel.x) & 2u) * 2u) + ((uint(pixel.x) & 1u) * 16u);
    uint part2 = (((uint(pixel.x ^ pixel.y) & 4u) / 2u) + ((uint(pixel.x ^ pixel.y) & 2u) * 4u) + ((uint(pixel.x ^ pixel.y) & 1u) * 32u));
    uint d_uint = (part1 + part2) & m;
    float d = float(d_uint) / 64.0;

    int idx = int(mod(floor(smoothIter), 12.0));
    int idx2 = (idx + 1) % 12;
    float f = smoothIter - floor(smoothIter);
    vec3 col = mix(palette[idx], palette[idx2], f);
    col = clamp(col + d/255.0, 0.0, 1.0);
    
    imageStore(destImage, pixel, vec4(col, 1.0));
}
)";

// Vertex shader for full-screen quad.
const char* vertexShaderSrc = R"(#version 410
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texCoord;
out vec2 fragTexCoord;
void main(){
    fragTexCoord = texCoord;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

// Fragment shader that samples the computed texture.
const char* fragmentShaderSrc = R"(#version 410
in vec2 fragTexCoord;
out vec4 outColor;
uniform sampler2D renderedTexture;
void main(){
    outColor = texture(renderedTexture, fragTexCoord);
}
)";

// Utility function to compile a shader.
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::fprintf(stderr, "Shader compile error: %s\n", buffer);
    }
    return shader;
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return -1;
    }
    // Request OpenGL 4.0 context since macOS maxes out at 4.1.
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_Window* window = SDL_CreateWindow("Compute Shader Mandelbrot",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          XRES, YRES, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }
    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (!context) {
        std::fprintf(stderr, "SDL_GL_CreateContext Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (GLEW_OK != glewErr) {
        std::fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(glewErr));
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    // Clear any spurious error.
    glGetError();

    // Check if glBindImageTexture is available.
    if (glBindImageTexture == nullptr) {
        std::fprintf(stderr, "glBindImageTexture is not available on this system.\n");
        std::fprintf(stderr, "Compute shaders and image load/store require OpenGL 4.2/4.3 which are not supported on macOS's OpenGL 4.1 (or below).\n");
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Create texture for compute shader output.
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, XRES, YRES);

    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    GLenum imageErr = glGetError();
    if (imageErr != GL_NO_ERROR) {
        std::fprintf(stderr, "glBindImageTexture error: %d\n", imageErr);
        glDeleteTextures(1, &texture);
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Compile and link shaders.
    GLuint compShader = compileShader(GL_COMPUTE_SHADER, computeShaderSrc);
    GLuint compProgram = glCreateProgram();
    glAttachShader(compProgram, compShader);
    glLinkProgram(compProgram);

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint renderProgram = glCreateProgram();
    glAttachShader(renderProgram, vertShader);
    glAttachShader(renderProgram, fragShader);
    glLinkProgram(renderProgram);

    float quadVertices[] = {
        // positions      // texCoords
        -1.0f, -1.0f,      0.0f, 0.0f,
         1.0f, -1.0f,      1.0f, 0.0f,
         1.0f,  1.0f,      1.0f, 1.0f,
        -1.0f,  1.0f,      0.0f, 1.0f
    };
    unsigned int quadIndices[] = { 0, 1, 2, 2, 3, 0 };
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    auto startTime = std::chrono::steady_clock::now();
    bool running = true;
    SDL_Event event;
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
               (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                running = false;
        }
        
        auto now = std::chrono::steady_clock::now();
        float t = std::chrono::duration<float>(now - startTime).count();
        
        float centerX = -0.743639266077433f;
        float centerY =  0.131824786875559f;
        float scale = 4.0f * std::pow(2.0f, -std::min(t, 53.0f) * 0.7f);
        
        glUseProgram(compProgram);
        glUniform1f(glGetUniformLocation(compProgram, "u_centerX"), centerX);
        glUniform1f(glGetUniformLocation(compProgram, "u_centerY"), centerY);
        glUniform1f(glGetUniformLocation(compProgram, "u_scale"), scale);
        glUniform1i(glGetUniformLocation(compProgram, "u_maxIter"), MAX_ITER);
        glUniform1i(glGetUniformLocation(compProgram, "u_width"), XRES);
        glUniform1i(glGetUniformLocation(compProgram, "u_height"), YRES);
        glDispatchCompute((XRES + 15) / 16, (YRES + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(renderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(renderProgram, "renderedTexture"), 0);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        
        SDL_GL_SwapWindow(window);
    }
    
    glDeleteProgram(compProgram);
    glDeleteProgram(renderProgram);
    glDeleteShader(compShader);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &texture);
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

