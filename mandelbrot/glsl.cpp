#include <SDL.h>
#include <GL/glew.h>
#include <cstdio>
#include <cmath>
#include <chrono>

#define XRES 1706
#define YRES 960
#define MAX_ITER 1000

const char* computeShaderSrc = R"(
#version 430
layout (local_size_x = 16, local_size_y = 16) in;
layout (rgba8, binding = 0) uniform image2D destImage;

uniform float u_centerX, u_centerY, u_scale;
uniform int u_maxIter, u_width, u_height;

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

    float scale = u_scale / float(u_height);
    float x0 = u_centerX + (float(pixel.x) - float(u_width) * 0.5) * scale;
    float y0 = u_centerY + (float(pixel.y) - float(u_height) * 0.5) * scale;

    float x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0;
    int iter = 0;
    for (; iter < u_maxIter; ++iter) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        if (x2 + y2 > 4.0) break;
    }

    float r2 = x2 + y2;
    float smoothIter = 0.0;
    if (iter > 0 && iter < u_maxIter) {
        float log_term = log(log(max(r2, 1e-12)) / 2.0);
        float base = max(float(u_maxIter - iter + 1) - log_term, 1e-6);
        smoothIter = log2(base) * (4.0 / log2(exp(1.0)));
    }

    int cx = pixel.x - (u_width / 2);
    int cy = pixel.y - (u_height / 2);
    uint part1 = ((uint(cx) & 4u) >> 2u) + ((uint(cx) & 2u) << 1u) + ((uint(cx) & 1u) << 4u);
    uint xorxy = uint(cx ^ cy);
    uint part2 = ((xorxy & 4u) >> 2u) + ((xorxy & 2u) << 2u) + ((xorxy & 1u) << 5u);
    float d = float((part1 + part2) & 0x3Fu) / 64.0;

    float s = max(smoothIter, 0.0);
    int idx = int(floor(s)) % 12;
    int idx2 = (idx + 1) % 12;
    float f = s - floor(s);
    vec3 col = mix(palette[idx], palette[idx2], f);
    col = clamp(col + d / 255.0, 0.0, 1.0);

    imageStore(destImage, pixel, vec4(col, 1.0));
}
)";

const char* vertexShaderSrc = R"(
#version 410
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texCoord;
out vec2 fragTexCoord;
void main(){
    fragTexCoord = texCoord;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 410
in vec2 fragTexCoord;
out vec4 outColor;
uniform sampler2D renderedTexture;
void main(){
    outColor = texture(renderedTexture, fragTexCoord);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::fprintf(stderr, "Compile error:\n%s\n", log);
    }
    return shader;
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_Window* window = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, XRES, YRES, SDL_WINDOW_OPENGL);
    SDL_GLContext ctx = SDL_GL_CreateContext(window);
    glewExperimental = GL_TRUE;
    glewInit();

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, XRES, YRES);
    glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    GLuint comp = compileShader(GL_COMPUTE_SHADER, computeShaderSrc);
    GLuint compProg = glCreateProgram();
    glAttachShader(compProg, comp);
    glLinkProgram(compProg);

    GLuint vert = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLuint renderProg = glCreateProgram();
    glAttachShader(renderProg, vert);
    glAttachShader(renderProg, frag);
    glLinkProgram(renderProg);

    float quad[] = {
        -1, -1, 0, 0,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1
    };
    unsigned int idx[] = { 0, 1, 2, 2, 3, 0 };
    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    auto start = std::chrono::steady_clock::now();
    bool running = true;
    SDL_Event e;

    while (running) {
        while (SDL_PollEvent(&e)) if (e.type == SDL_QUIT) running = false;

        float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        float centerX = -0.743639266077433f;
        float centerY =  0.131824786875559f;
        float scale = 4.0f * std::pow(2.0f, -std::min(t, 53.0f) * 0.7f);

        glUseProgram(compProg);
        glUniform1f(glGetUniformLocation(compProg, "u_centerX"), centerX);
        glUniform1f(glGetUniformLocation(compProg, "u_centerY"), centerY);
        glUniform1f(glGetUniformLocation(compProg, "u_scale"), scale);
        glUniform1i(glGetUniformLocation(compProg, "u_maxIter"), MAX_ITER);
        glUniform1i(glGetUniformLocation(compProg, "u_width"), XRES);
        glUniform1i(glGetUniformLocation(compProg, "u_height"), YRES);
        glDispatchCompute((XRES + 15) / 16, (YRES + 15) / 16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(renderProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(renderProg, "renderedTexture"), 0);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        SDL_GL_SwapWindow(window);
    }

    glDeleteTextures(1, &tex);
    glDeleteProgram(compProg);
    glDeleteProgram(renderProg);
    glDeleteShader(comp);
    glDeleteShader(vert);
    glDeleteShader(frag);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

