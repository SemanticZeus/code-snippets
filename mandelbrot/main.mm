#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <simd/simd.h>
#import <chrono>

static const NSUInteger Xres = 900;
static const NSUInteger Yres = 700;
static const NSUInteger MaxIter = 500;

@interface MetalApp : NSObject <NSApplicationDelegate>
@property NSWindow *window;
@property NSView *view;
@property CAMetalLayer *layer;
@property id<MTLDevice> device;
@property id<MTLLibrary> library;
@property id<MTLComputePipelineState> pipeline;
@property id<MTLCommandQueue> queue;
@property id<MTLTexture> offscreen;
@end

@implementation MetalApp

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    self.device = MTLCreateSystemDefaultDevice();
    self.queue = [self.device newCommandQueue];

    NSError *err = nil;
    NSURL *libURL = [NSURL fileURLWithPath:@"mandelbrot.metallib"];
    self.library = [self.device newLibraryWithURL:libURL error:&err];
    if (!self.library) { NSLog(@"Shader load error: %@", err); exit(1); }

    id<MTLFunction> kernel = [self.library newFunctionWithName:@"mandelbrot"];
    self.pipeline = [self.device newComputePipelineStateWithFunction:kernel error:&err];

    NSRect frame = NSMakeRect(100, 100, Xres, Yres);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                               styleMask:(NSWindowStyleMaskTitled |
                                                          NSWindowStyleMaskClosable |
                                                          NSWindowStyleMaskResizable)
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
    [self.window setTitle:@"Metal Mandelbrot"];
    [self.window makeKeyAndOrderFront:nil];

    self.layer = [CAMetalLayer layer];
    self.layer.device = self.device;
    self.layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    self.layer.drawableSize = CGSizeMake(Xres, Yres);

    self.view = [[NSView alloc] initWithFrame:frame];
    [self.view setWantsLayer:YES];
    [self.view setLayer:self.layer];
    [self.window setContentView:self.view];

    // Create offscreen render target (32-bit float color)
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:Xres
                                                                                   height:Yres
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    self.offscreen = [self.device newTextureWithDescriptor:desc];

    [NSTimer scheduledTimerWithTimeInterval:1.0 / 60.0
                                     target:self
                                   selector:@selector(render)
                                   userInfo:nil
                                    repeats:YES];
}

- (void)render {
    id<CAMetalDrawable> drawable = [self.layer nextDrawable];
    if (!drawable) return;

    static auto start = std::chrono::high_resolution_clock::now();
    float seconds = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - start).count();
    float clamped = fminf(seconds, 53.0f);
    float scale = 4.0f * powf(2.0f, -clamped * 0.7f);
    vector_float2 center = {-0.743639266077433f, 0.131824786875559f};
    vector_float2 pixel_scale = {scale / Yres, scale / Yres};

    id<MTLBuffer> centerBuf = [self.device newBufferWithBytes:&center length:sizeof(center) options:0];
    id<MTLBuffer> scaleBuf  = [self.device newBufferWithBytes:&pixel_scale length:sizeof(pixel_scale) options:0];
    id<MTLBuffer> iterBuf   = [self.device newBufferWithBytes:&MaxIter length:sizeof(MaxIter) options:0];
    id<MTLBuffer> timeBuf   = [self.device newBufferWithBytes:&seconds length:sizeof(seconds) options:0];

    id<MTLCommandBuffer> cmd = [self.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:self.pipeline];
    [enc setTexture:self.offscreen atIndex:0];
    [enc setBuffer:centerBuf offset:0 atIndex:0];
    [enc setBuffer:scaleBuf offset:0 atIndex:1];
    [enc setBuffer:iterBuf  offset:0 atIndex:2];
    [enc setBuffer:timeBuf  offset:0 atIndex:3];

    MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
    MTLSize threadgroups = MTLSizeMake((Xres + 15) / 16, (Yres + 15) / 16, 1);
    [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
    [enc endEncoding];

    // Blit to drawable
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromTexture:self.offscreen
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(Xres, Yres, 1)
                toTexture:drawable.texture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];

    [cmd presentDrawable:drawable];
    [cmd commit];
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        MetalApp *delegate = [MetalApp new];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}

