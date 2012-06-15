#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cutil.h>
#include <driver_functions.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
#include <nuiapi.h>

const char *sSDKsample = "Realtime raytracing";

const cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

const unsigned int width    = 1280;
const unsigned int height   = 720;

// Kinect 	
HANDLE _skeleton(0);
HANDLE m_hNextDepthFrameEvent(0); 
HANDLE m_hNextVideoFrameEvent(0); 
HANDLE m_hNextSkeletonEvent(0);
HANDLE m_pVideoStreamHandle(0);
HANDLE m_pDepthStreamHandle(0);
NUI_SKELETON_FRAME SkeletonFrame;

// OpenGL
GLuint pbo;     // OpenGL pixel buffer object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

bool linearFiltering = true;
bool animate = true;

unsigned int *d_output = NULL;

#define MAX(a,b) ((a > b) ? a : b)

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(const unsigned char* h_volume, cudaExtent volumeSize, int width, int height);
extern "C" void finalizeCuda();
#if 1
extern "C" void render_kernel(
   unsigned int *d_output, 
   int imageW, 
   int imageH, 
   float3 eye, 
   float3 sphere, 
   float radius, 
   float reflect, 
   PDWORD bkground, 
   PUSHORT depth, 
   float time,
   NUI_SKELETON_FRAME SkeletonFrame );
#else
extern "C" void render_kernel(float3 sphere, float radius, float reflect, unsigned int* d_output );
#endif


// Testing
float3 eye, sphere;
float radius;
float step = 1.0f;
float reflect = 0.0f;
float time = 0.0f;
int   tick = 0;

// render image using CUDA
void render()
{
   // map PBO to get CUDA device pointer
   cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
   size_t num_bytes; 
   cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));

   // Kinect 
   HRESULT hr = NuiSkeletonGetNextFrame( 0, &SkeletonFrame );

   // call CUDA kernel, writing results to PBO
   render_kernel(d_output, width, height, eye, sphere, radius, reflect, 0, 0, time, SkeletonFrame );
   cutilCheckMsg("kernel failed");
   cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

}

// display results using OpenGL (called by GLUT)
void display()
{
   render();

   // display results
   glClear(GL_COLOR_BUFFER_BIT);
   glDisable(GL_DEPTH_TEST);
   glRasterPos2i(0, 0);
   glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
   glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
   glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
   glutSwapBuffers();
   glutReportErrors();

   time += 0.02f;
}

void idle()
{
}

void keyboard(unsigned char key, int x, int y)
{
   switch(key) {
      case 'a': eye.x -= step; break;
      case 'd': eye.x += step; break;
      case 'w': eye.y += step; break;
      case 's': eye.y -= step; break;
      case 'q': eye.z -= step; break;
      case 'e': eye.z += step; break;

      case '4': sphere.x -= step; break;
      case '6': sphere.x += step; break;
      case '8': sphere.y += step; break;
      case '2': sphere.y -= step; break;
      case '7': sphere.z -= step; break;
      case '9': sphere.z += step; break;

      case '1': reflect -= 0.05f; reflect = ( reflect < 0.0f ) ? 0.0f : reflect; break;
      case '3': reflect += 0.05f; reflect = ( reflect > 1.0f ) ? 1.0f : reflect; break;

      case '-': radius -= 1.0f; break;
      case '+': radius += 1.0f; break;

      case  27: exit(0); break;
      default: break;
   }
   glutPostRedisplay();
}

void reshape(int x, int y)
{
   glViewport(0, 0, x, y);

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup()
{
   finalizeCuda();
}

void initKinect()
{
   HRESULT hr = NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);
   
   m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
   m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
   m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

   _skeleton = CreateEvent( NULL, TRUE, FALSE, NULL );			 
   hr = NuiSkeletonTrackingEnable( _skeleton, 0 );

   hr = NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
   hr = NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

   hr = NuiCameraElevationSetAngle( 0 );
}

void initGLBuffers()
{
   // create pixel buffer object
   glGenBuffersARB(1, &pbo);
   glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
   glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
   //glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

   // register this buffer object with CUDA
   cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));	
}

void initGL( int *argc, char **argv )
{
   // initialize GLUT callback functions
   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(width, height);
   glutCreateWindow("CUDA Raytracer");
   //glutFullScreen();
   glutDisplayFunc(display);
   glutKeyboardFunc(keyboard);
   glutReshapeFunc(reshape);
   glutIdleFunc(display);

   eye.x =   0.0f;
   eye.y =   0.0f;
   eye.z = -30.0f;

   sphere.x = -10.0f;
   sphere.y =   5.0f;
   sphere.z =  -5.0f;
   radius   =   2.0f;

   glewInit();
   if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
      fprintf(stderr, "Required OpenGL extensions missing.");
      exit(-1);
   }
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, char **argv, bool bUseOpenGL)
{
   int result = 0;
   if (bUseOpenGL) {
      result = cutilChooseCudaGLDevice(argc, argv);
   } else {
      result = cutilChooseCudaDevice(argc, argv);
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
   printf("[%s]\n", sSDKsample);

   initKinect();

   // First initialize OpenGL context, so we can properly set the GL for CUDA.
   // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
   initGL(&argc, argv);

   // use command-line specified CUDA device, otherwise use device with highest Gflops/s
   chooseCudaDevice(argc, argv, true);

   // OpenGL buffers
   initGLBuffers();

   size_t size = volumeSize.width*volumeSize.height*volumeSize.depth;
   unsigned char *h_volume = (unsigned char *) malloc(size); //loadRawFile(path, size);

   initCuda( h_volume, volumeSize, width, height );
   free(h_volume);

   atexit(cleanup);

   glutMainLoop();

   cudaDeviceReset();
   cutilExit(argc, argv);

   return 0;
}


