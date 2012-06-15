/*
This file belongs to the Ray tracing tutorial of http://www.codermind.com/
It is free to use for educational purpose and cannot be redistributed
outside of the tutorial pages.
Any further inquiry :
mailto:info@codermind.com
*/

#include <vector>
#include <fstream>
#include <cutil_inline.h>

using namespace std;

enum PrimitiveType { ptSphere, ptCheckboard };

// --------------------------------------------------------------------------------
struct vecteur {
   float x, y, z;

   vecteur& operator += (const vecteur &v2){
      this->x += v2.x;
      this->y += v2.y;
      this->z += v2.z;
      return *this;
   }
};

struct color {
   enum OFFSET 
   {
      OFFSET_RED = 0,
      OFFSET_GREEN = 1,
      OFFSET_BLUE = 2,
      OFFSET_MAX  = 3
   };
   float red, green, blue;

   __device__ inline color & operator += (const color &c2 ) {
      this->red +=  c2.red;
      this->green += c2.green;
      this->blue += c2.blue;
      return *this;
   }

   __device__ inline float & getChannel(OFFSET offset )
   {
      return reinterpret_cast<float*>(this)[offset];
   }

   __device__ inline float getChannel(OFFSET offset ) const
   {
      return reinterpret_cast<const float*>(this)[offset];
   }
};

struct Material {
   color diffuse;
   float reflection;
   color specular;
   float power;
};

struct Primitive 
{
   float3        center;
   float         size;
   float3        color;
   float         reflection;
   PrimitiveType type;
   int           materialId;
};

struct Lamp
{
   float3 center;
   color  intensity;
};

// --------------------------------------------------------------------------------
// Raytrace
// --------------------------------------------------------------------------------
const int  _dNbPrimitives = 3;
const int  _dNbLamps      = 2;
const int  _dNbMaterials  = 3;
Primitive* _dPrimitives;
Lamp*      _dLamps;
Material*  _dMaterials;
const unsigned int width = 800, height = 600;


// --------------------------------------------------------------------------------
__device__ int make_color(float r, float g, float b, float a)
{
   r = (r > 0.9f) ? 0.9f : r;
   g = (g > 0.9f) ? 0.9f : g;
   b = (b > 0.9f) ? 0.9f : b;
   return
      ((int)(a * 255.0f) << 24) |
      ((int)(b * 255.0f) << 16) |
      ((int)(g * 255.0f) <<  8) |
      ((int)(r * 255.0f) <<  0);
}

__device__ float srgbEncode(float c)
{
   if (c <= 0.0031308f) {
      return 12.92f * c; 
   }
   else {
      return 1.055f * powf(c, 0.4166667f) - 0.055f; // Inverse gamma 2.4
   }
}

__device__ inline color operator * (const color&c1, const color &c2 ) {
   color c = {c1.red * c2.red, c1.green * c2.green, c1.blue * c2.blue};
   return c;
}

__device__ inline color operator + (const color&c1, const color &c2 ) {
   color c = {c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue};
   return c;
}

__device__ inline color operator * (float coef, const color &c ) {
   color c2 = {c.red * coef, c.green * coef, c.blue * coef};
   return c2;
}

struct ray {
   float3 start;
   vecteur dir;
};

__device__ inline float3 operator + (const float3&p, const vecteur &v){
   float3 p2={p.x + v.x, p.y + v.y, p.z + v.z };
   return p2;
}

__device__ inline float3 operator - (const float3&p, const vecteur &v){
   float3 p2={p.x - v.x, p.y - v.y, p.z - v.z };
   return p2;
}

__device__ inline vecteur operator + (const vecteur&v1, const vecteur &v2){
   vecteur v={v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
   return v;
}

__device__ inline vecteur operator - (const float3&p1, const float3 &p2){
   vecteur v={p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
   return v;
}

__device__ inline vecteur operator * (float c, const vecteur &v)
{
   vecteur v2={v.x *c, v.y * c, v.z * c };
   return v2;
}

__device__ inline vecteur operator - (const vecteur&v1, const vecteur &v2){
   vecteur v={v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
   return v;
}

__device__ inline float operator * (const vecteur&v1, const vecteur &v2 ) {
   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

struct Scene {
   std::vector<Material*>  materialContainer;
   std::vector<Primitive*> sphereContainer;
   std::vector<Lamp*>      lightContainer;
   int sizex, sizey;
};

//bool init(char* inputName, scene &myScene);

#define invsqrtf(x) (1.0f / sqrtf(x))

__device__ bool hitSphere(const ray &r, Primitive s, float &t)
{
   // Intersection of a ray and a sphere
   // Check the articles for the rationale
   // NB : this is probably a naive solution
   // that could cause precision problems
   // but that will do it for now. 
   vecteur dist = s.center - r.start;
   float B = (r.dir.x * dist.x + r.dir.y * dist.y + r.dir.z * dist.z);
   float D = B*B - dist*dist + s.size * s.size;
   if (D < 0.0f) return false;
   float t0 = B - sqrtf(D);
   float t1 = B + sqrtf(D);
   bool retvalue = false;
   if ((t0 > 0.1f ) && (t0 < t))
   {
      t = t0;
      retvalue = true;
   }
   if ((t1 > 0.1f ) && (t1 < t))
   {
      t = t1;
      retvalue = true;
   }
   return retvalue;
}

__device__ color addRay(ray viewRay, Lamp* _dLamps, Primitive* _dPrimitives, Material* _dMaterials )
{
   color output = {0.0f, 0.0f, 0.0f}; 
   float coef = 1.0f;
   int level = 0;
   do {
      float3 ptHitfloat3;
      int currentSphere=-1;
      float t = 2000.0f;
      for (unsigned int i = 0; i < _dNbPrimitives ; ++i) {
         if (hitSphere(viewRay, _dPrimitives[i], t)) {
            currentSphere = i;
         }
      }
      if (currentSphere == -1)
         break;

      ptHitfloat3  = viewRay.start + t * viewRay.dir;
      vecteur vNormal = ptHitfloat3 - _dPrimitives[currentSphere].center;
      float temp = vNormal * vNormal;
      if (temp == 0.0f)
         break;
      temp = 1.0f / sqrtf(temp);
      vNormal = temp * vNormal;

      Material currentMat = _dMaterials[_dPrimitives[currentSphere].materialId];

      ray lightRay;
      lightRay.start = ptHitfloat3;

      for (unsigned int j = 0; j < _dNbLamps ; ++j) {
         Lamp currentLight = _dLamps[j];

         lightRay.dir = currentLight.center - ptHitfloat3;
         float fLightProjection = lightRay.dir * vNormal;

         if ( fLightProjection <= 0.0f )
            continue;

         float lightDist = lightRay.dir * lightRay.dir;
         float temp = lightDist;
         if ( temp == 0.0f )
            continue;
         temp = invsqrtf(temp);
         lightRay.dir = temp * lightRay.dir;
         fLightProjection = temp * fLightProjection;

         bool inShadow = false;
         float t = lightDist;
         for (unsigned int i = 0; i < _dNbPrimitives ; ++i) {
            if (hitSphere(lightRay, _dPrimitives[i], t)) {
               inShadow = true;
               break;
            }
         }

         if (!inShadow) {
            float lambert = (lightRay.dir * vNormal) * coef;
            output.red   += lambert * currentLight.intensity.red   * currentMat.diffuse.red;
            output.green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
            output.blue  += lambert * currentLight.intensity.blue  * currentMat.diffuse.blue;

            // Blinn 
            // The direction of Blinn is exactly at mid float3 of the light ray 
            // and the view ray. 
            // We compute the Blinn vector and then we normalize it
            // then we compute the coeficient of blinn
            // which is the specular contribution of the current light.

            float fViewProjection = viewRay.dir * vNormal;
            vecteur blinnDir = lightRay.dir - viewRay.dir;
            float temp = blinnDir * blinnDir;
            if (temp != 0.0f ) {
               float blinn = invsqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
               blinn = coef * powf(blinn, currentMat.power);
               output += blinn*currentMat.specular * currentLight.intensity;
            }
         }
      }
      coef *= currentMat.reflection;
      float reflet = 2.0f * (viewRay.dir * vNormal);
      viewRay.start = ptHitfloat3;
      viewRay.dir = viewRay.dir - reflet * vNormal;
      level++;
   } while ((coef > 0.0f) && (level < 10));  
   return output;
}

__global__ void kernel( Lamp* lamps, Primitive* primitives, Material* materials, unsigned int* d_output )
{
   // Compute the index
   unsigned int x     = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   unsigned int y     = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
   unsigned int index = __umul24(y, width) + x;
   
   color output = {0.0f, 0.0f, 0.0f};
   for (float fragmentx = float(x) ; fragmentx < x + 1.0f; fragmentx += 0.5f ) {
      for (float fragmenty = float(y) ; fragmenty < y + 1.0f; fragmenty += 0.5f ) {
         float sampleRatio=0.25f;
         ray viewRay = { {fragmentx, fragmenty, -1000.0f},{ 0.0f, 0.0f, 1.0f}};
         color temp = addRay (viewRay, lamps, primitives, materials );
         // pseudo photo exposure
         float exposure = -1.00f; // random exposure value. TODO : determine a good value automatically
         temp.blue = (1.0f - expf(temp.blue * exposure));
         temp.red =  (1.0f - expf(temp.red * exposure));
         temp.green = (1.0f - expf(temp.green * exposure));

         output += sampleRatio * temp;
      }
   }

   // gamma correction
   output.blue = srgbEncode(output.blue);
   output.red = srgbEncode(output.red);
   output.green = srgbEncode(output.green);

   d_output[index] = make_color( output.red, output.green, output.blue, 255 );
}

void SetCoord(float3& v, float x, float y, float z )
{
   v.x = x; v.y = y; v.z = z;
}

void SetColor(color& c, float r, float g, float b )
{
   c.red = r; c.green = g; c.blue = b;
}

/**
* --------------------------------------------------------------------------------
* Raytracer
* --------------------------------------------------------------------------------
*/
texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex;  // 3D texture
cudaArray *d_volumeArray = 0;


extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
   tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(const unsigned char* h_volume, cudaExtent volumeSize)
{
   // create 3D array
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
   cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

   // copy data to 3D array
   cudaMemcpy3DParms copyParams = {0};
   copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, volumeSize.width*sizeof(unsigned char), volumeSize.width, volumeSize.height);
   copyParams.dstArray = d_volumeArray;
   copyParams.extent   = volumeSize;
   copyParams.kind     = cudaMemcpyHostToDevice;
   cutilSafeCall( cudaMemcpy3D(&copyParams) );

   // set texture parameters
   tex.normalized     = true;                      // access with normalized texture coordinates
   tex.filterMode     = cudaFilterModeLinear;      // linear interpolation
   tex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
   tex.addressMode[1] = cudaAddressModeWrap;
   tex.addressMode[2] = cudaAddressModeWrap;

   // bind array to 3D texture
   cutilSafeCall(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

extern "C" void render_kernel(float3 sphere, float radius, float reflect, unsigned int* d_output )
{
   const dim3 blockSize(16, 16, 1);
   const dim3 gridSize(width / blockSize.x, height / blockSize.y);

   // Primitives
   Primitive primitives[_dNbPrimitives];
   primitives[0].center = sphere;
   primitives[0].size  = radius; 
   primitives[0].materialId = 0;
   
   SetCoord( primitives[1].center,  407, 290, 0 );
   primitives[1].size  = 100; 
   primitives[1].materialId = 1;

   SetCoord( primitives[2].center,  320, 140, 0 ); 
   primitives[2].size  = 100; 
   primitives[2].materialId = 2;

   // Lamps
   Lamp lamps[_dNbLamps];
   SetCoord( lamps[0].center,   0, 240, -100 ); 
   SetColor( lamps[0].intensity, 2.0, 2.0, 2.0 );
   
   SetCoord( lamps[1].center, 640, 240, -10000); 
   SetColor( lamps[1].intensity, 0.6, 0.7, 1.0 );

   // Materials
   Material materials[_dNbMaterials];
   SetColor( materials[0].diffuse, 1.0, 1.0, 0.0 );
   materials[0].reflection = reflect; 
   SetColor( materials[0].specular, 1.0, 1.0, 1.0 ); 
   materials[0].power = 60;

   SetColor( materials[1].diffuse, 0.0, 1.0, 0.0 );
   materials[1].reflection = 0.5; 
   SetColor( materials[1].specular, 1.0, 1.0, 1.0 ); 
   materials[1].power = 60;

   SetColor( materials[2].diffuse, 1.0, 0.0, 1.0 );
   materials[2].reflection = 0.5; 
   SetColor( materials[2].specular, 1.0, 1.0, 1.0 ); 
   materials[2].power = 60;


   cutilSafeCall(cudaMalloc( (void**)&_dPrimitives, _dNbPrimitives*sizeof(Primitive) ));
   cutilSafeCall(cudaMalloc( (void**)&_dLamps,      _dNbLamps*sizeof(Lamp) ));
   cutilSafeCall(cudaMalloc( (void**)&_dMaterials,  _dNbMaterials*sizeof(Material) ));

   cutilSafeCall(cudaMemcpy( _dPrimitives, primitives, _dNbPrimitives*sizeof(Primitive),cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dLamps,      lamps,      _dNbLamps*sizeof(Lamp),          cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dMaterials,  materials,  _dNbMaterials*sizeof(Material),  cudaMemcpyHostToDevice));

   kernel<<<gridSize, blockSize>>>( _dLamps, _dPrimitives, _dMaterials, d_output );

   /*
   cutilSafeCall(cudaMemcpy(primitives,_dPrimitives,_dNbPrimitives*sizeof(Primitive),cudaMemcpyDeviceToHost));
   cutilSafeCall(cudaMemcpy(lamps,     _dLamps,     _dNbLamps*sizeof(Lamp),          cudaMemcpyDeviceToHost));
   */

   cutilSafeCall(cudaFree(_dPrimitives));
   cutilSafeCall(cudaFree(_dLamps));
   cutilSafeCall(cudaFree(_dMaterials));
}
