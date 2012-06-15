#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <nuiapi.h>

//CUDA
texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex;  // 3D texture
cudaArray *d_volumeArray = 0;

// Raytrace

enum PrimitiveType { ptSphere, ptTriangle, ptCheckboard, ptCamera };

struct Primitive 
{
   PrimitiveType type;
   float         radius;
   float         reflection;
   float3        center;
   float3        color;
};

struct Lamp
{
   float3 center;
   float3 color;
   float  intensity;
};

const int nbMaxPrimitives    = 100;
const int nbStaticPrimitives = 3;
const int nbIterations       = 3; 

typedef unsigned int TColor;

const int  _dNbPrimitives = nbMaxPrimitives;
const int  _dNbLamps      = 2;
Primitive* _dPrimitives;
Lamp*      _dLamps;
float3*    _dRays;
GLubyte*   _dBkGround;
bool       viewChanged = true;


// --------------------------------------------------------------------------------
#define set(v,a,b,c) \
   v.x = a; v.y = b; v.z = c;

// --------------------------------------------------------------------------------
#define copy(v1,v2) \
   v1.x = v2.x; v1.y = v2.y; v1.z = v2.z;

// --------------------------------------------------------------------------------
#define makeColor(__r, __g, __b, __a, __color) \
{ \
   __r = (__r>0.9f) ? 0.9f : __r; \
   __g = (__g>0.9f) ? 0.9f : __g; \
   __b = (__b>0.9f) ? 0.9f : __b; \
   __color = ((int)(__a * 255.0f) << 24) | ((int)(__b * 255.0f) << 16) | ((int)(__g * 255.0f) <<  8) | ((int)(__r * 255.0f) <<  0); \
}

// --------------------------------------------------------------------------------
#define length(__vl) \
   sqrt( __vl.x*__vl.x + __vl.y*__vl.y + __vl.z*__vl.z )

// --------------------------------------------------------------------------------
#define fastLength(__v) \
   __v.x*__v.x + __v.y*__v.y + __v.z*__v.z

// --------------------------------------------------------------------------------
#define normalize( __v ) \
   { \
      float __d = length(__v); \
      if( __d != 0.0f ) {  \
         __v.x=__v.x/__d; \
         __v.y=__v.y/__d; \
         __v.z=__v.z/__d; \
      } \
   }

// --------------------------------------------------------------------------------
#define normalDotNormalized( __v1, __v2, __nd ) \
   { \
      normalize(__v1); \
      normalize(__v2); \
      __nd = (__v1.x*__v2.x + __v1.y*__v2.y + __v1.z*__v2.z); \
   }

// --------------------------------------------------------------------------------
#define normalDot( __v1, __v2, __nd ) \
   __nd = (__v1.x*__v2.x + __v1.y*__v2.y + __v1.z*__v2.z);

// --------------------------------------------------------------------------------
#define reflect( incident, normal, reflected ) \
   { \
      float __a; \
      normalDot( incident, normal, __a ); \
      float __d = 2.0f*__a; \
      set( reflected, incident.x-__d*normal.x, incident.y-__d*normal.y, incident.z-__d*normal.z); \
   }

// --------------------------------------------------------------------------------
#define subtract( v1, v2, r ) \
   r.x = v1.x-v2.x; \
   r.y = v1.y-v2.y; \
   r.z = v1.z-v2.z;

// --------------------------------------------------------------------------------
#define crossProduct( v1, v2, r ) \
   r.x = v1.y * v2.z - v1.z * v2.y; \
   r.y = v1.z * v2.x - v1.x * v2.z; \
   r.z = v1.x * v2.y - v1.y * v2.x;

// --------------------------------------------------------------------------------
#define isNull( v ) \
   (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f )

/**
--------------------------------------------------------------------------------
Sphere Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
--------------------------------------------------------------------------------
*/
#define sphereIntersection( __primitive, __origin, __ray, __O_C, __intersection, __collision ) \
   float si_A = 2.0f*(__ray.x*__ray.x + __ray.y*__ray.y + __ray.z*__ray.z);\
   if ( si_A==0.0f ) si_A=0.0001f; \
   float si_B = 2.0f*(__O_C.x*__ray.x + __O_C.y*__ray.y + __O_C.z*__ray.z);\
   float si_C = __O_C.x*__O_C.x + __O_C.y*__O_C.y + __O_C.z*__O_C.z - __primitive.radius*__primitive.radius;\
   float si_radius = si_B*si_B-2.0f*si_A*si_C;\
   float si_t1 = (-si_B-sqrt(si_radius))/si_A;\
   bool si_b1 = ( si_t1>0.0f );\
   set(__intersection, __origin.x+si_t1*__ray.x, __origin.y+si_t1*__ray.y, __origin.z+si_t1*__ray.z);\
   float si_t2 = (-si_B+sqrt(si_radius))/si_A;\
   bool si_b2 = ( si_t2>0.0f );\
   if( si_b2 ) {\
      float3 si_intersection2;\
      set(si_intersection2, __origin.x+si_t2*__ray.x, __origin.y+si_t2*__ray.y, __origin.z+si_t2*__ray.z);\
      if( si_b1 ) {\
         float3 si_o1, si_o2;\
         subtract( __intersection,   __origin, si_o1 );\
         subtract( si_intersection2, __origin, si_o2 );\
         if( fastLength(si_o1)>fastLength(si_o2) ) \
            __intersection = si_intersection2;\
      }\
      else \
         __intersection = si_intersection2;\
   }\
   __collision =( si_b1 || si_b2 );

// --------------------------------------------------------------------------------
// need for Normlisation for the returned Normal?
#define getNormalVector( __primitive, __intersection, __normal ) \
{ \
   switch( __primitive.type ) { \
      case ptSphere    : subtract( __intersection, __primitive.center, __normal ); break; \
      case ptTriangle  : set( __normal, 0.0f, 0.0f, -1.0f ); break; \
      case ptCheckboard: set( __normal, 0.0f, 1.0f, 0.0f ); break; \
   } \
}

/**
--------------------------------------------------------------------------------
Checkboard Intersection
primitive    : Object on which we want to find the intersection
origin       : Origin of the ray
orientation  : Orientation of the ray
intersection : Resulting intersection
invert       : true if we want to check the front of the sphere, false for the 
back
returns true if there is an intersection, false otherwise
--------------------------------------------------------------------------------
*/
#define checkboardIntersection( __primitive, __origin, __ray, __intersection, __collision ) \
{ \
   switch( __primitive.type ) { \
      case ptCheckboard    : \
         { \
            if( __ray.y<0.0f ) { \
               __intersection.y = __primitive.center.y; \
               float y = __origin.y-__primitive.center.y; \
               __intersection.x = __origin.x + y * __ray.x / -__ray.y; \
               __intersection.z = __origin.z + y * __ray.z / -__ray.y; \
               __collision = true; \
            } \
            break; \
         } \
      case ptCamera    :  \
         { \
            if( __ray.z<0.0f ) { \
               __intersection.z = __primitive.center.z; \
               float z = __origin.z-__primitive.center.z; \
               __intersection.x = __origin.x+z*__ray.x/__ray.z; \
               __intersection.y = __origin.y+z*__ray.y/__ray.z; \
               __collision = true; \
            } \
            break; \
         } \
   } \
}

/**
--------------------------------------------------------------------------------
Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray which origin is the considered 3D float3
and which direction is defined by the light source center.

              * Lamp                     Ray = Origin -> Light Source Center
               \
                \##
                #### object
                 ##
                   \
                    \  Origin
             --------O-------
--------------------------------------------------------------------------------
*/
#define shadow( __primitives, __nbVisiblePrimitives, __lampCenter, __origin, __objectId, __collision) \
{ \
   float3 shadow_intersection; \
   float3 shadow_O_L; \
   set( shadow_O_L, __lampCenter.x - __origin.x, __lampCenter.y - __origin.y, __lampCenter.z - __origin.z ); \
   int cptPrimitives = 0; \
   while( !__collision && cptPrimitives<__nbVisiblePrimitives ) { \
      if( cptPrimitives != __objectId && __primitives[cptPrimitives].type == ptSphere ) { \
         bool shadow_collision = false; \
         float3 shadow_O_C; \
         subtract( __origin, __primitives[cptPrimitives].center, shadow_O_C ); \
         float shadow_nd; \
         normalDot( shadow_O_L, shadow_O_C, shadow_nd ); \
         if( shadow_nd < -0.99f ) { \
            sphereIntersection( __primitives[cptPrimitives], __origin, shadow_O_L, shadow_O_C, shadow_intersection, shadow_collision ); \
            __collision |= shadow_collision; \
         }\
      } \
      cptPrimitives++; \
   } \
}

/**
* --------------------------------------------------------------------------------
* Colors
* --------------------------------------------------------------------------------
*/
#define getObjectColorAtIntersection( __primitive, __intersection, __colorAtIntersection ) \
{ \
   __colorAtIntersection = __primitive.color; \
   switch( __primitive.type ) { \
      case ptTriangle  :  \
      case ptSphere    :  \
         { \
            __colorAtIntersection = __primitive.color;  \
            break; \
         } \
      case ptCheckboard:  \
         { \
            int x = int((1000+__intersection.x)/__primitive.radius); \
            int z = int((1000+__intersection.z)/__primitive.radius); \
            if(x%2==0) { \
               if (z%2==0) { \
                  set( __colorAtIntersection, 0.0f, 0.0f, 0.0f ); \
               } \
            } \
            else { \
               if (z%2!=0) { \
                  set( __colorAtIntersection, 0.0f, 0.0f, 0.0f ); \
               } \
            } \
            break; \
         } \
      case ptCamera:  \
         { \
            int x = __intersection.x*__primitive.radius+320; \
            int y = __intersection.y*__primitive.radius+240; \
            if( x>=0 && x<640 && y>=0 && y<480 ) { \
               int index = (y*640+x)*4; \
               set( __colorAtIntersection, float(d_BkGround[index])/256.0f, float(d_BkGround[index+1])/256.0f, float(d_BkGround[index+2])/256.0f ); \
            } \
            else { \
               __colorAtIntersection = __primitive.color; \
            } \
            break; \
         } \
   } \
}

// --------------------------------------------------------------------------------
#define colorFromObject( __primitives, __nbVisiblePrimitives, __lamps, __rayDir, __NormalToSurface, __intersection, __objectId, __color ) \
{ \
   float cfo_lambert; \
   float3 cfo_lampsColor; \
   set(cfo_lampsColor, 0.0f, 0.0f, 0.0f ); \
   for( int cfo_cptLamps=0; cfo_cptLamps<_dNbLamps; cfo_cptLamps++ ) { \
      bool cfo_collision = false; \
      shadow( __primitives, __nbVisiblePrimitives, __lamps[cfo_cptLamps].center, __intersection, __objectId, cfo_collision ); \
      if( !cfo_collision )  { \
         float3 cfo_lightRay; \
         subtract( __lamps[cfo_cptLamps].center, __intersection, cfo_lightRay); \
         normalize(cfo_lightRay); \
         normalDot(cfo_lightRay, __NormalToSurface, cfo_lambert); \
         cfo_lambert = (cfo_lambert>0.0f) ? cfo_lambert : 0.0f; \
         cfo_lambert *= __lamps[cfo_cptLamps].intensity; \
         cfo_lampsColor.x += __lamps[cfo_cptLamps].color.x*cfo_lambert; \
         cfo_lampsColor.y += __lamps[cfo_cptLamps].color.y*cfo_lambert; \
         cfo_lampsColor.z += __lamps[cfo_cptLamps].color.z*cfo_lambert; \
      } \
   } \
   float3 cfo_color; \
   getObjectColorAtIntersection( __primitives[__objectId], __intersection, cfo_color  ); \
   set(__color,  \
      cfo_color.x*cfo_lampsColor.x, \
      cfo_color.y*cfo_lampsColor.y, \
      cfo_color.z*cfo_lampsColor.z); \
} 

/**
* --------------------------------------------------------------------------------
* Intersections with Objects
* --------------------------------------------------------------------------------
*/
#define quickSortPrimitives( __primitives, __nbPrimitives, __origin, __ray, __closestObjectId ) \
{ \
   __closestObjectId = 0; \
   float qsp_distance; \
   float qsp_minDistance = 1000000.0f; \
   float3 qsp_O_C; \
   float qsp_nd; \
   for( int qsp_cptObjects = 0; qsp_cptObjects<__nbPrimitives; qsp_cptObjects++ ) { \
      if( qsp_cptObjects != lr_newObjectId ) { \
         subtract( __primitives[qsp_cptObjects].center, __origin, qsp_O_C ); \
         normalDotNormalized( __ray, qsp_O_C, qsp_nd ); \
         if( qsp_nd > 0.99999f ) { \
            qsp_distance = length( qsp_O_C ) - __primitives[qsp_cptObjects].radius; \
            if(qsp_distance < qsp_minDistance) { \
               __closestObjectId = qsp_cptObjects; \
               qsp_minDistance = qsp_distance; \
            } \
         }\
      }\
   } \
}

#define newIntersectionWithObjects( __primitives, __nbPrimitives, __origin, __orientation, __closestObjectId, __closestIntersection, __intersections ) \
{ \
   __intersections = false; \
   float3 iwo_ray; \
   subtract( __orientation, __origin, iwo_ray ); \
   quickSortPrimitives( __primitives, __nbPrimitives, __origin, iwo_ray, __closestObjectId ); \
   switch( __primitives[__closestObjectId].type ) { \
      case ptSphere:  \
         { \
            float3 iwo_O_C; \
            subtract( __origin, __primitives[__closestObjectId].center, iwo_O_C ); \
            sphereIntersection( __primitives[__closestObjectId], __origin, iwo_ray, iwo_O_C, __closestIntersection, __intersections ); \
            break; \
         } \
      case ptTriangle:  \
         { \
            /*__intersections = TriangleIntersection( __primitives[__closestObjectId], __origin, __orientation, __closestIntersection );*/ \
            break; \
         } \
      case ptCheckboard: \
      case ptCamera: \
         { \
            checkboardIntersection( __primitives[__closestObjectId], __origin, iwo_ray, __closestIntersection, __intersections ); \
            break; \
         } \
   } \
}

/**
* --------------------------------------------------------------------------------
* Intersections with Objects
* --------------------------------------------------------------------------------
*/
#define intersectionWithObjects( __primitives, __nbPrimitives, __origin, __orientation, __closestObjectId, __closestIntersection, __intersections ) \
{ \
   __intersections = false; \
   float iwo_minDistance = 10000.0f; \
   float3 iow_intersection; \
   float3 iwo_ray; \
   subtract( __orientation, __origin, iwo_ray ); \
   for( int iwo_cptObjects = 0; iwo_cptObjects<__nbPrimitives; iwo_cptObjects++ ) { \
      if( iwo_cptObjects != lr_newObjectId ) { \
         bool   iwo_intersec = false; \
         float  iwo_a; \
         float3 iwo_O_C; \
         subtract( __origin, __primitives[iwo_cptObjects].center, iwo_O_C ); \
         normalDot( iwo_ray, iwo_O_C, iwo_a ); \
         if( iwo_a < 0.0f ) { \
            switch( __primitives[iwo_cptObjects].type ) { \
               case ptSphere:  \
                  { \
                     sphereIntersection( __primitives[iwo_cptObjects], __origin, iwo_ray, iwo_O_C, iow_intersection, iwo_intersec ); \
                     break; \
                  } \
               case ptTriangle:  \
                  { \
                     /*iwo_intersec = TriangleIntersection( __primitives[iwo_cptObjects], __origin, __orientation, iow_intersection );*/ \
                     break; \
                  } \
               case ptCheckboard: \
               case ptCamera: \
                  { \
                     checkboardIntersection( __primitives[iwo_cptObjects], __origin, iwo_ray, iow_intersection, iwo_intersec ); \
                     break; \
                  } \
            } \
            if( iwo_intersec ) { \
               float iwo_distance = \
                  (__origin.x-iow_intersection.x)*(__origin.x-iow_intersection.x) + \
                  (__origin.y-iow_intersection.y)*(__origin.y-iow_intersection.y) + \
                  (__origin.z-iow_intersection.z)*(__origin.z-iow_intersection.z); \
               if(iwo_distance<iwo_minDistance) { \
                  iwo_minDistance = iwo_distance; \
                  __closestObjectId = iwo_cptObjects; \
                  set(__closestIntersection, iow_intersection.x, iow_intersection.y, iow_intersection.z); \
                  __intersections = true; \
               } \
            } \
         } \
      } \
   } \
}

/**
*  ------------------------------------------------------------------------------------------ 
* Ray Intersections
*  ========================================================================================== 
*  Calculate the reflected vector                   
*                                                  
*                  ^ Normal to object surface (N)  
*       Reflection  |                              
*                 \ |  Eye (O_I)                    
*                  \| /                             
*   ----------------O--------------- Object surface 
*          closestIntersection                      
*                                                   
*  ========================================================================================== 
*  colours                                                                                    
*  ------------------------------------------------------------------------------------------ 
*  We now have to know the colour of this intersection                                        
*  Color_from_object will compute the amount of light received by the intersection float3 and 
*  will also compute the shadows. The resulted color is stored in result.                     
*  The first parameter is the closest object to the intersection (following the ray). It can  
*  be considered as a light source if its inner light rate is > 0.                            
*  ------------------------------------------------------------------------------------------ 
*/
#define launchRay( __primitives, __nbVisiblePrimitives, __lamps, __viewPos, __viewRay, __intersectionColor ) \
{ \
   float3 lr_intersectionColor; \
   set(lr_intersectionColor, 0.0f, 0.0f, 0.0f ); \
   int    lr_closestObjectId; \
   float3 lr_reflection; \
   float3 lr_closestIntersection; \
   int    lr_sourceObjectId = -1; \
   bool   lr_carryon = true; \
   float3 lr_newOrigin      = __viewPos; \
   float3 lr_newOrientation = __viewRay; \
   float  lr_newRatio       = 0.0f; \
   int    lr_newObjectId    = lr_sourceObjectId; \
   int    lr_iteration      = 0; \
   while( lr_iteration<nbIterations && lr_carryon ) { \
      intersectionWithObjects( __primitives, __nbVisiblePrimitives, lr_newOrigin, lr_newOrientation, lr_closestObjectId, lr_closestIntersection, lr_carryon); \
      if( lr_carryon ) { \
         float3 lr_O_I; \
         subtract( __viewPos, lr_closestIntersection, lr_O_I ); \
         float3 lr_normalToSurface; \
         getNormalVector( __primitives[lr_closestObjectId], lr_closestIntersection, lr_normalToSurface ); \
         reflect(lr_O_I,lr_normalToSurface,lr_reflection); \
         float3 lr_closestOrientation; \
         subtract( lr_closestIntersection, lr_reflection, lr_closestOrientation ); \
         float3 lr_objectColor; \
         colorFromObject( __primitives, __nbVisiblePrimitives, __lamps, lr_closestOrientation, lr_normalToSurface, lr_closestIntersection, lr_closestObjectId, lr_objectColor); \
         float lr_ratio = 1.0f-__primitives[lr_closestObjectId].reflection; \
         if( lr_iteration == 0 ) { \
            lr_intersectionColor.x = lr_objectColor.x*lr_ratio; \
            lr_intersectionColor.y = lr_objectColor.y*lr_ratio; \
            lr_intersectionColor.z = lr_objectColor.z*lr_ratio; \
            lr_newRatio = (1.0f-lr_ratio); \
         } \
         else { \
            lr_intersectionColor.x += lr_newRatio*lr_objectColor.x*lr_ratio; \
            lr_intersectionColor.y += lr_newRatio*lr_objectColor.y*lr_ratio; \
            lr_intersectionColor.z += lr_newRatio*lr_objectColor.z*lr_ratio; \
         } \
         lr_carryon = ( __primitives[lr_closestObjectId].reflection > 0.0f ); \
         lr_newOrigin      = lr_closestIntersection; \
         lr_newOrientation = lr_closestOrientation; \
         lr_newObjectId    = lr_closestObjectId; \
         lr_newRatio       = __primitives[lr_closestObjectId].reflection; \
      } \
      lr_iteration++; \
   } \
   makeColor( lr_intersectionColor.x, lr_intersectionColor.y, lr_intersectionColor.z, 0.0f, __intersectionColor ); \
}


/**
* --------------------------------------------------------------------------------
* Initialization Kernel!!!
* --------------------------------------------------------------------------------
*/
__global__ void init_rays_kernel(
   float3     viewPos, 
   float3     P1,
   float3     P4,
   float3     d12, 
   float3     d34,
   float2     imageSize, 
   float3*    rays)
{
   // Compute the index
   unsigned int x     = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   unsigned int y     = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
   unsigned int index = __umul24(y, imageSize.x) + x;

   // Ray orientation
   float3 Pt, Pb, V;
   set( Pt, P1.x+(d12.x*x), P1.y+(d12.y*x), P1.z+(d12.z*x) );
   set( Pb, P4.x+(d34.x*x), P4.y+(d34.y*x), P4.z+(d34.z*x) );
   set( V,  Pt.x-Pb.x, Pt.y-Pb.y, Pt.z-Pb.z);

   float3 dV;
   set( dV, V.x/imageSize.y, V.y/imageSize.y, V.z/imageSize.y );
   set( rays[index], Pb.x+(dV.x*y), Pb.y+(dV.y*y), Pb.z+(dV.z*y) );
}

/**
* --------------------------------------------------------------------------------
* Main Kernel!!!
* --------------------------------------------------------------------------------
*/
__global__ void kernel( 
   float3     viewPos, 
   float2     imageSize, 
   Primitive* primitives, 
   Lamp*      lamps, 
   int        nbVisiblePrimitives,
   int        nbLamps,
   float3*    rays,
   unsigned int* d_output,
   GLubyte*      d_BkGround )
{
   // Compute the index
   unsigned int x     = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   unsigned int y     = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
   unsigned int index = __umul24(y, imageSize.x) + x;

   launchRay( primitives, nbVisiblePrimitives, lamps, viewPos,  rays[index], d_output[index] );
}

/**
* --------------------------------------------------------------------------------
* Raytracer
* --------------------------------------------------------------------------------
*/
extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
   tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(const unsigned char* h_volume, cudaExtent volumeSize, int width, int height)
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

   // Allocate memory
   cutilSafeCall(cudaMalloc( (void**)&_dBkGround,   640*480*sizeof(DWORD)*4 ));
   cutilSafeCall(cudaMalloc( (void**)&_dPrimitives, _dNbPrimitives*sizeof(Primitive) ));
   cutilSafeCall(cudaMalloc( (void**)&_dLamps,      _dNbLamps*sizeof(Lamp) ));
   cutilSafeCall(cudaMalloc( (void**)&_dRays,       width*height*sizeof(float3) ));

}

extern "C"
void finalizeCuda()
{
   cutilSafeCall(cudaFree(_dBkGround));
   cutilSafeCall(cudaFree(_dPrimitives));
   cutilSafeCall(cudaFree(_dLamps));
   cutilSafeCall(cudaFree(_dRays));
}

extern "C"
void render_kernel(
   unsigned int* d_output, 
   int imageW, 
   int imageH, 
   float3 eye, 
   float3 sphere, 
   float radius, 
   float reflect, 
   PDWORD bkground, 
   PUSHORT depth, 
   float time,
   NUI_SKELETON_FRAME SkeletonFrame )
{
   // Initialisation de la scene

   // Objets:
   int nbVisiblePrimitives = nbStaticPrimitives;
   Primitive primitives[_dNbPrimitives];

   primitives[0].center = sphere;
   primitives[0].radius  = radius;
   set( primitives[0].color, 1.0f, 0.5f, 0.5f );
   primitives[0].reflection = 0.9f;
   primitives[0].type = ptSphere;

   set( primitives[1].center,  0.0f, -10.0f, 0.0f );
   primitives[1].radius  = 2.0f;
   set( primitives[1].color, 0.5f, 0.5f, 0.5f );
   primitives[1].reflection = 0.5f;
   primitives[1].type = ptCheckboard;

   set( primitives[2].center, -15.0f * cosf(time), -10.0f + 10.0f*abs(cosf(time)),  -15.0f*sinf(time) ); 
   primitives[2].radius  = 1.0f; 
   set( primitives[2].color, 0.5f, 1.0f, 0.5f ); 
   primitives[2].reflection =  0.5f; 
   primitives[2].type = ptSphere;

   for( int i=0; i<NUI_SKELETON_COUNT; ++i ) {
      if( SkeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED ) {
         for( int j=0; j<20; j++ ) {
            set( primitives[nbVisiblePrimitives].center,  
               SkeletonFrame.SkeletonData[i].SkeletonPositions[j].x * 10.f,
               SkeletonFrame.SkeletonData[i].SkeletonPositions[j].y * 10.f,
               SkeletonFrame.SkeletonData[i].SkeletonPositions[j].z * 10.f );
            primitives[nbVisiblePrimitives].radius  = 2.0f;
            set( primitives[nbVisiblePrimitives].color, 1.0f, 1.0f, 1.0f );
            primitives[nbVisiblePrimitives].reflection = 0.8f;
            primitives[nbVisiblePrimitives].type = ptSphere;
            nbVisiblePrimitives++;
         }
      }
   }

   // Lampes
   Lamp lamps[_dNbLamps];
   set( lamps[0].center,   0.0f,  200.0f, -200.0f ); lamps[0].intensity = 2.0f; set( lamps[0].color, 0.5f, 0.5f, 0.5f );
   set( lamps[1].center, -100.0f * sinf(time/2.2f),  200.0f, -100.0f * cosf(time/3.5f) ); lamps[1].intensity = 1.0f; set( lamps[1].color, 1.0f, 1.0f, 1.0f );

   cutilSafeCall(cudaMemcpy( _dPrimitives, primitives, nbVisiblePrimitives*sizeof(Primitive),cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dLamps,      lamps,      _dNbLamps*sizeof(Lamp),               cudaMemcpyHostToDevice));
#if 0
   cutilSafeCall(cudaMemcpy( _dBkGround,   bkground,   640*480*sizeof(DWORD)*4,           cudaMemcpyHostToDevice));
#endif // KINECT
   
   float3 topLeftCorner, bottomRightCorner;
   set( topLeftCorner,     (float)-imageW/40,(float) imageH/40, 0.0f );
   set( bottomRightCorner, (float) imageW/40,(float)-imageH/40, 0.0f );

   float2 imageSize;
   imageSize.x = (float)imageW;
   imageSize.y = (float)imageH;

   // 192 blocks maximum sur la GTX 260
   dim3 block(13, 13, 1);
   dim3 grid( unsigned int(imageSize.x/block.x), unsigned int(imageSize.y/block.y), 1);

   if( viewChanged ) {
      // Camera corners
      float3 P1,P2,P3,P4;
      set( P1, topLeftCorner.x,     topLeftCorner.y,     topLeftCorner.z );
      set( P2, bottomRightCorner.x, topLeftCorner.y,     topLeftCorner.z);
      set( P3, bottomRightCorner.x, bottomRightCorner.y, bottomRightCorner.z);
      set( P4, topLeftCorner.x,     bottomRightCorner.y, topLeftCorner.z );

      float3 V12, V34;
      set( V12, P2.x-P1.x, P2.y-P1.y, P2.z-P1.z );
      set( V34, P3.x-P4.x, P3.y-P4.y, P3.z-P4.z );

      float3 d12, d34;
      set( d12, V12.x/imageSize.x, V12.y/imageSize.x, V12.z/imageSize.x );
      set( d34, V34.x/imageSize.x, V34.y/imageSize.x, V34.z/imageSize.x );

      init_rays_kernel<<<grid,block>>>( eye, P1, P4, d12, d34, imageSize, _dRays );
      viewChanged = false;
      printf("View has changed\n");
   }

//   printf("%d objects\n",nbVisibleSpheres);

   kernel<<<grid,block>>>(
      eye, imageSize, _dPrimitives, _dLamps, nbVisiblePrimitives, _dNbLamps, _dRays, d_output, _dBkGround);

}
