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
#include <algorithm>

using namespace std;

const float PIOVER180 = 0.017453292519943295769236907684886f;

const int NbPrimitives   = 3;
const int NbLamps        = 2;
const int NbMaterials    = 3;
const int NbBlobs        = 1;


enum PrimitiveType { ptSphere, ptCheckboard };

// --------------------------------------------------------------------------------
// Vectors
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

__device__ inline float3 operator + (const float3&p, const vecteur &v)
{
   float3 p2={p.x + v.x, p.y + v.y, p.z + v.z };
   return p2;
}

__device__ inline float3 operator - (const float3&p, const vecteur &v)
{
   float3 p2={p.x - v.x, p.y - v.y, p.z - v.z };
   return p2;
}

__device__ inline vecteur operator + (const vecteur&v1, const vecteur &v2)
{
   vecteur v={v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
   return v;
}

__device__ inline vecteur operator - (const float3&p1, const float3 &p2)
{
   vecteur v={p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
   return v;
}

__device__ inline vecteur operator * (float c, const vecteur &v)
{
   vecteur v2={v.x *c, v.y * c, v.z * c };
   return v2;
}

__device__ inline vecteur operator - (const vecteur&v1, const vecteur &v2)
{
   vecteur v={v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
   return v;
}

__device__ inline float operator * (const vecteur&v1, const vecteur &v2 ) 
{
   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ float random()
{
   return 0.1f; // TODO
}

// --------------------------------------------------------------------------------
// Colors
// --------------------------------------------------------------------------------
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


// --------------------------------------------------------------------------------
// Perlin
// --------------------------------------------------------------------------------
struct Perlin
{
   int p[512];
};

void initPerlin(Perlin& perlin) 
{
   int permutation[] = { 151,160,137,91,90,15,
      131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
      190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
      88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
      77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
      102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
      135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
      5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
      223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
      129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
      251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
      49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
      138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };

   for (int i=0; i < 256 ; i++) {
      perlin.p[256+i] = perlin.p[i] = permutation[i];
   }
}

__device__ float fade(float t) 
{ 
   return t * t * t * (t * (t * 6 - 15) + 10); 
}
__device__ float lerp(float t, float a, float b) { 
   return a + t * (b - a); 
}

__device__ float grad(int hash, float x, float y, float z) {
   int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
   float u = h<8||h==12||h==13 ? x : y,   // INTO 12 GRADIENT DIRECTIONS.
      v = h<4||h==12||h==13 ? y : z;
   return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

__device__ float noise(Perlin* perlin, float x, float y, float z) 
{

   int X = (int)floor(x) & 255,                  // FIND UNIT CUBE THAT
      Y = (int)floor(y) & 255,                  // CONTAINS POINT.
      Z = (int)floor(z) & 255;
   x -= floor(x);                                // FIND RELATIVE X,Y,Z
   y -= floor(y);                                // OF POINT IN CUBE.
   z -= floor(z);
   float u = fade(x),                                // COMPUTE FADE CURVES
      v = fade(y),                                // FOR EACH OF X,Y,Z.
      w = fade(z);
   int A = perlin->p[X  ]+Y, AA = perlin->p[A]+Z, AB = perlin->p[A+1]+Z,      // HASH COORDINATES OF
      B = perlin->p[X+1]+Y, BA = perlin->p[B]+Z, BB = perlin->p[B+1]+Z;      // THE 8 CUBE CORNERS,

   return lerp(w, lerp(v, lerp(u, grad(perlin->p[AA  ], x  , y  , z   ),  // AND ADD
      grad(perlin->p[BA  ], x-1, y  , z   )), // BLENDED
      lerp(u, grad(perlin->p[AB  ], x  , y-1, z   ),  // RESULTS
      grad(perlin->p[BB  ], x-1, y-1, z   ))),// FROM  8
      lerp(v, lerp(u, grad(perlin->p[AA+1], x  , y  , z-1 ),  // CORNERS
      grad(perlin->p[BA+1], x-1, y  , z-1 )), // OF CUBE
      lerp(u, grad(perlin->p[AB+1], x  , y-1, z-1 ),
      grad(perlin->p[BB+1], x-1, y-1, z-1 ))));
}

// --------------------------------------------------------------------------------
// Rays
// --------------------------------------------------------------------------------
struct Ray {
   float3  start;
   vecteur dir;
};

// --------------------------------------------------------------------------------
// Materials
// --------------------------------------------------------------------------------
struct Material {
   enum {
      gouraud=0,
      noise=1,
      marble=2,
      turbulence=3
   } type;

   color diffuse;
   //Second diffuse color, optional for the procedural materials
   color diffuse2; 
   float bump, reflection, refraction, density;
   color specular;
   float power;
};

// --------------------------------------------------------------------------------
// Primitives
// --------------------------------------------------------------------------------
struct Primitive 
{
   float3        center;
   float         size;
   float3        color;
   float         reflection;
   PrimitiveType type;
   int           materialId;
};

__device__ bool hitSphere(const Ray &r, Primitive s, float &t)
{
   // Intersection of a Ray and a sphere
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

// --------------------------------------------------------------------------------
// Lamps
// --------------------------------------------------------------------------------
struct Lamp
{
   float3 center;
   color  intensity;
};

// --------------------------------------------------------------------------------
// Blobs
// --------------------------------------------------------------------------------

const int zoneNumber = 10;

// Space around a source of potential is divided into concentric spheric zones
// Each zone will define the gamma and beta number that approaches
// linearly (f(x) = gamma * x + beta) the curves of 1 / dist^2
// Since those coefficients are independant of the actual size of the spheres
// we can compute it once and only once in the initBlobZones function

// fDeltaInvSquare is the maximum value that the current float3 source in the current
// zone contributes to the potential field (defined incrementally)
// Adding them for each zone that we entered and exit will give us
// a conservative estimate of the value of that field per zone
// which allows us to exit early later if there is no chance
// that the potential hits our equipotential value.
struct
{
   float fCoef, fDeltaFInvSquare, fGamma, fBeta;
} zoneTab[zoneNumber] = 
{   
   {10.0f,     0, 0, 0},
   {5.0f,      0, 0, 0},
   {3.33333f,  0, 0, 0},
   {2.5f,      0, 0, 0},
   {2.0f,      0, 0, 0},
   {1.66667f,  0, 0, 0},
   {1.42857f,  0, 0, 0},
   {1.25f,     0, 0, 0},
   {1.1111f,   0, 0, 0},
   {1.0f,      0, 0, 0} 
};


// A second degree polynom is defined by its coeficient
// a * x^2 + b * x + c
struct Poly
{
   float a, b, c, fDistance, fDeltaFInvSquare;
};

struct Blob
{
   float3 centerList[256];
   int    centerSize;
   float  size;
   float  invSizeSquare;
   int    materialId;
};

// Predicate we use to sort polys per distance on the intersecting Ray
struct IsLessPredicate
{
   bool operator () ( Poly& elem1, Poly& elem2 ) {
      return elem1.fDistance < elem2.fDistance;
   }
};

__device__ bool isBlobIntersected(const Ray &r, const Blob &b, float &t)
{
#if 0
   // Having a static structure helps performance more than two times !
   // It obviously wouldn't work if we were running in multiple threads..
   // But it helps considerably for now
   vector<Poly> polynomMap;
   polynomMap.resize(0);

   float rSquare, rInvSquare;
   rSquare = b.size * b.size;
   rInvSquare = b.invSizeSquare;
   float maxEstimatedPotential = 0.0f;

   // outside of all the influence spheres, the potential is zero
   float A = 0.0f;
   float B = 0.0f;
   float C = 0.0f;

   for (unsigned int i= 0; i< b.centerSize; i++)
   {
      float3 currentPoint = b.centerList[i];

      vecteur vDist = currentPoint - r.start;
      const float A = 1.0f;
      const float B = - 2.0f * r.dir * vDist;
      const float C = vDist * vDist; 
      // Accelerate delta computation by keeping common computation outside of the loop
      const float BSquareOverFourMinusC = 0.25f * B * B - C;
      const float MinusBOverTwo = -0.5f * B; 
      const float ATimeInvSquare = A * rInvSquare;
      const float BTimeInvSquare = B * rInvSquare;
      const float CTimeInvSquare = C * rInvSquare;

      // the current sphere, has N zones of influences
      // we go through each one of them, as long as we've detected
      // that the intersecting Ray has hit them
      // Since all the influence zones of many spheres
      // are imbricated, we compute the influence of the current sphere
      // by computing the delta of the previous polygon
      // that way, even if we reorder the zones later by their distance
      // on the Ray, we can still have our estimate of 
      // the potential function.
      // What is implicit here is that it only works because we've approximated
      // 1/dist^2 by a linear function of dist^2
      for (int j=0; j < zoneNumber - 1; j++)
      {
         // We compute the "delta" of the second degree equation for the current
         // spheric zone. If it's negative it means there is no intersection
         // of that spheric zone with the intersecting Ray
         const float fDelta = BSquareOverFourMinusC + zoneTab[j].fCoef * rSquare;
         if (fDelta < 0.0f) 
         {
            // Zones go from bigger to smaller, so that if we don't hit the current one,
            // there is no chance we hit the smaller one
            break;
         }
         const float sqrtDelta = sqrtf(fDelta);
         const float t0 = MinusBOverTwo - sqrtDelta; 
         const float t1 = MinusBOverTwo + sqrtDelta;

         // because we took the square root (a positive number), it's implicit that 
         // t0 is smaller than t1, so we know which is the entering float3 (into the current
         // sphere) and which is the exiting float3.
         Poly poly0 = {zoneTab[j].fGamma * ATimeInvSquare ,
            zoneTab[j].fGamma * BTimeInvSquare , 
            zoneTab[j].fGamma * CTimeInvSquare + zoneTab[j].fBeta,
            t0,
            zoneTab[j].fDeltaFInvSquare}; 
         Poly poly1 = {- poly0.a, - poly0.b, - poly0.c, 
            t1, 
            -poly0.fDeltaFInvSquare};

         maxEstimatedPotential += zoneTab[j].fDeltaFInvSquare;

         // just put them in the vector at the end
         // we'll sort all those float3 by distance later
         polynomMap.push_back(poly0);
         polynomMap.push_back(poly1);
      };
   }

   if (polynomMap.size() < 2 || maxEstimatedPotential < 1.0f)
   {
      return false;
   }

   // sort the various entry/exit points per distance
   // by going from the smaller distance to the bigger
   // we can reconstruct the field approximately along the way
   std::sort(polynomMap.begin(),polynomMap.end(), IsLessPredicate());

   maxEstimatedPotential = 0.0f;
   bool bResult = false;
   vector<Poly>::const_iterator it = polynomMap.begin();
   vector<Poly>::const_iterator itNext = it + 1;
   for (; itNext != polynomMap.end(); it = itNext, ++itNext)
   {
      // A * x2 + B * y + C, defines the condition under which the intersecting
      // Ray intersects the equipotential surface. It works because we designed it that way
      // (refer to the article).
      A += it->a;
      B += it->b;
      C += it->c;
      maxEstimatedPotential += it->fDeltaFInvSquare;
      if (maxEstimatedPotential < 1.0f)
      {
         // No chance that the potential will hit 1.0f in this zone, go to the next zone
         // just go to the next zone, we may have more luck
         continue;
      }
      const float fZoneStart =  it->fDistance;
      const float fZoneEnd = itNext->fDistance;

      // the current zone limits may be outside the Ray start and the Ray end
      // if that's the case just go to the next zone, we may have more luck
      if (t > fZoneStart &&  0.01f < fZoneEnd )
      {
         // This is the exact resolution of the second degree
         // equation that we've built
         // of course after all the approximation we've done
         // we're not going to have the exact float3 on the iso surface
         // but we should be close enough to not see artifacts
         float fDelta = B * B - 4.0f * A * (C - 1.0f) ;
         if (fDelta < 0.0f)
         {
            continue;
         }

         const float fInvA = (0.5f / A);
         const float fSqrtDelta = sqrtf(fDelta);

         const float t0 = fInvA * (- B - fSqrtDelta); 
         const float t1 = fInvA * (- B + fSqrtDelta);
         if ((t0 > 0.01f ) && (t0 >= fZoneStart ) && (t0 < fZoneEnd) && (t0 <= t ))
         {
            t = t0;
            bResult = true;
         }

         if ((t1 > 0.01f ) && (t1 >= fZoneStart ) && (t1 < fZoneEnd) && (t1 <= t ))
         {
            t = t1;
            bResult = true;
         }

         if (bResult)
         {
            return true;
         }
      }
   }
#endif // 0
   return false;
}

__device__ void blobInterpolation(float3& pos, const Blob& b, vecteur &vOut)
{
   vecteur gradient = {0.0f,0.0f,0.0f};

   float fRSquare = b.size * b.size;
   for (unsigned int i= 0; i< b.centerSize; i++) {
      // This is the true formula of the gradient in the
      // potential field and not an estimation.
      // gradient = normal to the iso surface
      vecteur normal = pos - b.centerList[i];
      float fDistSquare = normal * normal;
      if (fDistSquare <= 0.001f) 
         continue;
      float fDistFour = fDistSquare * fDistSquare;
      normal = (fRSquare/fDistFour) * normal;

      gradient = gradient + normal;
   }
   vOut = gradient;
}


// --------------------------------------------------------------------------------
// Raytrace
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

// --------------------------------------------------------------------------------
// Maps
// --------------------------------------------------------------------------------
struct Cubemap
{
	enum {
		up = 0,
		down = 1,
		right = 2,
		left = 3,
		forward = 4,
		backward = 5
	};

   //SimpleString name[6];
   int sizeX, sizeY;
   color *texture; 
   float exposure;
   bool bExposed;
   bool bsRGB;

   Cubemap() : sizeX(0), sizeY(0), texture(0), exposure(1.0f), bExposed(false), bsRGB(false) 
   {
   };
   bool Init();
   void setExposure(float newExposure) 
   {
      exposure = newExposure; 
   }
   ~Cubemap() 
   { 
      if (texture) 
         delete [] texture; 
   }
};

struct Perspective 
{
   enum {
      orthogonal = 0,
      conic = 1
   } Type;

   float FOV;
   float clearPoint;
   float dispersion;
   float invProjectionDistance;
};

// --------------------------------------------------------------------------------
// Cube Maps
// --------------------------------------------------------------------------------
__device__ color readTexture(const color* tab, float u, float v, int sizeU, int sizeV)
{
    u = fabsf(u);
    v = fabsf(v);
    int umin = int(sizeU * u);
    int vmin = int(sizeV * v);
    int umax = int(sizeU * u) + 1;
    int vmax = int(sizeV * v) + 1;
    float ucoef = fabsf(sizeU * u - umin);
    float vcoef = fabsf(sizeV * v - vmin);
    
    // The texture is being addressed on [0,1]
    // There should be an addressing type in order to 
    // determine how we should access texels when
    // the coordinates are beyond those boundaries.

    // Clamping is our current default and the only
    // implemented addressing type for now.
    // Clamping is done by bringing anything below zero
    // to the coordinate zero
    // and everything beyond one, to one.
    umin = min(max(umin, 0), sizeU - 1);
    umax = min(max(umax, 0), sizeU - 1);
    vmin = min(max(vmin, 0), sizeV - 1);
    vmax = min(max(vmax, 0), sizeV - 1);

    // What follows is a bilinear interpolation
    // along two coordinates u and v.

    color output =
        (1.0f - vcoef) * 
        ((1.0f - ucoef) * tab[umin  + sizeU * vmin] 
        + ucoef * tab[umax + sizeU * vmin])
        +   vcoef * 
        ((1.0f - ucoef) * tab[umin  + sizeU * vmax] 
        + ucoef * tab[umax + sizeU * vmax]);
    return output;
}

__device__ color readCubemap(const Cubemap& cm, const Ray& myRay)
{
    color * currentColor ;
    color outputColor = {0.0f,0.0f,0.0f};
    if(!cm.texture)
    {
        return outputColor;
    }
    if ((fabsf(myRay.dir.x) >= fabsf(myRay.dir.y)) && (fabsf(myRay.dir.x) >= fabsf(myRay.dir.z)))
    {
        if (myRay.dir.x > 0.0f)
        {
            currentColor = cm.texture + Cubemap::right * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.z / myRay.dir.x+ 1.0f) * 0.5f,  
                (myRay.dir.y / myRay.dir.x+ 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.x < 0.0f)
        {
            currentColor = cm.texture + Cubemap::left * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.z / myRay.dir.x+ 1.0f) * 0.5f,
                1.0f - ( myRay.dir.y / myRay.dir.x + 1.0f) * 0.5f,  
                cm.sizeX, cm.sizeY);
        }
    }
    else if ((fabsf(myRay.dir.y) >= fabsf(myRay.dir.x)) && (fabsf(myRay.dir.y) >= fabsf(myRay.dir.z)))
    {
        if (myRay.dir.y > 0.0f)
        {
            currentColor = cm.texture + Cubemap::up * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.y + 1.0f) * 0.5f,
                1.0f - (myRay.dir.z/ myRay.dir.y + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.y < 0.0f)
        {
            currentColor = cm.texture + Cubemap::down * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.x / myRay.dir.y + 1.0f) * 0.5f,  
                (myRay.dir.z/myRay.dir.y + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
    }
    else if ((fabsf(myRay.dir.z) >= fabsf(myRay.dir.x)) && (fabsf(myRay.dir.z) >= fabsf(myRay.dir.y)))
    {
        if (myRay.dir.z > 0.0f)
        {
            currentColor = cm.texture + Cubemap::forward * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.z + 1.0f) * 0.5f,  
                (myRay.dir.y/myRay.dir.z + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.z < 0.0f)
        {
            currentColor = cm.texture + Cubemap::backward * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.z + 1.0f) * 0.5f,  
                1.0f - (myRay.dir.y /myRay.dir.z+1) * 0.5f, cm.sizeX, cm.sizeY);
        }
    }
    if (cm.bsRGB)
    {
       // We make sure the data that was in sRGB storage mode is brought back to a 
       // linear format. We don't need the full accuracy of the sRGBEncode function
       // so a powf should be sufficient enough.
       outputColor.blue   = powf(outputColor.blue, 2.2f);
       outputColor.red    = powf(outputColor.red, 2.2f);
       outputColor.green  = powf(outputColor.green, 2.2f);
    }

    if (cm.bExposed)
    {
        // The LDR (low dynamic range) images were supposedly already
        // exposed, but we need to make the inverse transformation
        // so that we can expose them a second time.
        outputColor.blue  = -logf(1.001f - outputColor.blue);
        outputColor.red   = -logf(1.001f - outputColor.red);
        outputColor.green = -logf(1.001f - outputColor.green);
    }

    outputColor.blue  /= cm.exposure;
    outputColor.red   /= cm.exposure;
    outputColor.green /= cm.exposure;

    return outputColor;
}


// --------------------------------------------------------------------------------
// scene
// --------------------------------------------------------------------------------
struct Scene 
{
   int          width;
   int          height;
   Cubemap      cm;
   Perspective  perspective;
   int          complexity;
   struct {
      float MidPoint;
      float Power;
      float Black;
      float PowerScale;
   } Tonemap;
};

struct Context {
   float fRefractionCoef;
   color cLightScattering;
   int   level;
};

__device__ Context getDefaultAir() 
{
   Context airContext = {1.0f, {0.0f,0.0f,0.0f}, 0};
   return airContext;
};

#define invsqrtf(x) (1.0f / sqrtf(x))

// Device arrays
Primitive* _dPrimitives;
Lamp*      _dLamps;
Material*  _dMaterials;
Blob*      _dBlobs;
Perlin*    _dPerlin;

// --------------------------------------------------------------------------------
// Rays
// --------------------------------------------------------------------------------
__device__ color addRay(Scene& scene, Ray viewRay, Lamp* lamps, Primitive* primitives, Blob* blobs, Material* materials, Perlin* perlin, Context myContext )
{
   color output = {0.0f, 0.0f, 0.0f}; 
   float coef = 1.0f;
   int level = 0;
   do 
   {
      float3 ptHitPoint;
      vecteur vNormal;
      Material currentMat;
      {
         int currentBlob=-1;
         int currentSphere=-1;
         float t = 2000.0f;
         for (unsigned int i = 0; i < NbBlobs ; ++i) {
            if (isBlobIntersected(viewRay, blobs[i], t)) {
               currentBlob = i;
            }
         }

         for (unsigned int i = 0; i < NbPrimitives ; ++i) {
            if (hitSphere(viewRay, primitives[i], t)) {
               currentSphere = i;
               currentBlob = -1;
            }
         }

         if (currentBlob != -1) {
            ptHitPoint  = viewRay.start + t * viewRay.dir;
            blobInterpolation(ptHitPoint, blobs[currentBlob], vNormal);
            float temp = vNormal * vNormal;
            if (temp == 0.0f)
               break;
            vNormal = invsqrtf(temp) * vNormal;
            currentMat = materials[blobs[currentBlob].materialId];
         }
         else if (currentSphere != -1) {
            ptHitPoint  = viewRay.start + t * viewRay.dir;
            vNormal = ptHitPoint - primitives[currentSphere].center;
            float temp = vNormal * vNormal;
            if (temp == 0.0f)
               break;
            temp = invsqrtf(temp);
            vNormal = temp * vNormal;
            currentMat = materials[primitives[currentSphere].materialId];
         }
         else {
            break;
         }
      }

      float bInside;

      if (vNormal * viewRay.dir > 0.0f) {
         vNormal = -1.0f * vNormal;
         bInside = true;
      }
      else {
         bInside = false;
      }

      if (currentMat.bump) {
         float noiseCoefx = float(noise(perlin, 0.1 * float(ptHitPoint.x), 0.1 * float(ptHitPoint.y),0.1 * float(ptHitPoint.z)));
         float noiseCoefy = float(noise(perlin, 0.1 * float(ptHitPoint.y), 0.1 * float(ptHitPoint.z),0.1 * float(ptHitPoint.x)));
         float noiseCoefz = float(noise(perlin, 0.1 * float(ptHitPoint.z), 0.1 * float(ptHitPoint.x),0.1 * float(ptHitPoint.y)));

         vNormal.x = (1.0f - currentMat.bump ) * vNormal.x + currentMat.bump * noiseCoefx;  
         vNormal.y = (1.0f - currentMat.bump ) * vNormal.y + currentMat.bump * noiseCoefy;  
         vNormal.z = (1.0f - currentMat.bump ) * vNormal.z + currentMat.bump * noiseCoefz;  

         float temp = vNormal * vNormal;
         if (temp == 0.0f)
            break;
         temp = invsqrtf(temp);
         vNormal = temp * vNormal;
      }

      float fViewProjection = viewRay.dir * vNormal;
      float fReflectance, fTransmittance;
      float fCosThetaI, fSinThetaI, fCosThetaT, fSinThetaT;

      if(((currentMat.reflection != 0.0f) || (currentMat.refraction != 0.0f) ) && (currentMat.density != 0.0f)) {
         // glass-like Material, we're computing the fresnel coefficient.

         float fDensity1 = myContext.fRefractionCoef; 
         float fDensity2;
         if (bInside) {
            // We only consider the case where the Ray is originating a medium close to the void (or air) 
            // In theory, we should first determine if the current object is inside another one
            // but that's beyond the purpose of our code.
            fDensity2 = getDefaultAir().fRefractionCoef;
         }
         else {
            fDensity2 = currentMat.density;
         }

         // Here we take into account that the light movement is symmetrical
         // From the observer to the source or from the source to the oberver.
         // We then do the computation of the coefficient by taking into account
         // the Ray coming from the viewing float3.
         fCosThetaI = fabsf(fViewProjection); 

         if (fCosThetaI >= 0.999f) {
            // In this case the Ray is coming parallel to the normal to the surface
            fReflectance = (fDensity1 - fDensity2) / (fDensity1 + fDensity2);
            fReflectance = fReflectance * fReflectance;
            fSinThetaI = 0.0f;
            fSinThetaT = 0.0f;
            fCosThetaT = 1.0f;
         }
         else {
            fSinThetaI = sqrtf(1 - fCosThetaI * fCosThetaI);
            // The sign of SinThetaI has no importance, it is the same as the one of SinThetaT
            // and they vanish in the computation of the reflection coefficient.
            fSinThetaT = (fDensity1 / fDensity2) * fSinThetaI;
            if (fSinThetaT * fSinThetaT > 0.9999f) {
               // Beyond that angle all surfaces are purely reflective
               fReflectance = 1.0f ;
               fCosThetaT = 0.0f;
            }
            else {
               fCosThetaT = sqrtf(1 - fSinThetaT * fSinThetaT);
               // First we compute the reflectance in the plane orthogonal 
               // to the plane of reflection.
               float fReflectanceOrtho = (fDensity2 * fCosThetaT - fDensity1 * fCosThetaI ) 
                  / (fDensity2 * fCosThetaT + fDensity1  * fCosThetaI);
               fReflectanceOrtho = fReflectanceOrtho * fReflectanceOrtho;
               // Then we compute the reflectance in the plane parallel to the plane of reflection
               float fReflectanceParal = (fDensity1 * fCosThetaT - fDensity2 * fCosThetaI )
                  / (fDensity1 * fCosThetaT + fDensity2 * fCosThetaI);
               fReflectanceParal = fReflectanceParal * fReflectanceParal;

               // The reflectance coefficient is the average of those two.
               // If we consider a light that hasn't been previously polarized.
               fReflectance =  0.5f * (fReflectanceOrtho + fReflectanceParal);
            }
         }
      }
      else {
         // Reflection in a metal-like Material. Reflectance is equal in all directions.
         // Note, that metal are conducting electricity and as such change the polarity of the
         // reflected Ray. But of course we ignore that..
         fReflectance = 1.0f;
         fCosThetaI = 1.0f;
         fCosThetaT = 1.0f;
      }

      fTransmittance = currentMat.refraction * (1.0f - fReflectance);
      fReflectance = currentMat.reflection * fReflectance;

      float fTotalWeight = fReflectance + fTransmittance;
      bool bDiffuse = false;

      if (fTotalWeight > 0.0f) {
         float fRoulette = (1.0f / RAND_MAX) * random();

         if (fRoulette <= fReflectance) {
            coef *= currentMat.reflection;

            float fReflection = - 2.0f * fViewProjection;

            viewRay.start = ptHitPoint;
            viewRay.dir = viewRay.dir + (fReflection * vNormal);
         }
         else if(fRoulette <= fTotalWeight)
         {
            coef *= currentMat.refraction;
            float fOldRefractionCoef = myContext.fRefractionCoef;
            if (bInside) {
               myContext.fRefractionCoef = getDefaultAir().fRefractionCoef;
            }
            else {
               myContext.fRefractionCoef = currentMat.density;
            }

            // Here we compute the transmitted Ray with the formula of Snell-Descartes
            viewRay.start = ptHitPoint;

            viewRay.dir = viewRay.dir + fCosThetaI * vNormal;
            viewRay.dir = (fOldRefractionCoef / myContext.fRefractionCoef) * viewRay.dir;
            viewRay.dir = viewRay.dir + ((-fCosThetaT) * vNormal);
         }
         else {
            bDiffuse = true;
         }
      }
      else {
         bDiffuse = true;
      }


      if (!bInside && bDiffuse) {
         // Now the "regular lighting"
         Ray lightRay;
         lightRay.start = ptHitPoint;
         for (unsigned int j = 0; j < NbLamps ; ++j) {
            Lamp currentLight = lamps[j];

            lightRay.dir = currentLight.center - ptHitPoint;
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
            for (unsigned int i = 0; i < NbPrimitives ; ++i) {
               if (hitSphere(lightRay, primitives[i], t)) {
                  inShadow = true;
                  break;
               }
            }
            for (unsigned int i = 0; i < NbBlobs ; ++i) {
               if (isBlobIntersected(lightRay, blobs[i], t)) {
                  inShadow = true;
                  break;
               }
            }

            if (!inShadow && (fLightProjection > 0.0f)) {
               float lambert = (lightRay.dir * vNormal) * coef;
               float noiseCoef = 0.0f;
               switch(currentMat.type)
               {
               case Material::turbulence:
                  {
                     for (int level = 1; level < 10; level ++)
                     {
                        noiseCoef += (1.0f / level )  
                           * fabsf(float(noise(perlin, level * 0.05 * ptHitPoint.x,  
                           level * 0.05 * ptHitPoint.y,
                           level * 0.05 * ptHitPoint.z)));
                     };
                     output = output +  coef * (lambert * currentLight.intensity)  
                        * (noiseCoef * currentMat.diffuse + (1.0f - noiseCoef) * currentMat.diffuse2);
                  }
                  break;
               case Material::marble:
                  {
                     for (int level = 1; level < 10; level ++)
                     {
                        noiseCoef +=  (1.0f / level)  
                           * fabsf(float(noise(perlin, level * 0.05 * ptHitPoint.x,  
                           level * 0.05 * ptHitPoint.y,  
                           level * 0.05 * ptHitPoint.z)));
                     };
                     noiseCoef = 0.5f * sinf( (ptHitPoint.x + ptHitPoint.y) * 0.05f + noiseCoef) + 0.5f;
                     output = output +  coef * (lambert * currentLight.intensity)  
                        * (noiseCoef * currentMat.diffuse + (1.0f - noiseCoef) * currentMat.diffuse2);
                  }
                  break;
               default:
                  {
                     output.red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
                     output.green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
                     output.blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;
                  }
                  break;
               }

               // Blinn 
               // The direction of Blinn is exactly at mid float3 of the light Ray 
               // and the view Ray. 
               // We compute the Blinn vector and then we normalize it
               // then we compute the coeficient of blinn
               // which is the specular contribution of the current light.

               vecteur blinnDir = lightRay.dir - viewRay.dir;
               float temp = blinnDir * blinnDir;
               if (temp != 0.0f )
               {
                  float blinn = invsqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                  blinn = coef * powf(blinn, currentMat.power);
                  output += blinn *currentMat.specular  * currentLight.intensity;
               }
            }
         }
         coef = 0.0f ;
      }

      level++;
   } while ((coef > 0.0f) && (level < 10));  

   if (coef > 0.0f) {
      output += coef * readCubemap(scene.cm, viewRay);
   }
   return output;
}

// --------------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------------
__global__ void kernel( Scene scene, float exposure, Lamp* lamps, Primitive* primitives, Blob* blobs, Material* materials, Perlin* perlin, unsigned int* d_output )
{
   // Compute the index
   unsigned int x     = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
   unsigned int y     = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
   unsigned int index = __umul24(y, scene.width) + x;

   float sampleRatio = 0.25f;
   color output = {0.1f, 0.0f, 0.0f};
   color temp = {0.0f, 0.1f, 0.0f};

   for (float fragmentx = float(x) ; fragmentx < x + 1.0f; fragmentx += 0.5f ) {
      for (float fragmenty = float(y) ; fragmenty < y + 1.0f; fragmenty += 0.5f ) {
         float fTotalWeight = 0.0f;

         if (scene.perspective.Type == Perspective::orthogonal) {
            Ray viewRay = { {fragmentx, fragmenty, -10000.0f}, { 0.0f, 0.0f, 1.0f}};
            for (int i = 0; i < scene.complexity; ++i) {                  
               color rayResult = addRay( scene, viewRay, lamps, primitives, blobs, materials, perlin, getDefaultAir() );
               fTotalWeight += 1.0f; 
               temp += rayResult;
            }
            temp = (1.0f / fTotalWeight) * temp;
         }
         else {
            vecteur dir = {(fragmentx - 0.5f * scene.width) * scene.perspective.invProjectionDistance, 
               (fragmenty - 0.5f * scene.height) * scene.perspective.invProjectionDistance, 
               1.0f}; 

            float norm = dir * dir;
            if (norm == 0.0f) 
               break;
            dir = invsqrtf(norm) * dir;
            // the starting float3 is always the optical center of the camera
            // we will add some perturbation later to simulate a depth of field effect
            float3 start = {0.5f * scene.width,  0.5f * scene.height, 0.0f};
            // The float3 aimed is one of the invariant of the current pixel
            // that means that by design every Ray that contribute to the current
            // pixel must go through that float3 in space (on the "sharp" plane)
            // of course the divergence is caused by the direction of the Ray itself.
            float3 ptAimed = start + scene.perspective.clearPoint * dir;

            for (int i = 0; i < scene.complexity; ++i) {                  
               Ray viewRay = { {start.x, start.y, start.z}, {dir.x, dir.y, dir.z} };

               if (scene.perspective.dispersion != 0.0f) {
                  vecteur vDisturbance;                        
                  vDisturbance.x = (scene.perspective.dispersion / RAND_MAX) * (1.0f * random());
                  vDisturbance.y = (scene.perspective.dispersion / RAND_MAX) * (1.0f * random());
                  vDisturbance.z = 0.0f;

                  viewRay.start = viewRay.start + vDisturbance;
                  viewRay.dir = ptAimed - viewRay.start;

                  norm = viewRay.dir * viewRay.dir;
                  if (norm == 0.0f)
                     break;
                  viewRay.dir = invsqrtf(norm) * viewRay.dir;
               }
               color rayResult = addRay( scene, viewRay, lamps, primitives, blobs, materials, perlin, getDefaultAir() );
               fTotalWeight += 1.0f;
               temp += rayResult;
            }
            temp = (1.0f / fTotalWeight) * temp;
         }

         // pseudo photo exposure
         temp.blue  *= exposure;
         temp.red   *= exposure;
         temp.green *= exposure;

         if( scene.Tonemap.Black > 0.0f ) {
            temp.blue   = 1.0f - expf(scene.Tonemap.PowerScale * powf(temp.blue, scene.Tonemap.Power)   
               / (scene.Tonemap.Black + powf(temp.blue, scene.Tonemap.Power - 1.0f)) );
            temp.red    = 1.0f - expf(scene.Tonemap.PowerScale * powf(temp.red, scene.Tonemap.Power)    
               / (scene.Tonemap.Black + powf(temp.red, scene.Tonemap.Power - 1.0f)) );
            temp.green  = 1.0f - expf(scene.Tonemap.PowerScale * powf(temp.green, scene.Tonemap.Power)  
               / (scene.Tonemap.Black + powf(temp.green, scene.Tonemap.Power - 1.0f)) );
         }
         else {
            // If the black level is 0 then all other parameters have no effect
            temp.blue   = 1.0f - expf(scene.Tonemap.PowerScale * temp.blue);
            temp.red    = 1.0f - expf(scene.Tonemap.PowerScale * temp.red);
            temp.green  = 1.0f - expf(scene.Tonemap.PowerScale * temp.green);
         }
      }
      output = output + (sampleRatio * temp);
   }

   // gamma correction
   output.blue  = srgbEncode(output.blue);
   output.red   = srgbEncode(output.red);
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

// Exposure
float AutoExposure(Scene& scene)
{
    float exposure = -1.0f;
#if 0
    #define ACCUMULATION_SIZE 16
    float accufacteur = float(max(scene.width, scene.height));

    accufacteur = accufacteur / ACCUMULATION_SIZE;

    float mediumPoint = 0.0f;
    const float mediumPointWeight = 1.0f / (ACCUMULATION_SIZE*ACCUMULATION_SIZE);
    for (int y = 0; y < ACCUMULATION_SIZE; ++y) {
        for (int x = 0 ; x < ACCUMULATION_SIZE; ++x) {

            if (scene.perspective.Type == Perspective::orthogonal) {
                Ray viewRay = { {float(x)*accufacteur, float(y) * accufacteur, -1000.0f}, { 0.0f, 0.0f, 1.0f}};
                color currentColor = addRay( scene, viewRay, lamps, primitives, blobs, materials, perlin, getDefaultAir() );
                float luminance = 0.2126f * currentColor.red
                                + 0.715160f * currentColor.green
                                + 0.072169f * currentColor.blue;
                mediumPoint = mediumPoint + mediumPointWeight * (luminance * luminance);
            }
            else {
                vecteur dir = {(float(x)*accufacteur - 0.5f * scene.width) * scene.perspective.invProjectionDistance, 
                                (float(y) * accufacteur - 0.5f * scene.height) * scene.perspective.invProjectionDistance, 
                                1.0f}; 

                float norm = dir * dir;
                // I don't think this can happen but we've never too prudent
                if (norm == 0.0f) 
                    break;
                dir = invsqrtf(norm) * dir;

                Ray viewRay = { {0.5f * scene.width,  0.5f * scene.height, 0.0f}, {dir.x, dir.y, dir.z} };
                color currentColor = addRay( scene, viewRay, lamps, primitives, blobs, materials, perlin, getDefaultAir() );
                float luminance = 0.2126f * currentColor.red
                                + 0.715160f * currentColor.green
                                + 0.072169f * currentColor.blue;
                mediumPoint = mediumPoint + mediumPointWeight * (luminance * luminance);
            }
        }
    }
    
    float mediumLuminance = sqrtf(mediumPoint);

    if (mediumLuminance > 0.0f)
    {
        exposure = - logf(1.0f - scene.Tonemap.MidPoint) / mediumLuminance;
    }

#endif // 0
    return exposure;
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
   // Scene initialization
   Scene scene;
   scene.width            = 640;
   scene.height           = 480;
   scene.complexity       = 1;
   scene.perspective.Type = Perspective::conic;

   // field of view on horizontal axis (from extrem right to extrem left)
   // in degrees
   scene.perspective.FOV        = 90.0f;
   // Distance of the plane where objects are "in focus"
   scene.perspective.clearPoint = 450.0f;
   // Amount of blur that is added to objects not in focus
   scene.perspective.dispersion = 10.0f;

   scene.perspective.invProjectionDistance = 1.0f / (0.5f * scene.width / tanf (float(PIOVER180) * 0.5f * scene.perspective.FOV));

   // this user value determines if the image is low key or high key
   // used to compute exposure value
   //scene.Tonemap.Midpoint = 0.7f;
   // This enhances contrast if the black level is more than zero and has no effect otherwise
   scene.Tonemap.Power = 3.0f;
   // Affects flatness of the response curve in the black levels
   scene.Tonemap.Black = 0.1f;


   // Primitives
   Primitive primitives[NbPrimitives];
   primitives[0].center = sphere;
   primitives[0].size  = radius; 
   primitives[0].materialId = 0;

   SetCoord( primitives[1].center, 600, 290, 480 );
   primitives[1].size  = 100; 
   primitives[1].materialId = 1;

   SetCoord( primitives[2].center, 450, 140, 400 ); 
   primitives[2].size  = 50; 
   primitives[2].materialId = 2;

   // Lamps
   Lamp lamps[NbLamps];
   SetCoord( lamps[0].center,   0, 240, 300 ); 
   SetColor( lamps[0].intensity, 5.0, 5.0, 5.0 );

   SetCoord( lamps[1].center, 640, 480, -100); 
   SetColor( lamps[1].intensity, 0.6, 0.7, 1.0 );

   // Materials
   Material materials[NbMaterials];
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

   // Blobs
   Blob blobs[NbBlobs];
   SetCoord(blobs[0].centerList[0], 160.0, 290.0, 320.0);
   SetCoord(blobs[0].centerList[1], 400.0, 290.0, 480.0);
   SetCoord(blobs[0].centerList[2], 250.0, 140.0, 400.0); 
   blobs[0].centerSize = 3;
   blobs[0].size = 80;
   blobs[0].materialId = 2;

   //Perlin
   Perlin perlin[1];
   cutilSafeCall(cudaMalloc( (void**)&_dPerlin, sizeof(Perlin) ));
   initPerlin( perlin[0] );
   
   cutilSafeCall(cudaMalloc( (void**)&_dPrimitives, NbPrimitives*sizeof(Primitive) ));
   cutilSafeCall(cudaMalloc( (void**)&_dLamps,      NbLamps*sizeof(Lamp) ));
   cutilSafeCall(cudaMalloc( (void**)&_dMaterials,  NbMaterials*sizeof(Material) ));
   cutilSafeCall(cudaMalloc( (void**)&_dBlobs,      NbBlobs*sizeof(Blob) ));

   cutilSafeCall(cudaMemcpy( _dPrimitives, primitives, NbPrimitives*sizeof(Primitive),cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dLamps,      lamps,      NbLamps*sizeof(Lamp),          cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dMaterials,  materials,  NbMaterials*sizeof(Material),  cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dBlobs,      blobs,      NbBlobs*sizeof(Blob),          cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy( _dPerlin,     perlin,     sizeof(Perlin),                cudaMemcpyHostToDevice));

   // Kernel!!
   const dim3 blockSize(16, 16, 1);
   const dim3 gridSize(scene.width / blockSize.x, scene.height / blockSize.y);

   float exposure = 0.0f; // TODO: AutoExposure(scene);
   kernel<<<gridSize, blockSize>>>( scene, exposure, _dLamps, _dPrimitives, _dBlobs, _dMaterials, _dPerlin, d_output );

   cutilSafeCall(cudaFree(_dPrimitives));
   cutilSafeCall(cudaFree(_dLamps));
   cutilSafeCall(cudaFree(_dMaterials));
   cutilSafeCall(cudaFree(_dBlobs));
   cutilSafeCall(cudaFree(_dPerlin));
}
