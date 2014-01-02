
/* code for generating the below sinus table

  for i:=0 to 255 do sinus[i]:= round (sin (i*3.14/128) * 256)
  for i:=256 to 319 do sinus[i]:=sinus[i-256];
*/

signed int sinus[320] =
{
 0,5,12,18,24,30,37,43,49,55,61,67,73,79,85,91,97,103,108,114,120, 
 125,131,136,141,146,151,156,161,166,171,175,180,184,189,193,197,
 201,205,208,212,215,219,222,225,228,230,233,235,238,240,242,244,
 246,247,249,250,251,252,253,254,254,255,255,255,255,255,254,254,
 253,252,251,250,249,247,246,244,242,240,238,236,233,231,228,225,
 222,219,215,212,208,205,201,197,193,189,185,180,176,171,166,162,
 157,152,147,142,136,131,125,120,114,109,103,97,91,86,80,74,68,62,
 55,49,43,37,31,24,18,12,6,0,-6,-12,-18,-25,-31,-37,-43,-50,-56,
 -62,-68,-74,-80,-86,-92,-98,-103,-109,-115,-120,-126,-131,-137,
 -142,-147,-152,-157,-162,-167,-172,-176,-181,-185,-189,-194,-198,
 -201,-205,-209,-213,-216,-219,-222,-226,-228,-231,-234,-236,-239,
 -241,-243,-245,-247,-248,-250,-251,-252,-253,-254,-255,-255,-256,
 -256,-256,-256,-256,-255,-255,-254,-253,-252,-251,-250,-248,-247,
 -245,-243,-241,-239,-237,-234,-232,-229,-226,-223,-220,-217,-213,
 -210,-206,-202,-198,-194,-190,-186,-182,-177,-172,-168,-163,-158,
 -153,-148,-143,-138,-132,-127,-121,-116,-110,-104,-99,-93,-87,-81,
 -75,-69,-63,-57,-51,-45,-38,-32,-26,-20,-13,-7,0,5,12,18,24,30,37,
 43,49,55,61,67,73,79,85,91,97,103,108,114,120,125,131,136,141,146,
 151,156,161,166,171,175,180,184,189,193,197,201,205,208,212,215,
 219,222,225,228,230,233,235,238,240,242,244,246,247,249,250,251,
 252,253,254,254,255,255
};


#include <math.h>

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>


SDL_Surface *Screen;
SDL_Surface *image;

unsigned char * ScreenBuffer;


unsigned int ScreenWidth;
unsigned int ScreenHeight;
unsigned int BytesPerLine;
unsigned int ScreenBufferSize;
unsigned char Depth;


void VideoOn();
void VideoOff();
void ShowBuffer();
void InitLoadPicture();
SDL_Surface * ConvertFrom24to32 (SDL_Surface * source, SDL_PixelFormat *format32);

void InitCuda();

extern void Bump32 (unsigned int w, unsigned int h, signed int lx, signed int ly,
        unsigned char *dest);

extern int BumpMapTextureOn (unsigned int bumpSideSize);
extern int BumpMapTextureOff ();

extern int BumpSourceTextureOn(unsigned char *src,unsigned int w, unsigned int h);
extern int BumpSourceTextureOff();




