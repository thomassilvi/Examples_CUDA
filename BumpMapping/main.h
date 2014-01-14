

#include <math.h>

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>


SDL_Surface *Screen;
SDL_Surface *image;

signed int *sinus;
unsigned char * ScreenBuffer;


unsigned int ScreenWidth;
unsigned int ScreenHeight;
unsigned int BytesPerLine;
unsigned int ScreenBufferSize;
unsigned char Depth;


void InitSinusTable();
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




