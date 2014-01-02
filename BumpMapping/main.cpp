
#include "main.h"


int main(int argc, char **argv)
{
	SDL_Event event;
	SDL_Surface *surfaceTmp;
	unsigned char LetsGoOn;
	unsigned int angle;
	signed int lx, ly;
	unsigned char *deviceTarget;
	unsigned int frameCount;

	if (argc <= 1) 
	{
		printf("Usage: %s <image filename>\n",argv[0]);
		printf("Example: %s car.png\n");
		exit(0);
	}

	InitCuda();
	InitLoadPicture();

	printf ("Load %s\n",argv[1]);
	surfaceTmp = IMG_Load(argv[1]);

	if(!surfaceTmp) 
	{
		printf("IMG_Load: %s\n", IMG_GetError());
		IMG_Quit();
		exit(1);
	}

	printf("Width : %d\n",surfaceTmp->w);
	printf("Height : %d\n",surfaceTmp->h);
	printf("Depth : %d\n",surfaceTmp->format->BitsPerPixel);

	ScreenWidth = surfaceTmp->w;
	ScreenHeight = surfaceTmp->h;
	Depth = 32;
	BytesPerLine = ScreenWidth * (Depth / 8);
	ScreenBufferSize = BytesPerLine * ScreenHeight;


	VideoOn();


	if (surfaceTmp->format->BitsPerPixel < 32)
	{
		printf ("Convert to perform to 32bpp\n");	
		image = ConvertFrom24to32 (surfaceTmp, Screen->format);
	}
	else
	{
		image = surfaceTmp;
	}

	BumpMapTextureOn (256);
	BumpSourceTextureOn((unsigned char*)(image->pixels), 
		(unsigned int)(surfaceTmp->w), (unsigned int)(surfaceTmp->h));

	cudaMalloc ((void**) &deviceTarget, ScreenBufferSize);


	//

	angle = 0;
	frameCount=0;

	LetsGoOn=1;
	
	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start,0);

        while (LetsGoOn)
	{
		frameCount++;
                angle+=1;
                angle&=255;

                lx=(sinus[angle+64]/2)+ ScreenWidth / 2;
                ly=(sinus[angle]/2)+ ScreenHeight / 2;

		Bump32 (ScreenWidth, ScreenHeight, lx, ly, deviceTarget);
		cudaMemcpy (ScreenBuffer, deviceTarget, ScreenBufferSize, cudaMemcpyDeviceToHost);


		ShowBuffer();

                // manage events

                SDL_PollEvent(&event);

                switch (event.type)
                {
                        case SDL_KEYDOWN:
                        {
                                switch(event.key.keysym.sym)
                                {
                                        case SDLK_ESCAPE:
                                        {
                                                LetsGoOn=0;
                                                break;
                                        }
                                        default:
                                        {
                                                break;
                                        }
                                }
                                break;
                        }
                        case SDL_QUIT:
                        {
                                LetsGoOn=0;
                                break;
                        }
                }
	}

	cudaEventRecord (stop,0);
	cudaEventSynchronize (stop);

	float elapsedTime;
	cudaEventElapsedTime (&elapsedTime, start, stop);

	printf ("Frames count : %d\nElapsed time (ms) : %3.1f\n",frameCount,elapsedTime);
	printf ("FPS %3.1f\n",(1000.0 * frameCount)/elapsedTime);

	BumpSourceTextureOff();
	BumpMapTextureOff();

	cudaFree (deviceTarget);

	VideoOff();
}

//###################################################################################################

void VideoOn()
{
        if (SDL_Init(SDL_INIT_VIDEO)<0)
	{
		exit(1);
	}

        atexit(SDL_Quit);
        Screen=SDL_SetVideoMode(ScreenWidth,ScreenHeight,Depth,SDL_SWSURFACE);

        SDL_ShowCursor(SDL_DISABLE);

        if (Screen==NULL)
	{
		exit(1);
	}

        ScreenBuffer=(unsigned char *) Screen->pixels;

        if (SDL_MUSTLOCK(Screen)) 
        {
                SDL_LockSurface(Screen);
        }

}

//###################################################################################################

void VideoOff()
{
	IMG_Quit();
	SDL_Quit();
}

//###################################################################################################

void ShowBuffer()
{
        if (SDL_MUSTLOCK(Screen))
        { 
                SDL_UnlockSurface(Screen); 
        } 

        SDL_UpdateRect(Screen, 0, 0, 0, 0);

        if (SDL_MUSTLOCK(Screen)) 
        { 
                SDL_LockSurface(Screen); 
        }
}

//###################################################################################################

void InitLoadPicture()
{
	int flags = IMG_INIT_JPG;

	int initted = IMG_Init(flags);

	if(initted&flags != flags) 
	{
		printf("IMG_Init: Failed to init required jpg and png support!\n");
		printf("IMG_Init: %s\n", IMG_GetError());
		exit(1);
	}
}

//###################################################################################################

SDL_Surface * ConvertFrom24to32 (SDL_Surface * source, SDL_PixelFormat *format32)
{
	SDL_Surface *result;
	int i;
	int sourceSize;
	int j;
	unsigned char *sourceTmp, *targetTmp;
	
	result = SDL_CreateRGBSurface(SDL_SWSURFACE, source->w, source->h, 
		Screen->format->BitsPerPixel, source->format->Rmask, source->format->Gmask, 
		source->format->Bmask, source->format->Amask);

	sourceSize = source->h * source->pitch;

	sourceTmp = (unsigned char*) source->pixels;
	targetTmp = (unsigned char*) result->pixels;
	
	i = 0;
	j = 0;
	while (i<sourceSize)
	{
		targetTmp[j+0]	= sourceTmp[i+2];
		targetTmp[j+1]	= sourceTmp[i+1];
		targetTmp[j+2]	= sourceTmp[i+0];
		targetTmp[j+3]	= 0 ;
		i = i + 3;
		j = j + 4;
	}


	return result;
}

//###################################################################################################

void InitCuda()
{
	cudaError_t err; 
	int intTmp;

        err = cudaGetDeviceCount(&intTmp);
        if (err != cudaSuccess) 
	{
                printf("Error getting device count : %d\n", err);
		exit(1);
        }

	cudaGetDevice(&intTmp);
	printf("\nCUDA Device Id = %i\n\n", intTmp);

	cudaDeviceProp prop;
	cudaGetDeviceProperties (&prop, intTmp);

	printf ("##############################################\n");
	printf ("name : %s\n",prop.name);
	printf ("grid size : %d %d %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf ("threads size : %d %d %d\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf ("##############################################\n");

}

//###################################################################################################

