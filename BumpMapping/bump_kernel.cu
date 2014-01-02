
#include <stdio.h>

texture<unsigned char, 2, cudaReadModeElementType> texBumpMap;
unsigned char *deviceBumpMap;

texture<unsigned char, 1, cudaReadModeElementType> texSourceBumpMap;
unsigned char *deviceSourceBumpMap;


//###################################################################################################

__global__ void BumpCuda (unsigned int w, unsigned int h, 
	signed int lx, signed int ly, unsigned char *dest)
{
	unsigned int x, y , i;
	signed int nx,ny;

	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x+2<w) && (y+2<h)) 
	{
		i = (x + y * w) * 4;
	
		nx  = -x + lx + 128;
		nx += tex1Dfetch(texSourceBumpMap,(x+y*w)*4);
		nx -= tex1Dfetch(texSourceBumpMap,(x+1+y*w)*4);

		ny  = -y+ ly + 128;
		ny -= tex1Dfetch(texSourceBumpMap,(x+y*w)*4);
		ny += tex1Dfetch(texSourceBumpMap,(x+(y+1)*w)*4);

	        if (nx>255 || nx<0) nx=0;
	        if (ny>255 || ny<0) ny=0;
	
		dest[i+0] = tex2D(texBumpMap, nx, ny);
		dest[i+1] = tex2D(texBumpMap, nx, ny);
		dest[i+2] = tex2D(texBumpMap, nx, ny);
		dest[i+3] = tex2D(texBumpMap, nx, ny);
	}
}

//###################################################################################################

void Bump32 (
	unsigned int w, unsigned int h, signed int lx, signed int ly,
	unsigned char *dest)
{
	dim3 grid(w / 16, h / 16);
	dim3 threads (16,16);

	BumpCuda<<<grid,threads>>>(w,h,lx,ly,dest);

}

//###################################################################################################

int BumpSourceTextureOn(unsigned char *src,unsigned int w, unsigned int h)
{
	cudaError_t err;
	cudaChannelFormatDesc desc;
	unsigned int totalSize;

	totalSize = w * h * 4;
	

	err = cudaMalloc((void**)&deviceSourceBumpMap,totalSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:72: %s\n",cudaGetErrorString(err));
		return (-1);
	}

	err = cudaMemcpy(deviceSourceBumpMap,src,totalSize,cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:94: %s\n",cudaGetErrorString(err));
		return (-1);
	}

	desc = cudaCreateChannelDesc<unsigned char>();

	err = cudaBindTexture (NULL,texSourceBumpMap,deviceSourceBumpMap,desc,totalSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:94: %s\n",cudaGetErrorString(err));
		return (-1);
	}


	return (0);
}

//###################################################################################################

int BumpSourceTextureOff()
{
	cudaUnbindTexture(texSourceBumpMap);

	return (0);
}

//###################################################################################################

int BumpMapTextureOn (unsigned int bumpSideSize)
{
	cudaError_t err;
	cudaChannelFormatDesc desc;
	unsigned char *tmpBumpMap;
        signed int x,y;
        float nX,nY,nZ;
	unsigned int i;


	err = cudaMalloc((void**)&deviceBumpMap,bumpSideSize * bumpSideSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:66: %s\n",cudaGetErrorString(err));
		return (-1);
	}

	desc = cudaCreateChannelDesc<unsigned char>();

	err = cudaBindTexture2D( NULL, texBumpMap, deviceBumpMap,
		desc, bumpSideSize, bumpSideSize, sizeof(unsigned char) * bumpSideSize);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:77: %s\n",cudaGetErrorString(err));
		cudaFree(deviceBumpMap);
		return (-1);
	}

	tmpBumpMap = (unsigned char*) malloc (bumpSideSize * bumpSideSize);

	if (tmpBumpMap == NULL)
	{
		fprintf(stderr,"Error:84: can not malloc tmpBumpMap\n");
		cudaUnbindTexture(texBumpMap);
		cudaFree(deviceBumpMap);
		return (-1);
	}

	i=0;
        for (y=1;y<=bumpSideSize;y++)
	{
                for (x=1;x<=bumpSideSize;x++)
		{
                        nX=(x-128)/128.0;
                        nY=(y-128)/128.0;

                        nZ=1-sqrt(nX*nX+nY*nY);

                        if (nZ<0) nZ=0;

                        tmpBumpMap[x-1+i]=(unsigned char)(nZ*255); 
		}
		i+=bumpSideSize;
	}

	err = cudaMemcpy(deviceBumpMap,tmpBumpMap, bumpSideSize * bumpSideSize,cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr,"Error:108: %s\n",cudaGetErrorString(err));
		free(tmpBumpMap);
		cudaUnbindTexture(texBumpMap);
		cudaFree(deviceBumpMap);
		return (-1);
	}
	
	free(tmpBumpMap);

	return (0);
}

//###################################################################################################

int BumpMapTextureOff ()
{
	cudaUnbindTexture(texBumpMap);
	cudaFree(deviceBumpMap);

	return (0);
}

//###################################################################################################

