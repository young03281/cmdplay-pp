#include "gpuAsciifier.cuh"
#include <math.h>


cmdplay::gpuAsciiFier::gpuAsciiFier(const std::string& brightnessLevels, int frameWidth, int frameHeight,
	bool useColors, bool useColorDithering, bool useTextDithering, bool useAccurateColors, bool useAccurateColorsFullPixel):
	m_brightnessLevels(const_cast<char*>(brightnessLevels.c_str())), m_frameWidth(frameWidth), m_frameHeight(frameHeight),
	m_useColorDithering(useColorDithering), m_useTextDithering(useTextDithering), m_useColors(useColors),
	m_useAccurateColors(useAccurateColors), m_useAccurateColorsFullPixel(useAccurateColorsFullPixel)
{
	m_framepixelbytescount = m_frameWidth * m_frameHeight * 4;
	m_framebuffersize = (m_frameWidth + 1) * m_frameHeight;
	m_frameWidthWithStride = m_frameWidth;
	m_brightnessLevelCount = strlen(m_brightnessLevels);
	cudaHostAlloc((void**)&m_framechars, m_framebuffersize, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&d_framechars, (void*)m_framechars, 0);
	cudaMallocManaged((void**)&d_brightnessLevels, m_brightnessLevelCount);
	cudaMemcpy(d_brightnessLevels, m_brightnessLevels, m_brightnessLevelCount, cudaMemcpyHostToDevice);

}

int cmdplay::gpuAsciiFier::getBufferSize() {
	return m_framebuffersize;
}

char* cmdplay::gpuAsciiFier::BuildFrame(uint8_t * d_rgbData) {

	asciifier << <m_frameWidth * m_frameHeight / 512 + 1, 512 >> > (d_rgbData, d_framechars, d_brightnessLevels, m_frameWidth, m_frameHeight, m_brightnessLevelCount);

	memset(m_framechars + m_framebuffersize, '\0', 1);

	return m_framechars;
}

__global__ void asciifier(uint8_t* rgbData, char* framechars, char* brightnesslevel, int width, int height, int brightnesslevelcount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int byteindex = index * 4;
	int check = index / width;
	int frameindex = index + check;
	if (index < width * height) {
		float r, g, b;
		r = (float)(int)rgbData[byteindex] / 255;
		g = (float)(int)rgbData[byteindex + 1] / 255;
		b = (float)(int)rgbData[byteindex + 2] / 255;

		int brightnessindex = (0.299 * r + 0.587 * g + 0.114 * b) * brightnesslevelcount;

		if (brightnessindex < 0) {
			brightnessindex = 0;
		}if (brightnessindex >= brightnesslevelcount) {
			brightnessindex = brightnesslevelcount - 1;
		}
		framechars[frameindex] = brightnesslevel[brightnessindex];
		if ((index + 1) % (width + 1) == 0) {
			framechars[index] = '\n';
		}
	}
}