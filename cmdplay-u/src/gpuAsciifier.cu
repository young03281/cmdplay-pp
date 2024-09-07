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
	m_framechars = (char*)malloc(sizeof(char) * m_framebuffersize);
	m_brightnessLevelCount = strlen(m_brightnessLevels);
	cudaMalloc((void**)&d_framechars, m_framebuffersize);
	cudaMallocManaged((void**)&d_brightnessLevels, m_brightnessLevelCount);
	for (int i = 0; i < m_brightnessLevelCount; i++) {
		memset(d_brightnessLevels + i, m_brightnessLevels[i], 1);
	};

}

int cmdplay::gpuAsciiFier::getBufferSize() {
	return m_framebuffersize;
}

char* cmdplay::gpuAsciiFier::BuildFrame(uint8_t * d_rgbData) {

	

	asciifier<< <m_frameWidth * m_frameHeight /512 + 1, 512>> > (d_rgbData, d_framechars,d_brightnessLevels, m_frameWidth , m_frameHeight, m_brightnessLevelCount);

	cudaMemcpy(m_framechars, d_framechars, m_framebuffersize, cudaMemcpyDeviceToHost);

	for (int i = 1; i < m_frameHeight ; ++i) {
		m_framechars[(m_frameWidthWithStride + 1) * i - 1] = '\n';

	}
	m_framechars[m_framebuffersize] = '\0';

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

		int brightnessindex = (0.299 * r + 0.587 * g + 0.114 * b) * 12;

		if (brightnessindex < 0) {
			brightnessindex = 0;
		}if (brightnessindex >= 13) {
			brightnessindex = 13 - 1;
		}
		framechars[frameindex] = brightnesslevel[brightnessindex];
	}
}