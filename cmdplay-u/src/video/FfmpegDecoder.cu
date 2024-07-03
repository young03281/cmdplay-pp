#include "FfmpegDecoder.cuh"
#include "../FfmpegException.hpp"
#include "../Stopwatch.hpp"
#include <iostream>
#include <vector>

cmdplay::video::FfmpegDecoder::FfmpegDecoder()
{
	std::cout << "[INFO] Initializing video playback" << std::endl;

	std::lock_guard<std::mutex> lg{ m_avLock };
	{
		m_frame = av_frame_alloc();
		m_frameRGB = av_frame_alloc();
		if (m_frame == nullptr || m_frameRGB == nullptr)
			throw FfmpegException("AVFrameAllocation");

		m_packet = av_packet_alloc();
		if (m_packet == nullptr)
			throw FfmpegException("AVPacketAllocException");
	}

	m_workerThread = std::thread(WorkerThread, this);
	bool msg = false;
	while (!m_workerThreadRunning)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		if (!msg)
		{
			msg = true;
			std::cout << "[INFO] Waiting for video worker thread to come online..." << std::endl;
		}
	}
	std::cout << "[INFO] Started video worker thread" << std::endl;
	std::cout << "[INFO] Initialized video playback" << std::endl;
}

cmdplay::video::FfmpegDecoder::~FfmpegDecoder()
{
	std::cout << "[INFO] Freeing video playback..." << std::endl;
	m_workerThreadShouldExit = true;
	cmdplay::Stopwatch timeoutWatch;
	while (m_workerThreadRunning)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (timeoutWatch.GetElapsed() > 4)
		{
			std::cout << "[WARN] Video worker thread failed to exit within timeout" << std::endl;
			break;
		}
	}
	if (!m_workerThreadRunning)
		m_workerThread.join();
	UnloadVideo();
	av_frame_free(&m_frame);
	av_frame_free(&m_frameRGB);
	av_packet_free(&m_packet);

	std::cout << "[INFO] Freed video playback" << std::endl;
}

void cmdplay::video::FfmpegDecoder::LoadVideo(const std::string& src, int width, int height)
{
	m_width = width;
	m_height = height;
	UnloadVideo();
	std::lock_guard<std::mutex> lg{ m_avLock };
	int ret;
	ret = avformat_open_input(&m_formatCtx, src.c_str(), nullptr, nullptr);
	if (ret < 0)
		throw FfmpegException("AVOpenInput");

	ret = avformat_find_stream_info(m_formatCtx, nullptr);
	if (ret < 0)
		throw FfmpegException("AVFindStreamInfo");

	m_containsAudioStream = false;
	for (uint32_t i = 0; i < m_formatCtx->nb_streams; ++i)
		if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && m_videoStreamIndex == -1)
		{
			m_videoStreamIndex = i;
		}
		else if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
			m_containsAudioStream = true;

	if (m_videoStreamIndex == -1)
		throw FfmpegException("NoStreamFound");

	m_codec = avcodec_find_decoder(m_formatCtx->streams[m_videoStreamIndex]->codecpar->codec_id);
	if (m_codec == nullptr)
		throw FfmpegException("AVFindDecoder");

	m_codecCtxOrig = avcodec_alloc_context3(m_codec);
	if (m_codecCtxOrig == nullptr)
		throw FfmpegException("AVAllocContext");

	ret = avcodec_parameters_to_context(m_codecCtxOrig,
		m_formatCtx->streams[m_videoStreamIndex]->codecpar);
	if (ret < 0)
		throw FfmpegException("AVParametersToContext");

	m_codecCtx = avcodec_alloc_context3(m_codec);
	if (m_codecCtx == nullptr)
		throw FfmpegException("AVAllocContext");

	ret = avcodec_parameters_to_context(m_codecCtx,
		m_formatCtx->streams[m_videoStreamIndex]->codecpar);
	if (ret < 0)
		throw FfmpegException("AVParametersToContext2");

	ret = avcodec_open2(m_codecCtx, m_codec, nullptr);
	if (ret < 0)
		throw FfmpegException("AVOpen2");

	int bufferSize = av_image_get_buffer_size(AV_PIX_FMT_RGB0,
		m_codecCtx->width, m_codecCtx->height, 32);
	m_buffer = static_cast<uint8_t*>(av_malloc(bufferSize * sizeof(uint8_t)));
	if (m_buffer == nullptr)
		throw FfmpegException("AVMalloc");

	ret = av_image_fill_arrays(
		m_frameRGB->data,
		m_frameRGB->linesize,
		m_buffer,
		AV_PIX_FMT_RGB0,
		width,
		height,
		32
	);
	if (ret < 0)
		throw FfmpegException("AVFillArrays");

	m_swsCtx = sws_getContext(
		m_codecCtx->width,
		m_codecCtx->height,
		m_codecCtx->pix_fmt,
		width,
		height,
		AV_PIX_FMT_RGB0,
		SWS_BILINEAR,
		nullptr,
		nullptr,
		nullptr
	);
	m_mainThreadFrameBufferVideoWidth = m_codecCtx->width;
	m_mainThreadFrameBufferVideoHeight = m_codecCtx->height;
	m_mainThreadFramebufferSize = m_mainThreadFrameBufferVideoWidth *
		m_mainThreadFrameBufferVideoHeight * 4;
	m_mainThreadFramebuffer =
		static_cast<unsigned char*>(calloc(m_mainThreadFramebufferSize, sizeof(unsigned char)));

	m_videoLoaded = true;
}

void cmdplay::video::FfmpegDecoder::UnloadVideo()
{
	std::lock_guard<std::mutex> avlg{ m_avLock };
	std::lock_guard<std::mutex> fblg{ m_mainThreadFramebufferLock };
	if (m_formatCtx != nullptr)
		avformat_close_input(&m_formatCtx);

	m_videoStreamIndex = -1;

	if (m_codecCtxOrig != nullptr)
		avcodec_free_context(&m_codecCtxOrig);

	if (m_codecCtx != nullptr)
		avcodec_free_context(&m_codecCtx);

	if (m_buffer != nullptr)
	{
		av_free(m_buffer);
		m_buffer = nullptr;
	}
	if (m_mainThreadFramebuffer != nullptr)
	{
		free(m_mainThreadFramebuffer);
		m_mainThreadFramebuffer = nullptr;
	}
	if (m_swsCtx != nullptr)
	{
		sws_freeContext(m_swsCtx);
		m_swsCtx = nullptr;
	}
	m_videoLoaded = false;
}

void cmdplay::video::FfmpegDecoder::LoadVideoAsync(const std::string& src, int width, int height)
{
	m_asyncLoadTask = std::async([&, src]() { LoadVideo(src, width, height); });
}

void cmdplay::video::FfmpegDecoder::UnloadVideoAsync()
{
	m_asyncLoadTask = std::async([&]() { UnloadVideo(); });
}

void cmdplay::video::FfmpegDecoder::SetPlaybackPosition(float currentTime)
{
	m_currentTime = currentTime;
}

bool cmdplay::video::FfmpegDecoder::IsVideoReadyToPlay()
{
	return m_videoLoaded;
}

void cmdplay::video::FfmpegDecoder::ResetPlaybackPosition()
{
	std::lock_guard<std::mutex> lg{ m_avLock };
	m_codecCtx->frame_num = 0;
	av_seek_frame(m_formatCtx, m_videoStreamIndex, 0, 0);
}

cmdplay::video::DecodedFrame* cmdplay::video::FfmpegDecoder::GetNextFrame()
{
	std::lock_guard<std::mutex> fblg{ m_mainThreadFramebufferLock };

	if (m_decodedFrames.size() == 0)
		return nullptr;

	

	auto nextFrame = m_decodedFrames[0];
	m_decodedFrames.erase(m_decodedFrames.begin());
	return nextFrame;
}

int cmdplay::video::FfmpegDecoder::getframessize() {
	return m_decodedFrames.size();
}

void cmdplay::video::FfmpegDecoder::SkipTo(float playbackTime)
{
	std::lock_guard<std::mutex> fblg{ m_mainThreadFramebufferLock };

	while (m_decodedFrames.size() > 1)
	{
		if (playbackTime > m_decodedFrames[0]->m_time &&
			playbackTime > m_decodedFrames[1]->m_time)
		{
			delete m_decodedFrames[0];
			m_decodedFrames.erase(m_decodedFrames.begin());
		}	
		else
			break;
	}
}

void cmdplay::video::FfmpegDecoder::Resize(int width, int height)
{
	std::lock_guard<std::mutex> avlg{ m_avLock };
	if (m_swsCtx != nullptr)
		sws_freeContext(m_swsCtx);

	m_width = width;
	m_height = height;
	m_swsCtx = sws_getContext(
		m_codecCtx->width,
		m_codecCtx->height,
		m_codecCtx->pix_fmt,
		width,
		height,
		AV_PIX_FMT_RGB0,
		SWS_BILINEAR,
		nullptr,
		nullptr,
		nullptr
	);
		
	int ret = av_image_fill_arrays(
		m_frameRGB->data,
		m_frameRGB->linesize,
		m_buffer,
		AV_PIX_FMT_RGB0,
		width,
		height,
		32
	);
	if (ret < 0)
		throw FfmpegException("AVFillArrays");

	std::lock_guard<std::mutex> fblg{ m_mainThreadFramebufferLock };
	for (int i = 0; i < m_decodedFrames.size(); ++i)
		delete m_decodedFrames[i];

	m_decodedFrames.clear();
}

bool cmdplay::video::FfmpegDecoder::ContainsAudioStream()
{
	return m_containsAudioStream;
}

__global__ void cmdplay::video::transferRGB(int line_size, unsigned char * data, unsigned char* src, int h, int w) {
	//int srcpixloc = blockIdx.x * line_size + threadIdx.x * 4;
	int gpuindex = (blockIdx.x * blockDim.x + threadIdx.x);	
	int srcpixloc = gpuindex / w * line_size + gpuindex % w * 4;
	data[gpuindex * 4] = src[srcpixloc];
	data[gpuindex * 4 + 1] = src[srcpixloc + 1];
	data[gpuindex * 4 + 2] = src[srcpixloc + 2];
	data[gpuindex * 4 + 3] = 0;
}

void cmdplay::video::FfmpegDecoder::WorkerThread(FfmpegDecoder* instance)
{
	instance->m_workerThreadRunning = true;
	while (!instance->m_workerThreadShouldExit)
	{
		if (!instance->m_videoLoaded)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			continue;
		}

		std::lock_guard<std::mutex> avlg{ instance->m_avLock };

		auto stream = instance->m_formatCtx->streams[instance->m_videoStreamIndex];
		double fps = static_cast<double>(stream->r_frame_rate.num) /
			static_cast<double>(stream->r_frame_rate.den);

		int64_t shouldFrameIndex = static_cast<int64_t>(
			fps * static_cast<double>(instance->m_currentTime));
		int64_t decodedFrameIndex = instance->m_codecCtx->frame_num;

		if (shouldFrameIndex > decodedFrameIndex)
		{
			// Gets the next frame from the video
			if (av_read_frame(instance->m_formatCtx, instance->m_packet) >= 0)
			{
				if (instance->m_packet->stream_index == instance->m_videoStreamIndex)
				{
					int ret;
					// Requests a frame decode
					ret = avcodec_send_packet(instance->m_codecCtx, instance->m_packet);
					if (ret < 0)
					{
						std::cout << "[ERROR] avcodec send packet";
						std::this_thread::sleep_for(std::chrono::milliseconds(250));
						continue;
					}
					while (ret >= 0)
					{
						// Receives the decoded frame
						ret = avcodec_receive_frame(instance->m_codecCtx, instance->m_frame);
						if (ret == AVERROR(EAGAIN)) // Frame is not decoded, try again
							continue;
						else if (ret == AVERROR_EOF)
							break;
						else if (ret < 0)
						{
							std::cout << "[ERROR] avcodec receive frame" << std::endl;
							std::this_thread::sleep_for(std::chrono::milliseconds(250));
							break;
						}
						// Scales the frame to the correct resolution
						sws_scale(instance->m_swsCtx,
							instance->m_frame->data,
							instance->m_frame->linesize,
							0,
							instance->m_codecCtx->height,
							instance->m_frameRGB->data,
							instance->m_frameRGB->linesize
						);

						// Copies the frame data to the buffer
						std::lock_guard<std::mutex> fblg{ instance->m_mainThreadFramebufferLock };
						unsigned char* buffer = instance->m_mainThreadFramebuffer;
						DecodedFrame *newFrame = new DecodedFrame(instance->m_width * instance->m_height * 4,
							instance->m_frame->pts * static_cast<float>(
								av_q2d(instance->m_formatCtx->streams[instance->m_videoStreamIndex]->time_base)));

						
						
						int h = instance->m_height;
						int w = instance->m_width;
						int linesize = instance->m_frameRGB->linesize[0];
						int le = newFrame->m_dataLength;
						int sizedata, sizesrc;

						sizedata = sizeof(unsigned char) * le;
						sizesrc = linesize * h;

						//= instance->m_frameRGB->data[0] if theres with a

						
						uint8_t* src, *d_src;
						cudaMalloc((void**)&d_src, sizesrc);
						cudaMemcpy(d_src, instance->m_frameRGB->data[0], sizesrc, cudaMemcpyHostToDevice);


						uint8_t* d_data;
						if (cudaMalloc((void**)&(newFrame->m_data), sizedata) != ::cudaSuccess)
							throw FfmpegException("cuda_malloc failure d_data");

						transferRGB <<<h*w/1024 + 1, 1024 >>> (linesize, newFrame->m_data, d_src, h, w);

						instance->m_decodedFrames.push_back(newFrame);
						cudaFree(d_src);
						

					}
				}
				av_packet_unref(instance->m_packet);
			}
		}
	}
	instance->m_workerThreadRunning = false;
}
