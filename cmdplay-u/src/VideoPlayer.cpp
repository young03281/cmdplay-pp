#include "VideoPlayer.hpp"

#include "ConsoleUtils.hpp"
#include "MiniAudioException.hpp"
#include "Stopwatch.hpp"
#include <conio.h>
#include <iostream>
#include <ctime>

cmdplay::VideoPlayer::VideoPlayer(const std::string& filePath, const std::string& brightnessLevels) :
	m_filePath(filePath), m_brightnessLevels(brightnessLevels)
{
	ConsoleUtils::GetWindowSize(&m_windowWidth, &m_windowHeight);
	InitAsciifier();
	m_decoder = std::make_unique<video::FfmpegDecoder>();
}

void cmdplay::VideoPlayer::InitAsciifier()
{
	m_asciifier = std::make_unique<Asciifier>(m_brightnessLevels,
		m_windowWidth, m_windowHeight, m_colorsEnabled,
		m_colorDitheringEnabled, m_textDitheringEnabled, m_accurateColorsEnabled, m_accurateColorsFullPixelEnabled);

	d_asciifier = std::make_unique<gpuAsciiFier>(m_brightnessLevels,
		m_windowWidth, m_windowHeight, m_colorsEnabled,
		m_colorDitheringEnabled, m_textDitheringEnabled, m_accurateColorsEnabled, m_accurateColorsFullPixelEnabled);
}

void cmdplay::VideoPlayer::LoadVideo()
{
	m_decoder->LoadVideo(m_filePath, m_windowWidth, m_windowHeight);
	try
	{
		if (m_decoder->ContainsAudioStream()) {
			m_audioSource = std::make_unique<audio::FfmpegAudio>(m_filePath);
			m_audioSource->DecodeAll();
		}
		else
			throw MiniAudioException("");
	}
	catch (...) { m_audioSource = nullptr; }
}

void cmdplay::VideoPlayer::Enter()
{
	if (m_audioSource != nullptr)
		m_audioSource->PlayASync();
	float syncTime = 0.0f;
	bool playing = true;
	Stopwatch syncWatch;

	while (true)
	{
		// for some reason, windows enables the cursor every time we resize, so we should set it each time
		ConsoleUtils::ShowConsoleCursor(false);

		//std::cout << m_decoder->getframessize() << std::endl;

		if (_kbhit())
		{
			char c = static_cast<char>(_getch());
			if (c >= 'A' && c <= 'Z')
			{
				c += 32;
			}
			if (c == 'q')
			{
				break;
			}

			switch (c)
			{
			case ' ':
			{
				playing = !playing;
				if (playing) {
					syncWatch.Resume();
					if (m_audioSource != nullptr)
						m_audioSource->Resume();
				}
				else {
					syncWatch.Pause();
					if (m_audioSource != nullptr)
						m_audioSource->Pause();
				}
				break;
			}
			case 'c':
			{
				m_colorsEnabled = !m_colorsEnabled;
				InitAsciifier();
				if (!m_colorsEnabled)
				{
					// Reset colors
					std::cout << "\x1B[0m";
				}
				break;
			}
			case 'a':
			{
				m_accurateColorsEnabled = !m_accurateColorsEnabled;
				InitAsciifier();
				if (!m_accurateColorsEnabled)
				{
					// Reset colors
					std::cout << "\x1B[0m";
				}
				break;
			}
			case 'b':
			{
				m_accurateColorsFullPixelEnabled = !m_accurateColorsFullPixelEnabled;
				InitAsciifier();
				break;
			}
			case 't':
			{
				m_textDitheringEnabled = !m_textDitheringEnabled;
				InitAsciifier();
				break;
			}
			case 'd':
			{
				m_colorDitheringEnabled = !m_colorDitheringEnabled;
				InitAsciifier();
				break;
			}
			default: 
				break;
			}

		}

		if (m_audioSource != nullptr)
			syncTime = m_audioSource->GetPlaybackTime();
		else
			syncTime = syncWatch.GetElapsed();
		


		if (!playing)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}

		int newWidth = 0, newHeight = 0;
		ConsoleUtils::GetWindowSize(&newWidth, &newHeight);

		// resize everything if window size has been changed
		if (newWidth != m_windowWidth || newHeight != m_windowHeight)
		{
			m_windowWidth = newWidth;
			m_windowHeight = newHeight;
			m_decoder->Resize(m_windowWidth, m_windowHeight);
			InitAsciifier();
		}

		m_decoder->SetPlaybackPosition(syncTime + PREBUFFER_TIME);
		// skip to the newest already decoded frame which can be shown
		m_decoder->SkipTo(syncTime);
		auto nextFrame = m_decoder->GetNextFrame();
		if (nextFrame == nullptr)
			continue;

		// wait until our frame can be played back
		while (nextFrame->m_time > syncTime)
		{
			if (m_audioSource == nullptr)
				syncTime = syncWatch.GetElapsed();
			else
				syncTime = m_audioSource->GetPlaybackTime();
		}
		cmdplay::ConsoleUtils::SetCursorPosition(0, 0);
		nextFrame->start = std::clock();
		char* frame = d_asciifier->BuildFrame(nextFrame->m_data);
		std::cout << frame;
		std::cout << std::endl << double(std::clock() - nextFrame->start + nextFrame->t1 + nextFrame->t2) / (double)CLOCKS_PER_SEC << "   " << double(std::clock() - nextFrame->start) / (double)CLOCKS_PER_SEC << "      " << double(nextFrame->t1) / (double)CLOCKS_PER_SEC << "     " << double(nextFrame->t2) / (double)CLOCKS_PER_SEC << "     " << double(nextFrame->t1) / (double)CLOCKS_PER_SEC << "    " << (double)1 / 120;

		cudaFreeHost(frame);
		nextFrame->~DecodedFrame();
	}
	ConsoleUtils::ShowConsoleCursor(true);
}
