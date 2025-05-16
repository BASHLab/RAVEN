<p align="center">
    <img src="./assets/raven_logo.png" width="100" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center">
    <a href="" style="color:#825987">
        RAVEN: Query-Guided Representation Alignment for Question
        Answering over Audio, Video, Embedded Sensors, and Natural Language
    </a>
</h3>
<img src="./assets/raven_architecture.png" width="800" />

---
## 🛠️ Requirements and Installation
Basic Dependencies:
* Python >= 3.8
* Pytorch >= 2.2.0
* CUDA Version >= 11.8
* transformers == 4.40.0 (for reproducing paper results)
* tokenizers == 0.19.1

```bash
cd RAVEN
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
pip install opencv-python==4.5.5.64
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```
---
## 👍 Acknowledgement
The codebase of RAVEN is adapted from [**VideoLLaMA2**](https://github.com/DAMO-NLP-SG/VideoLLaMA2) and [**FastChat**](https://github.com/lm-sys/FastChat). We are also grateful for the following projects our VideoLLaMA 2 arise from:
* [**LLaMA 2**](https://github.com/meta-llama/llama), [**Mistral-7B**](https://mistral.ai/news/announcing-mistral-7b/), [**OpenAI CLIP**](https://openai.com/index/clip/), [**Honeybee**](https://github.com/kakaobrain/honeybee).
* [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT), [**Video-LLaVA**](https://github.com/PKU-YuanGroup/Video-LLaVA). 
* [**WebVid**](https://github.com/m-bain/webvid), [**Panda-70M**](https://github.com/snap-research/Panda-70M), [**LanguageBind**](https://github.com/PKU-YuanGroup/LanguageBind), [**InternVid**](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid).
* [**VideoChat2**](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [**Valley**](https://github.com/RupertLuo/Valley), [**VTimeLLM**](https://github.com/huangb23/VTimeLLM), [**ShareGPT4V**](https://sharegpt4v.github.io/).
