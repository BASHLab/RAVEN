# Audio, Video, Sensor Question Answer (**AVS-QA**)

**AVS-QA** dataset follows this structure
```json
[
    {
        "file_name_root": <filename root>,
        "source": <source> # EPIC-KITCHEN or EGO4D,
        "conversation": [
            {
                "question": <question>,
                "answer": <answer>,
                "question_type": <question type>
            }
            ...
        ]
    }
    ...
]
```
To download corresponding 📷 video, 🎤 audio, and 📝 IMU:
- [EPIC-KITCHEN](https://epic-kitchens.github.io/2025)
- [EGO4D](https://ego4d-data.org/docs/start-here/)
  - Natural Language Query
  - Moments Query

Download the data and arrange them in following format
Dataset: [BASH-Lab/AVS-QA](BASH-Lab/AVS-QA)
```bash
RAVEN
├── datasets
│   ├── custom_sft
│   |   ├── sensor
│   |   ├── videos
│   │   |   ├── EGO4D
│   │   |   ├── EPIC-KITCHEN

```
