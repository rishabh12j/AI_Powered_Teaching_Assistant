# GeekNavi: AI-Powered Teaching Assistant

Upload a recorded lecture (or paste a YouTube link) and GeekNavi writes detailed study notes plus a multiple-choice quiz with an answer key, so you can test what you learnt.

**Live demo:** https://huggingface.co/spaces/Rishabh12j/AI_Powered_Teaching_Assistant

## Versions
- **v0.1 (2024), Team Gen4:** Whisper on Baseten for transcription, Gemini 1.5 for notes, a LLaMA 3 agent on Groq for quizzes, Backblaze B2 for file storage. See [v0.1 architecture](#v01-architecture-2024) below.
- **v0.2 (2026), rebuilt on Groq by Rishabh Jain:** one provider for everything, a two-agent CrewAI pipeline, YouTube captions instead of audio downloads, and Gradio 6. This is the code in this repo and on the live demo.

## How it works (v0.2)
```
audio / YouTube captions ──► Whisper (Groq) ──► transcript
                                                 │
                        ┌────────────────────────┴───────────────────────┐
                        ▼                                                ▼
          Agent 1: note-taker (CrewAI)  ──notes as context──►  Agent 2: quiz writer (CrewAI)
                        │                                                │
                   study notes                             8–10 MCQs + answer key
```
- Speech-to-text runs on Groq Whisper; both agents run on a Groq-hosted LLM.
- The quiz writer receives the note-taker's output as context, so questions follow the structure of the notes.

## Run locally
Needs Python 3.10 or newer and a free [Groq API key](https://console.groq.com/keys).
```bash
git clone https://github.com/rishabh12j/ai_powered_teaching_assistant.git
cd ai_powered_teaching_assistant
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_key
python app.py
```
Then open the link printed in the terminal (default http://127.0.0.1:7860).

## Use it
1. Upload an audio file of a lecture (up to 25 MB), or paste a YouTube link.
2. Click **Make notes and quiz**.
3. Read the **Notes**, **Quiz** and **Transcript** tabs.

Processing time depends on the length of the lecture.

## Configuration
| Setting | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Text model for both agents |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model |

## Limits
- Audio files up to 25 MB; very long transcripts are trimmed to the first 24,000 characters.
- YouTube often refuses to share captions with cloud servers. If a link fails, upload the audio instead.

## Future plans
Carried over from the v0.1 roadmap:
1. Ask questions about the course material and get answers grounded in the lecture.
2. Generate diagrams of a course or module (flowcharts and sequences).
3. Upload chat logs and slide decks as extra input.
4. Add references for more comprehensive notes.

## v0.1 architecture (2024)
The original GeekNavi combined Google Gemini, OpenAI/Groq, Baseten and Backblaze B2 to transcribe audio, generate notes and create quiz questions.

![GeekNavi v0.1 architecture](https://github.com/attrib07/Geeknavi-/assets/44226488/23d1f2e2-96a8-45c1-8f84-18be9c0ee4e1)

The v0.1 code is in this repo's commit history.

## Team Gen4 (v0.1)
| Name | Discord | Email |
|---|---|---|
| Rishabh Jain | b0rn2clutch | rishabh12j@gmail.com |
| Sai Amith Chillamcherla | amith98 | amith.ch@outlook.com |
| Aman Hanspal | hanspaa2017108 | amanhanspal05@gmail.com |
| Vicky Kumar | resetvicky01 | vickykumar07vikram@gmail.om |

---

**v0.2 author:** Rishabh Jain, AI engineer (MSc Robotics and Embedded AI, Maynooth University)  
[More demos](https://huggingface.co/Rishabh12j) · [GitHub](https://github.com/rishabh12j) · [Portfolio](https://rishabh12j.github.io)
