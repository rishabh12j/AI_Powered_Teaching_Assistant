"""AI teaching assistant: turns a lecture (audio file or YouTube link) into study notes and a quiz.

Speech-to-text runs on Groq Whisper; two CrewAI agents (a note-taker and a quiz-maker) write the
notes and the multiple-choice quiz on a Groq-hosted LLM.

Secrets (Space settings): GROQ_API_KEY (the older name Token also works).
Optional variable GROQ_MODEL overrides the default text model.
"""

import os
import re
from pathlib import Path

import gradio as gr
from crewai import LLM, Agent, Crew, Process, Task
from groq import Groq

GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("Token")
TEXT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
MAX_AUDIO_MB = 25
MAX_TRANSCRIPT_CHARS = 24_000  # keeps requests inside free-tier token limits


def youtube_id(url: str) -> str | None:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url or "")
    return match.group(1) if match else None


def embed_video(url: str) -> str:
    vid = youtube_id(url)
    if not vid:
        return ""
    return (
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{vid}" '
        'frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; '
        'picture-in-picture" allowfullscreen></iframe>'
    )


def transcribe_audio(path: str) -> str:
    size_mb = Path(path).stat().st_size / 1_000_000
    if size_mb > MAX_AUDIO_MB:
        raise ValueError(f"The audio file is {size_mb:.0f} MB; the limit here is {MAX_AUDIO_MB} MB.")
    client = Groq(api_key=GROQ_KEY)
    with open(path, "rb") as fh:
        result = client.audio.transcriptions.create(file=(Path(path).name, fh.read()), model=STT_MODEL)
    return result.text


def youtube_transcript(url: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    vid = youtube_id(url)
    if not vid:
        raise ValueError("That does not look like a YouTube link.")
    try:
        fetched = YouTubeTranscriptApi().fetch(vid, languages=["en", "en-GB", "en-US", "hi"])
    except Exception as exc:
        raise ValueError(
            "YouTube would not share captions for this video with a cloud server "
            f"({type(exc).__name__}). Download the audio and upload it instead."
        ) from exc
    return " ".join(snippet.text for snippet in fetched)


def teaching_crew(transcript: str) -> Crew:
    llm = LLM(model=f"groq/{TEXT_MODEL}", api_key=GROQ_KEY, temperature=0.3)
    note_taker = Agent(
        role="Lecture note-taker",
        goal="Write complete, well-organised study notes that let a student learn the lecture without listening to it",
        backstory="An experienced teaching assistant who turns spoken lectures into clear notes with headings.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    quiz_maker = Agent(
        role="Quiz writer",
        goal="Write multiple-choice questions that check understanding of every key point in the lecture",
        backstory="A teacher who writes fair, unambiguous multiple-choice questions with one correct answer.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    notes = Task(
        description=(
            "Write detailed study notes from this lecture transcript. Use Markdown headings and bullet points, "
            "cover every topic in order, and do not include timestamps.\n\nTRANSCRIPT:\n" + transcript
        ),
        expected_output="Markdown study notes with headings that cover the whole lecture.",
        agent=note_taker,
    )
    quiz = Task(
        description=(
            "Using the lecture transcript and the notes, write 8 to 10 multiple-choice questions. Each question "
            "has options A to D, exactly one correct answer, and a one-line explanation. Put the answers after "
            "the questions under a heading 'Answers'.\n\nTRANSCRIPT:\n" + transcript
        ),
        expected_output="A Markdown multiple-choice quiz followed by an answer key with explanations.",
        agent=quiz_maker,
        context=[notes],
    )
    return Crew(agents=[note_taker, quiz_maker], tasks=[notes, quiz], process=Process.sequential, verbose=False)


def run(audio_path, youtube_url):
    if not GROQ_KEY:
        msg = "This Space is missing its Groq key. The owner needs to add it as a secret named `GROQ_API_KEY`."
        return msg, "", ""
    try:
        if audio_path:
            transcript = transcribe_audio(audio_path)
        elif (youtube_url or "").strip():
            transcript = youtube_transcript(youtube_url)
        else:
            return "Upload an audio lesson or paste a YouTube link first.", "", ""
    except Exception as exc:
        return f"Could not get a transcript: {exc}", "", ""

    note = ""
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]
        note = f"\n\n_Only the first {MAX_TRANSCRIPT_CHARS:,} characters of the transcript were used._"
    try:
        crew = teaching_crew(transcript)
        crew.kickoff()
        notes_md = crew.tasks[0].output.raw + note
        quiz_md = crew.tasks[1].output.raw
    except Exception as exc:
        return f"The notes could not be generated right now ({type(exc).__name__}: {exc}).", "", transcript
    return notes_md, quiz_md, transcript


with gr.Blocks(title="AI-Powered Teaching Assistant") as demo:
    gr.Markdown(
        "# 🎓 AI-Powered Teaching Assistant\n"
        "Upload a recorded lesson or paste a YouTube link. Whisper transcribes it, then two CrewAI agents "
        "write study notes and a multiple-choice quiz."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="Audio lesson (up to 25 MB)", type="filepath", sources=["upload"])
            url_in = gr.Textbox(label="…or a YouTube link", placeholder="https://www.youtube.com/watch?v=…")
            go = gr.Button("Make notes and quiz", variant="primary")
        video = gr.HTML()
    with gr.Tab("Notes"):
        notes_out = gr.Markdown()
    with gr.Tab("Quiz"):
        quiz_out = gr.Markdown()
    with gr.Tab("Transcript"):
        transcript_out = gr.Textbox(lines=12, show_label=False)

    url_in.change(embed_video, url_in, video)
    go.click(run, [audio_in, url_in], [notes_out, quiz_out, transcript_out], api_name="run")

if __name__ == "__main__":
    demo.launch()
