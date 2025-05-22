import ffmpeg
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def video_to_audio(input_path: str, output_path: str):
    """
    Convert any media to mono Opus @16 kHz, 32 kbps, VoIP profile—
    optimal for GPT-4o-mini-transcribe.

    Args:
        input_path (str): Path to the source video.
        output_path (str): Path to save the resulting MP3.

    Returns:
        str: "Converted" on success or the ffmpeg.Error on failure.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vn=None,  # disable video
                map_metadata=-1,  # drop all metadata
                ac=1,  # mono
                ar=16000,  # 16 kHz
                acodec='libopus',  # Opus codec
                audio_bitrate='32k',  # 32 kbps
                application='voip'  # VoIP profile for speech
            )
            .overwrite_output()  # перезаписать, если файл есть
            .run()  # для тишины лога убрать quiet=True
        )

        return "Converted"
    except ffmpeg.Error as e:
        return e


def audio_to_text(input_path: str):
    """
    Конвертирует аудиофайл в русский текст с помощью модели Whisper.

    Открывает указанный аудиофайл в бинарном режиме, отправляет его на
    транскрибацию через OpenAI API и возвращает результат в виде строки.

    Args:
        input_path (str): Путь к аудиофайлу (поддерживаемые форматы: mp3, wav, ogg и др.).

    Returns:
        str: Транскрибированный текст на русском языке.

    Raises:
        FileNotFoundError: Если файл по пути `input_path` не найден.
        openai.error.OpenAIError: Если произошла ошибка при обращении к API.
    """
    audio_file = open(input_path, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        response_format="text",
        language="ru",
        file=audio_file,
    )

    return transcription


def make_meet_report(transcription: str):
    """
    Generate a 5–7 sentence summary paragraph from a full meeting transcription.

    Sends the provided text to OpenAI’s GPT-4o-mini with a prompt that extracts key ideas,
    conclusions, and causal links—omitting filler and side details. Returns a single
    coherent paragraph that conveys the essence of the discussion without needing the
    original transcript.

    Args:
        transcription (str): Full meeting transcript.

    Returns:
        str: A concise summary paragraph.
    """

    prompt = """
    Ты — эксперт-лингвист, специализирующийся на высокоточном сжатии информации.
    Прочитай следующий текст и создай краткий пересказ,    
    который: фиксирует ключевые идеи, выводы и причинно-следственные связи;
    используй списки там где это будет уместно;
    позволяет читателю понять суть обсуждения без обращения к оригиналу;
    без второстепенных деталей; исключай «воду»;    
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "user", "content": transcription},
            {"role": "system", "content": prompt},
        ]
    )

    return completion.choices[0].message