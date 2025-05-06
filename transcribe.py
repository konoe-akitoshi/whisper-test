import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from speaker_diarization import diarize_audio
import openai
import os
from dotenv import load_dotenv
import subprocess
import math
import tempfile
import argparse

def extract_segments(audio_path, segments, tmpdir):
    extracted = []
    for i, seg in enumerate(segments):
        start = seg["segment"]["start"]
        end = seg["segment"]["end"]
        label = seg["label"]
        duration = end - start
        if duration < 0.1:
            continue
        out_file = os.path.join(tmpdir, f"segment_{i+1}_{label}.m4a")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(duration),
            "-c", "copy", out_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        extracted.append((label, out_file))
    return extracted

MAX_SIZE = 25 * 1024 * 1024  # 25MB

def split_audio_by_size(input_path, tmpdir, max_size=MAX_SIZE):
    file_size = os.path.getsize(input_path)
    if file_size <= max_size:
        return [input_path]

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    duration = float(result.stdout.strip())
    num_parts = math.ceil(file_size / max_size)
    part_duration = duration / num_parts

    split_files = []
    for i in range(num_parts):
        start = i * part_duration
        output_file = os.path.join(tmpdir, f"part_{i+1}.m4a")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-t", str(part_duration),
            "-c", "copy", output_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        split_files.append(output_file)
    return split_files

def transcribe_audio_with_api(audio_path, tmpdir):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY環境変数が設定されていません。")
    openai.api_key = api_key

    audio_files = split_audio_by_size(audio_path, tmpdir)
    texts = []
    total = len(audio_files)
    for idx, file in enumerate(audio_files, 1):
        print(f"[{idx}/{total}] {os.path.basename(file)} を文字起こし中...")
        with open(file, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="ja"
            )
            texts.append(transcript)
        print(f"[{idx}/{total}] {os.path.basename(file)} 完了")
        if file != audio_path:
            os.remove(file)
    return "\n".join(texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="話者分離＋Whisper文字起こし")
    parser.add_argument("audio_file", help="入力音声ファイル（m4a/wavなど）")
    parser.add_argument("--min_duration_off", type=float, default=0.1, help="話者切替の最小無音区間（秒, デフォルト: 0.1）")
    parser.add_argument("--threshold", type=float, default=0.715, help="クラスタリングの閾値（デフォルト: 0.715）")
    args = parser.parse_args()
    audio_file = args.audio_file

    load_dotenv()

    txt_file = os.path.splitext(audio_file)[0] + ".txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_file = os.path.join(tmpdir, os.path.splitext(os.path.basename(audio_file))[0] + "_tmp.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_file, "-ar", "16000", "-ac", "1", wav_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("話者分離を実行中...")
        segments = diarize_audio(
            wav_file,
            min_duration_off=args.min_duration_off,
            threshold=args.threshold
        )
        print(f"話者区間数: {len(segments)}")
        extracted = extract_segments(audio_file, segments, tmpdir)
        with open(txt_file, "w", encoding="utf-8") as f:
            for idx, (label, seg_file) in enumerate(extracted, 1):
                print(f"[{idx}/{len(extracted)}] {label} 区間を文字起こし中...")
                text = transcribe_audio_with_api(seg_file, tmpdir)
                f.write(f"{label}: {text.strip()}\n")
                print(f"[{idx}/{len(extracted)}] {label} 完了")
    print(f"話者ラベル付き文字起こし結果を {txt_file} に保存しました。")