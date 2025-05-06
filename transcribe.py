from speaker_diarization import diarize_audio
import openai
import os
from dotenv import load_dotenv
import subprocess
import math

def extract_segments(audio_path, segments):
    """
    区間情報リストをもとに、各話者区間ごとに音声ファイルを切り出す
    戻り値: [(label, segment_file_path), ...]
    """
    extracted = []
    for i, seg in enumerate(segments):
        start = seg["segment"]["start"]
        end = seg["segment"]["end"]
        label = seg["label"]
        duration = end - start
        if duration < 0.5:
            # Whisper APIが受け付けない短すぎる区間はスキップ
            continue
        out_file = f"segment_{i+1}_{label}.m4a"
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path, "-ss", str(start), "-t", str(duration),
            "-c", "copy", out_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        extracted.append((label, out_file))
    return extracted

MAX_SIZE = 25 * 1024 * 1024  # 25MB

def split_audio_by_size(input_path, max_size=MAX_SIZE):
    file_size = os.path.getsize(input_path)
    if file_size <= max_size:
        return [input_path]

    # 音声の全体長（秒）を取得
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    duration = float(result.stdout.strip())
    # 分割数を計算
    num_parts = math.ceil(file_size / max_size)
    part_duration = duration / num_parts

    split_files = []
    for i in range(num_parts):
        start = i * part_duration
        output_file = f"part_{i+1}.m4a"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-ss", str(start), "-t", str(part_duration),
            "-c", "copy", output_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        split_files.append(output_file)
    return split_files

def transcribe_audio_with_api(audio_path):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY環境変数が設定されていません。")
    openai.api_key = api_key

    # ファイルサイズが大きい場合は分割
    audio_files = split_audio_by_size(audio_path)
    texts = []
    total = len(audio_files)
    for idx, file in enumerate(audio_files, 1):
        print(f"[{idx}/{total}] {file} を文字起こし中...")
        with open(file, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="ja"
            )
            texts.append(transcript)
        print(f"[{idx}/{total}] {file} 完了")
        # 分割ファイルは削除（元ファイルは残す）
        if file != audio_path:
            os.remove(file)
    return "\n".join(texts)
# m4a→wav変換（pyannote.audioはwavのみ対応）
if __name__ == "__main__":
    load_dotenv()
    audio_file = "test.m4a"
    # m4a→wav変換（pyannote.audioはwavのみ対応）
    wav_file = os.path.splitext(audio_file)[0] + "_tmp.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file, "-ar", "16000", "-ac", "1", wav_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 1. 話者分離（wavで実行）
    print("話者分離を実行中...")
    segments = diarize_audio(wav_file)
    print(f"話者区間数: {len(segments)}")
    # 2. 区間ごとに音声切り出し（元のm4aから）
    extracted = extract_segments(audio_file, segments)
    # 3. 各区間をWhisperで文字起こし
    txt_file = os.path.splitext(audio_file)[0] + ".txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for idx, (label, seg_file) in enumerate(extracted, 1):
            print(f"[{idx}/{len(extracted)}] {label} 区間を文字起こし中...")
            text = transcribe_audio_with_api(seg_file)
            f.write(f"{label}: {text.strip()}\n")
            print(f"[{idx}/{len(extracted)}] {label} 完了")
            if seg_file != audio_file:
                os.remove(seg_file)
    # 一時wavファイルを削除
    if os.path.exists(wav_file):
        os.remove(wav_file)
    print(f"話者ラベル付き文字起こし結果を {txt_file} に保存しました。")