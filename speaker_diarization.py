import os
from dotenv import load_dotenv

def diarize_audio(audio_path, min_duration_off=0.1, threshold=0.715):
    """
    pyannote.audioのローカル推論で話者分離を行い、
    区間情報（話者ラベル・開始・終了）を返す
    min_duration_off: 話者切替の最小無音区間（秒）
    threshold: クラスタリングの閾値（話者分離の厳しさ）
    """
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN環境変数が設定されていません。")

    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    pipeline.instantiate({
        "segmentation": {"min_duration_off": min_duration_off},
        "clustering": {"threshold": threshold}
    })

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "segment": {"start": float(turn.start), "end": float(turn.end)},
            "label": f"SPEAKER_{speaker}"
        })
    return segments