import os
from dotenv import load_dotenv

def diarize_audio(audio_path):
    """
    pyannote.audioのローカル推論で話者分離を行い、
    区間情報（話者ラベル・開始・終了）を返す
    """
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN環境変数が設定されていません。")

    from pyannote.audio import Pipeline
    import torch

    # パイプラインのロード
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    # GPUがあれば使う
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # 推論実行
    diarization = pipeline(audio_path)

    # 区間情報をリストで返す
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "segment": {"start": float(turn.start), "end": float(turn.end)},
            "label": f"{speaker}"
        })
    return segments