import os
import re
import soundfile as sf
from pydub import AudioSegment
from funasr import AutoModel
import gradio as gr

# 模型路径
model_dir = "iic/SenseVoiceSmall"
vad_model_dir = "fsmn-vad"  # VAD 模型路径

# 加载 VAD 模型
vad_model = AutoModel(
    model=vad_model_dir,
    trust_remote_code=True,
    device="cuda:0",
    disable_update=True
)

# 加载 SenseVoice 模型
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    device="cuda:0",
    disable_update=True
)

# 定义要移除的模式
pattern = r'<\|.*?\|>'

def convert_to_mp3(file_path):
    base_name, ext = os.path.splitext(file_path)
    mp3_file = base_name + ".mp3"

    if ext == ".mp3":
        return file_path

    audio = AudioSegment.from_file(file_path)
    audio.export(mp3_file, format="mp3")
    return mp3_file

def process_audio(file_path, display_pure_text=False):
    try:
        mp3_path = convert_to_mp3(file_path)
        vad_res = vad_model.generate(
            input=mp3_path,
            cache={},
            max_single_segment_time=30000,  # 最大单个片段时长
        )
        segments = vad_res[0]['value']

        audio_data, sample_rate = sf.read(mp3_path)
        results = []

        for segment in segments:
            start_time, end_time = segment
            cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)
            temp_audio_file = "temp_cropped.wav"
            sf.write(temp_audio_file, cropped_audio, sample_rate)

            res = model.generate(
                input=temp_audio_file,
                cache={},
                language="auto",  # 自动检测语言
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,  # 启用 VAD 断句
                merge_length_s=15,
            )

            for result in res:
                text = result["text"]
                cleaned_text = re.sub(pattern, '', text)
                results.append({"index": len(results) + 1, "start": start_time, "end": end_time, "text": cleaned_text})

        if display_pure_text:
            return "\n".join([res['text'] for res in results])
        else:
            return display_results(results)
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"

def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]

def format_time(time_in_ms):
    hours = time_in_ms // 3600000
    minutes = (time_in_ms % 3600000) // 60000
    seconds = ((time_in_ms % 3600000) % 60000) // 1000
    milliseconds = ((time_in_ms % 3600000) % 60000) % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def display_results(results):
    output = ""
    for result in results:
        formatted_time_start = format_time(result['start'])
        formatted_time_end = format_time(result['end'])
        output += f"{result['index']}\n{formatted_time_start} --> {formatted_time_end}\n{result['text']}\n\n"
    return output

# Gradio 接口定义
iface = gr.Interface(
    fn=lambda file_path, display_pure_text: process_audio(file_path, display_pure_text),
    inputs=[
        gr.File(file_types=[".wav", ".flac", ".mp3", ".mp4", ".flv", ".rmvb"], label="上传音频文件"),
        gr.Checkbox(label="仅显示纯文本", value=False)  # 更改参数名称
    ],
    outputs=gr.Textbox(label="转录结果"),
    title="音频转文字",
    description="上传一个音频文件，程序将会转录其中的文字。",
    article="<p style='text-align: center'>本应用使用 FunASR 和 Gradio 构建。</p>"
)

if __name__ == "__main__":
    iface.launch(share=False)