import os
import re
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter.ttk import Progressbar

# 初始化变量
audio_files = []
results = []

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

# 创建 Tkinter 窗口
root = tk.Tk()
root.title("音频转文字结果展示")

# 创建变量用于存储用户的选择
save_pure_text = tk.BooleanVar(value=False)
display_pure_text = tk.BooleanVar(value=False)

# 处理音频文件的函数
def process_audio(file_path):
    global results
    if file_path:
        try:
            # 使用 VAD 模型处理音频文件
            vad_res = vad_model.generate(
                input=file_path,
                cache={},
                max_single_segment_time=30000,  # 最大单个片段时长
            )

            # 从 VAD 模型的输出中提取每个语音片段的开始和结束时间
            segments = vad_res[0]['value']  # 假设只有一段音频，且其片段信息存储在第一个元素中

            # 加载原始音频数据
            audio_data, sample_rate = sf.read(file_path)

            results = []
            total_segments = len(segments)
            progress_bar['maximum'] = total_segments
            progress_value = 0
            progress_label = tk.Label(root, text="0/%s" % total_segments)
            progress_label.pack()

            for segment in segments:
                start_time, end_time = segment  # 获取开始和结束时间
                cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)

                # 将裁剪后的音频保存为临时文件
                temp_audio_file = "temp_cropped.wav"
                sf.write(temp_audio_file, cropped_audio, sample_rate)

                # 语音转文字处理
                res = model.generate(
                    input=temp_audio_file,
                    cache={},
                    language="auto",  # 自动检测语言
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,  # 启用 VAD 断句
                    merge_length_s=15,
                )

                # 移除特定模式
                for result in res:
                    text = result["text"]
                    cleaned_text = re.sub(pattern, '', text)

                    # 添加时间戳并转换为 SRT 格式
                    results.append({"index": len(results) + 1, "start": start_time, "end": end_time, "text": cleaned_text})

                progress_value += 1
                progress_bar['value'] = progress_value
                progress_label.config(text=f"{progress_value}/{total_segments}")
                root.update_idletasks()

            # 将结果添加到文本框中
            text_area.delete(1.0, tk.END)
            display_results(file_path)
            text_area.insert(tk.END, f"\n已处理文件: {file_path}\n")
        except Exception as e:
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, f"Error occurred during processing: {str(e)}\n")

# 定义裁剪音频的函数
def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]

# 格式化时间为 SRT 格式
def format_time(time_in_ms):
    hours = time_in_ms // 3600000
    minutes = (time_in_ms % 3600000) // 60000
    seconds = ((time_in_ms % 3600000) % 60000) // 1000
    milliseconds = ((time_in_ms % 3600000) % 60000) % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# 显示结果
def display_results(file_path):
    if display_pure_text.get():
        # 仅显示纯文本
        for result in results:
            text_area.insert(tk.END, result['text'] + '\n')
    else:
        # 显示为字幕格式
        for result in results:
            formatted_time_start = format_time(result['start'])
            formatted_time_end = format_time(result['end'])
            text_area.insert(tk.END, f"{result['index']}\n{formatted_time_start} --> {formatted_time_end}\n{result['text']}\n\n")

# 创建按钮选择音频文件
def select_audio_file():
    global audio_files
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.mp4 *.flv"), ("All Files", "*.*")])
    if file_path:
        audio_files = [file_path]
        file_label.config(text=f"Selected File: {file_path}")

# 创建按钮选择多个文件
def select_multiple_files():
    global audio_files
    file_paths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.mp4 *.flv"), ("All Files", "*.*")])
    if file_paths:
        audio_files = list(file_paths)
        file_label.config(text=f"Selected Files: {len(file_paths)} files")

# 创建按钮选择文件夹
def select_audio_folder():
    global audio_files
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        audio_files = []
        supported_extensions = ('.wav', '.flac', '.mp3', '.mp4', '.flv')
        for filename in os.listdir(folder_selected):
            if filename.lower().endswith(supported_extensions):
                audio_files.append(os.path.join(folder_selected, filename))
        file_label.config(text=f"Selected Folder: {folder_selected}")

# 创建按钮执行转写和保存
def process_and_save():
    global audio_files, results
    if not audio_files:
        select_audio_file()
        select_multiple_files()
        select_audio_folder()
    
    # 处理选定的文件
    for file_path in audio_files:
        process_audio(file_path)
    
    # 根据用户选择保存文件
    save_transcriptions()

# 创建进度条
progress_bar = Progressbar(root, orient='horizontal', length=200, mode='determinate')
progress_bar.pack()

# 创建滚动文本框
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
text_area.pack()

# 创建复选框
save_pure_text_checkbox = tk.Checkbutton(root, text="仅保存纯文本", variable=save_pure_text)
save_pure_text_checkbox.pack()

display_pure_text_checkbox = tk.Checkbutton(root, text="仅显示纯文本", variable=display_pure_text)
display_pure_text_checkbox.pack()

# 创建按钮
select_button = tk.Button(root, text="选择单个文件", command=select_audio_file)
select_button.pack()

select_multiple_button = tk.Button(root, text="选择多个文件", command=select_multiple_files)
select_multiple_button.pack()

select_folder_button = tk.Button(root, text="选择文件夹", command=select_audio_folder)
select_folder_button.pack()

file_label = tk.Label(root, text="Selected File: None")
file_label.pack()

process_and_save_button = tk.Button(root, text="开始转写并保存", command=process_and_save)
process_and_save_button.pack()

# 保存转录文字
def save_transcriptions():
    for file_path in audio_files:
        base_name = os.path.splitext(file_path)[0]
        output_path = base_name + ".txt"

        if save_pure_text.get():
            # 仅保存纯文本
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result['text'] + '\n')
        else:
            # 保存带有时间戳的文本
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    formatted_time_start = format_time(result['start'])
                    formatted_time_end = format_time(result['end'])
                    f.write(f"{result['index']}\n{formatted_time_start} --> {formatted_time_end}\n{result['text']}\n\n")
        
        text_area.insert(tk.END, f"\n已保存文件: {output_path}\n")

# 运行 Tkinter 主循环
root.mainloop()