import cv2
import os
from gemini import analyze_parking_image, calculate_occupancy_rate
import json
import argparse

def extract_frames(video_path, output_dir, frame_interval=300):
    """
    从视频中每隔指定帧数提取一帧
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 提取帧的间隔(默认300帧，假设视频30fps则为每10秒一帧)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    # 获取视频的FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_count += 1
            print(f"Saved frame {saved_count} at {frame_count/fps:.2f}s")
            
        frame_count += 1
        
    cap.release()
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_count}")
    return saved_count

def analyze_video(video_path):
    """
    分析整个视频的停车状态
    """
    # 为每个视频创建单独的临时目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = f"temp_frames_{video_name}"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # 提取帧
    print(f"Extracting frames from {video_path}...")
    num_frames = extract_frames(video_path, temp_dir)
    
    # 分析每一帧
    results = {}
    for i in range(num_frames):
        frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")
        print(f"Analyzing frame {i+1}/{num_frames}...")
        frame_result = analyze_parking_image(frame_path)
        if frame_result:
            results[f"frame_{i}"] = frame_result
            
            # 保存中间结果
            output_file = f"parking_analysis_gemini_{video_name}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
                
    return results

if __name__ == "__main__":
    # 支持命令行参数
    parser = argparse.ArgumentParser(description='Parking Lot Analysis using Gemini')
    parser.add_argument('--videos', nargs='+', help='Path to video files', default=[
        "/autodl-tmp/projects/SoM/videos/215011112024.mp4",
        "/autodl-tmp/projects/SoM/videos/071511112024.mp4",
        "/autodl-tmp/projects/SoM/videos/213911112024.mp4",
        "/autodl-tmp/projects/SoM/videos/214511112024.mp4",
        "/autodl-tmp/projects/SoM/videos/215411112024.mp4"
    ])
    parser.add_argument('--interval', type=int, default=300, help='Frame interval for extraction')
    args = parser.parse_args()
    
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            continue
            
        print(f"\nAnalyzing {video_path}...")
        try:
            results = analyze_video(video_path)
            occupancy_rate = calculate_occupancy_rate(results)
            print(f"Average occupancy rate for {video_path}: {occupancy_rate:.2f}%")
        except Exception as e:
            print(f"Error processing {video_path}: {e}") 