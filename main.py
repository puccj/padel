from padel_analyzer import PadelAnalyzer
from postprocess import create_heatmaps

def main ():
    input_video_path = "input_videos/18-05-2024-15-44.mp4"
    # output_video_path = "output_videos/output_video.avi"
    # csv_path = "output_data/prova.csv"
    cam_name = "cam1"

    # analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)
    analyzer = PadelAnalyzer(input_video_path, cam_name)

    analyzer.process_all(method=PadelAnalyzer.Method.ACCURATE)
    # analyzer.process_all(method=PadelAnalyzer.Method.FAST)

    # Create 4 heatmaps for each period
    create_heatmaps('output_data/camera-1.csv', 'output_heatmaps/period1')
    create_heatmaps('output_data/camera-2.csv', 'output_heatmaps/period2')
    create_heatmaps('output_data/camera-3.csv', 'output_heatmaps/period3')


if __name__ == "__main__":
    main()