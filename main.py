from padel_analyzer import PadelAnalyzer
from postprocess import create_heatmaps

def main ():
    input_video_path = "input_videos/input_video.mp4"
    output_video_path = "output_videos/output_video.avi"
    csv_path = "output_data/prova.csv"
    cam_name = "cam1"

    analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)

    analyzer.process_all()

    create_heatmaps(csv_path)


if __name__ == "__main__":
    main()