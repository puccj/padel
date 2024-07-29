from padel_analyzer import PadelAnalyzer
from postprocess import create_heatmaps

def main ():
    input_video_path = "input_videos/input_video.mp4"
    # output_video_path = "output_videos/output_video.mp4"
    # csv_path = "output_data/prova.csv"
    cam_name = "cam1"

    # analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)
    analyzer = PadelAnalyzer(input_video_path, cam_name)

    out_video, out_csvs = analyzer.process_all(method=PadelAnalyzer.Method.ACCURATE, recalculate_matrix=False, debug=True)
    # analyzer.process_all(method=PadelAnalyzer.Method.FAST)

    # Create 4 heatmaps for each period
    # for csv in out_csvs:
    #     create_heatmaps(csv)


if __name__ == "__main__":
    main()