from padel_analyzer import PadelAnalyzer
from csv_analyzer import CsvAnalyzer

def main ():
    input_video_path = "input_videos/input_video.mp4"
    # output_video_path = "output_videos/output_video.mp4"
    # csv_path = "output_data/prova.csv"
    cam_name = "test"

    # analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)
    analyzer = PadelAnalyzer(input_video_path, cam_name, recalculate_matrix=False, save_interval=200)

    out_video, fps, out_csvs = analyzer.process(method=PadelAnalyzer.Model.ACCURATE, debug=True)
    # analyzer.process_all(method=PadelAnalyzer.Method.FAST)

    for csv in out_csvs:
        csv_analyzer = CsvAnalyzer(csv, fps)
        csv_analyzer.create_heatmaps(alpha=0.05, draw_net=False)
        csv_analyzer.create_videos(field_height=800, draw_net=False, speed_factor=2, trace=0, alpha=0.05)
        # csv_analyzer.create_heatmaps_and_video(field_height=800, draw_net=False, trace=0, alpha=0.05, speed_factor=2, trace=0, alpha=0.05)

        csv_analyzer.create_graphs()


if __name__ == "__main__":
    main()