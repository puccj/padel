from padel_analyzer import PadelAnalyzer
from csv_analyzer import CsvAnalyzer
import argparse

def main(input_video_path, cam_name='test', second_camera=False, recalculate=False, show_video=False, debug=False):
    
    # analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)
    analyzer = PadelAnalyzer(input_video_path, cam_name, second_camera=second_camera, recalculate=recalculate, save_interval=200)

    out_video, fps, out_csvs = analyzer.process(model=PadelAnalyzer.Model.ACCURATE, show=show_video, debug=debug)
    # analyzer.process_all(model=PadelAnalyzer.Model.FAST)

    for csv in out_csvs:
        csv_analyzer = CsvAnalyzer(csv, fps)
        csv_analyzer.create_heatmaps(alpha=0.05, draw_net=False)
        csv_analyzer.create_videos(field_height=800, draw_net=False, speed_factor=2, trace=0, alpha=0.05)
        # csv_analyzer.create_heatmaps_and_video(field_height=800, draw_net=False, trace=0, alpha=0.05, speed_factor=2, trace=0, alpha=0.05)

        csv_analyzer.create_graphs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', '--input_path', type=str, default='input_videos/test.mp4', help='input image path')
    parser.add_argument('-n', '--name', type=str, default='test', help='camera name')
    parser.add_argument('-r', '--recalculate', action=argparse.BooleanOptionalAction, help='recalculate camera matrices and fps')
    parser.add_argument('-s', '--show',  action=argparse.BooleanOptionalAction, help='show video')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help='debug mode')
    parser.add_argument('-c', '--camera', '--second_camera', action=argparse.BooleanOptionalAction, help='second camera')
    args = parser.parse_args()
    
    main(args.path, args.name, args.camera, args.recalculate, args.show, args.debug)