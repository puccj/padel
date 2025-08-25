from padel_analyzer import PadelAnalyzer
from csv_analyzer import CsvAnalyzer
import argparse

def main(input_video_path,
         cam_name='test', 
         cam_type=None,
         second_camera=False,
         model='models/yolov11x.pt',
         ball_model='models/ball-11x-1607.pt',
         recalculate=False,
         show_video=False,
         debug=False,
         mini_court=True):
    
    # analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)
    analyzer = PadelAnalyzer(input_video_path, cam_name, cam_type, second_camera=second_camera, recalculate=recalculate, save_interval=200)

    out_video, fps, out_csvs = analyzer.process(model, ball_model, show_video, debug, mini_court)
    # analyzer.process_all(model=PadelAnalyzer.Model.FAST)

    print(f"Output video saved to: {out_video}")
    print(f"Starting CSV analysis...")

    for i, csv in enumerate(out_csvs):
        csv_analyzer = CsvAnalyzer(csv, fps)
        csv_analyzer.create_heatmaps(alpha=0.05, draw_net=False)
        csv_analyzer.create_videos(field_height=800, draw_net=False, speed_factor=2, trace=0, alpha=0.05)
        # csv_analyzer.create_heatmaps_and_video(field_height=800, draw_net=False, trace=0, alpha=0.05, speed_factor=2, trace=0, alpha=0.05)

        csv_analyzer.create_graphs()

        print(f"Analysis complete ({i+1}/{len(out_csvs)})")
        print("Players' ID")
        print(csv_analyzer.get_selected_ids())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Padel Analyzer',
                                     description='Analyze padel videos to track players and balls.')
    
    parser.add_argument('input_path', type=str, default='input_videos/test.mp4', help='input image path')
    parser.add_argument('-n', '--name', type=str, default='test', help='camera name (e.g: LU01, ST05, etc...)')
    parser.add_argument('-t', '--type', type=str, default=None, 
                        help='camera model name (or an alias of it). If set to None (default), no fisheye correction will be applied.')
    parser.add_argument('-c', '--camera', '--second_camera', action=argparse.BooleanOptionalAction, 
                        help='flag to indicate if this is the second camera and the measurements ' \
                        'should be flipped (e.g: up corner is (10,20) instead of (0,0))')
    parser.add_argument('-M', '--model', type=str, default='models/yolo11x.pt',
                        help='path to the YOLO model for player detection. Default is "models/yolo11x.pt".')
    parser.add_argument('-b', '--ball_model', type=str, default='/home/lookatme/padel/models/ball-11x-1607.pt',
                        help='path to the YOLO model for ball detection. Default is "/home/lookatme/padel/models/ball-11x-1607.pt".')
    parser.add_argument('-r', '--recalculate', action=argparse.BooleanOptionalAction, help='recalculate camera matrices and fps')
    parser.add_argument('-s', '--show',  action=argparse.BooleanOptionalAction, help='show video')
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help='debug mode')
    parser.add_argument('-m', '--mini_court', action=argparse.BooleanOptionalAction, help='draw mini court')
    args = parser.parse_args()

    main(args.input_path, args.name, args.type, args.camera, args.model, args.ball_model, args.recalculate, args.show, args.debug, args.mini_court)