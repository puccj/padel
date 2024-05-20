from padel_analyzer import PadelAnalyzer
from postprocess import create_heatmaps

def main ():
    # Parameters
    # input_video_path = "input_videos/input_video.mp4"
    input_video_path = "input_videos/videopadel2.mp4"
    output_video_path = "output_videos/output_video.avi"
    csv_path = "output_data/prova.csv"
    # cam_name = "prova"
    cam_name = "video2"


    analyzer = PadelAnalyzer(input_video_path, cam_name, output_video_path, csv_path)

    analyzer.process_all()

    create_heatmaps(csv_path)


    

if __name__ == "__main__":
    main()