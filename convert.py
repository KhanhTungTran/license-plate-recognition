import os


def convert_to_mp4(file_path, target_file_path):
    os.popen("ffmpeg -i {} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {}".format(file_path, target_file_path))
    return True


folder = './videos'
video = '2-3pm.avi'
file_name = video.split('.')[0]
target_video = file_name+'.mp4'
file_path = os.path.join(folder, video)
target_file_path = os.path.join(folder, target_video)
convert_to_mp4(file_path, target_file_path)
