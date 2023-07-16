import os
import cv2
from config import Config
from utils import ObsHolder, make_grid

def images_to_video(save_no, fps):
    save_dir = "./saves"
    image_folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f)) and f.isdigit()]
    image_folder = f'{save_dir}/{image_folders[save_no]}'


    video_name = f'{image_folder}/vid.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # read the first image to get the shape
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    images_to_video(-1, fps=3)

    sampler = ObsHolder(Config())

    grid, _, _ = make_grid(25)

    for xs in grid:
        sampler.make_obs(xs)
    sampler.plot_samples()
