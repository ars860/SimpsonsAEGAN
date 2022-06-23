import argparse
import glob
import os

from PIL import Image


def make_gifs(results_folder, type):
    results = results_folder  # "data_linear_upsampling_colorspace_stolen_discriminator_results"
    type = type  # "generated"

    # filepaths
    fp_in = f"{results}/{type}/gen.*.png"
    fp_out = f"{results}/gif/{type}/epochs.gif"

    os.makedirs(f"{results}/gif/{type}", exist_ok=True)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gif maker parameters')
    parser.add_argument('--results', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)

    args = parser.parse_args()

    make_gifs(args.results, args.type)
