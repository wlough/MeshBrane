#!/bin/bash

# image_dir=$1
# if image_dir is not provided, use the current directory
if [ -z "$1" ]
then
    image_dir=$(pwd)
else
    image_dir=$1
fi
# movie_path=$2
# if movie_dir is not provided, use image_dir
if [ -z "$2" ]; then
    movie_dir=$image_dir
else
    movie_dir=$2
fi

image_format="png"
image_prefix="frame"
index_length=6
movie_name="movie"
movie_format="mp4"

frame_rate=30
# frame_rate=80
# frame_size="1080x720"
frame_size="1920x1080"
video_codec="libx264"
# video_quality=1
video_quality=15
pixel_format="yuv420p"
# playback speed
# filter_v="setpts=0.25*PTS"
filter_v="pad=ceil(iw/2)*2:ceil(ih/2)*2,setpts=0.25*PTS"


image_filename="${image_prefix}_%0${index_length}d.${image_format}"
movie_filename="${movie_name}.${movie_format}"


starting_dir=$(pwd)

run_command="ffmpeg"
ffmpegFLAGS=""
# overwrite output file without asking if it already exists
ffmpegFLAGS="$ffmpegFLAGS -y"
# frame rate (Hz)
ffmpegFLAGS="$ffmpegFLAGS -r $frame_rate"
# frame width x height (pixels)
ffmpegFLAGS="$ffmpegFLAGS -s $frame_size"
# input files path and format
ffmpegFLAGS="$ffmpegFLAGS -i $image_filename"
# video codec
ffmpegFLAGS="$ffmpegFLAGS -vcodec $video_codec"
# video quality, lower means better
ffmpegFLAGS="$ffmpegFLAGS -crf $video_quality"
# pixel format
ffmpegFLAGS="$ffmpegFLAGS -pix_fmt $pixel_format"
# speed up
ffmpegFLAGS="$ffmpegFLAGS -filter:v $filter_v"
# output file
ffmpegFLAGS="$ffmpegFLAGS $movie_filename"


run_command="$run_command $ffmpegFLAGS"
echo $run_command
# Start the process
cd $image_dir
$run_command
cd $starting_dir

if [ $movie_dir != $image_dir ]; then
    mv $image_dir/$movie_filename $movie_dir
fi

echo "Movie saved at ${movie_dir}/$movie_filename"
