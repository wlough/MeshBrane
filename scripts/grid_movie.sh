#!/bin/bash

num_rows=$1
num_cols=$2
output_dir=$3

# max of rows or cols
max_num=$((num_rows > num_cols ? num_rows : num_cols))
bigR=1920
bigC=1080
littleR=$((bigR / max_num))
littleC=$((bigC / max_num))
little_size="${littleR}x${littleC}"
bigR=$((littleR * num_cols))
bigC=$((littleC * num_rows))
# bigR must be divisible by 2 so add 1 if odd
if [ $((bigR % 2)) -ne 0 ]; then
    bigR=$((bigR + 1))
fi
# bigC must be divisible by 2 so add 1 if odd
if [ $((bigC % 2)) -ne 0 ]; then
    bigC=$((bigC + 1))
fi
final_size="${bigR}x${bigC}"

# echo "bigR: $bigR"
# echo "bigC: $bigC"
# echo "littleR: $littleR"
# echo "littleC: $littleC"
# echo "final_size: $final_size"

# bigR=1920
# bigC=1080
# final_size="${bigR}x${bigC}"
# littleR=$((bigR / num_rows))
# littleC=$((bigC / num_cols))
# little_size="${littleR}x${littleC}"
run_prefix="run"
path_start=${output_dir}/${run_prefix}
path_end=/visualizations/movie.mp4

get_input_path() {
    local r=$1
    local c=$2
    local input_path="${path_start}"
    input_path="${input_path}_"
    if [ $r -lt 10 ]; then
        input_path="${input_path}0"
    fi
    input_path="${input_path}${r}"

    input_path="${input_path}_"
    if [ $c -lt 10 ]; then
        input_path="${input_path}0"
    fi
    input_path="${input_path}${c}"

    input_path="${input_path}${path_end}"
    echo "$input_path"
}

# if num_cols>1 and num_rows>1, use grid
get_grid_command() {

    # add input paths
    ffmpeg_command="ffmpeg"
    for r in $(seq 0 $((num_rows - 1))); do
        for c in $(seq 0 $((num_cols - 1))); do
            input_path=$(get_input_path $r $c)
            ffmpeg_command+=" -i $input_path"
        done
    done

    # Add the filter_complex part
    ffmpeg_command+=' -filter_complex "'
    for r in $(seq 0 $((num_rows - 1))); do
        for c in $(seq 0 $((num_cols - 1))); do
            index=$((r * num_cols + c))
            ffmpeg_command+="[${index}:v]scale=${little_size}[v${index}];"
        done
    done

    for r in $(seq 0 $((num_rows - 1))); do
        for c in $(seq 0 $((num_cols - 1))); do
            index=$((r * num_cols + c))
            ffmpeg_command+="[v${index}]"
        done
        ffmpeg_command+="hstack=inputs=${num_cols}[r${r}]; "
    done

    for r in $(seq 0 $((num_rows - 1))); do
        ffmpeg_command+="[r${r}]"
    done

    # ffmpeg_command+="vstack=inputs=${num_rows}[v]; [v]scale=${final_size}\" "
    ffmpeg_command+="vstack=inputs=${num_rows}[v];"
    ffmpeg_command+="[v]scale=${final_size}\" "

    # Add the output path
    ffmpeg_command+="${output_dir}/grid_movie.mp4"
    echo $ffmpeg_command
}

# if num_cols is 1, use vstack
get_vstack_command() {
    # add input paths
    ffmpeg_command="ffmpeg"
    for r in $(seq 0 $((num_rows - 1))); do
        for c in $(seq 0 $((num_cols - 1))); do
            input_path=$(get_input_path $r $c)
            ffmpeg_command+=" -i $input_path"
        done
    done
    # Add the filter_complex part
    ffmpeg_command+=' -filter_complex "'
    for r in $(seq 0 $((num_rows - 1))); do
        index=$r
        ffmpeg_command+="[${index}:v]scale=${little_size}[v${index}];"
    done
    for r in $(seq 0 $((num_rows - 1))); do
        ffmpeg_command+="[v${r}]"
    done
    ffmpeg_command+="vstack=inputs=${num_rows}[v];"
    ffmpeg_command+="[v]scale=${final_size}\" "
    # Add the output path
    ffmpeg_command+="${output_dir}/grid_movie.mp4"
    echo $ffmpeg_command
}

#  if num_rows is 1, use hstack
get_hstack_command() {
    # add input paths
    ffmpeg_command="ffmpeg"
    for r in $(seq 0 $((num_rows - 1))); do
        for c in $(seq 0 $((num_cols - 1))); do
            input_path=$(get_input_path $r $c)
            ffmpeg_command+=" -i $input_path"
        done
    done
    # Add the filter_complex part
    ffmpeg_command+=' -filter_complex "'
    for c in $(seq 0 $((num_cols - 1))); do
        index=$c
        ffmpeg_command+="[${index}:v]scale=${little_size}[v${index}];"
    done
    for c in $(seq 0 $((num_cols - 1))); do
        ffmpeg_command+="[v${c}]"
    done
    ffmpeg_command+="hstack=inputs=${num_cols}[v];"
    ffmpeg_command+="[v]scale=${final_size}\" "
    # Add the output path
    ffmpeg_command+="${output_dir}/grid_movie.mp4"
    echo $ffmpeg_command

}


# for r in $(seq 0 $((num_rows - 1))); do
#     for c in $(seq 0 $((num_cols - 1))); do
#         input_path=$(get_input_path $r $c)
#         echo "Checking dimensions for $input_path"
#         ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$input_path"
#     done
# done

if [ $num_cols -gt 1 ] && [ $num_rows -gt 1 ]; then
    ffmpeg_command=$(get_grid_command)
elif [ $num_cols -eq 1 ]; then
    ffmpeg_command=$(get_vstack_command)
elif [ $num_rows -eq 1 ]; then
    ffmpeg_command=$(get_hstack_command)
else
    echo "Error: num_cols and num_rows must be greater than 0"
    exit 1
fi

echo $ffmpeg_command
eval $ffmpeg_command