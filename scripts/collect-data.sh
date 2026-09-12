#!/bin/sh
set -eu

base_repo_id="${BASE_REPO_ID:?Source config/lab.env first}"
base_dataset_root="${BASE_DATASET_ROOT:?Source config/lab.env first}"
default_num_episodes=150

usage() {
    cat <<'EOF'
Usage:
  ./collect-data.sh [--resume] [--display]
  ./collect-data.sh --batches N --episodes-per-batch M [--start-batch K] [--display]

Options:
  --resume                  Continue the legacy single dataset.
  --display                 Show live camera and robot data in Foxglove.
  --batches N               Final batch index (runs start-batch through N).
  --episodes-per-batch M    Number of episodes in each independent dataset.
  --start-batch K           Start batch mode at K instead of 1.
  -h, --help                Show this help.

Examples:
  ./collect-data.sh --batches 15 --episodes-per-batch 10
  ./collect-data.sh --batches 15 --episodes-per-batch 10 --start-batch 4 --display
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 2
}

is_positive_integer() {
    case "$1" in
        ""|*[!0-9]*|0)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

resume=false
display=false
batch_count=""
episodes_per_batch=""
start_batch=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --resume)
            resume=true
            ;;
        --display)
            display=true
            ;;
        --batches|--episodes-per-batch|--start-batch)
            option="$1"
            shift
            [ "$#" -gt 0 ] || fail "${option} requires a value."
            case "${option}" in
                --batches) batch_count="$1" ;;
                --episodes-per-batch) episodes_per_batch="$1" ;;
                --start-batch) start_batch="$1" ;;
            esac
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            ;;
    esac
    shift
done

if [ -n "${batch_count}" ] || [ -n "${episodes_per_batch}" ]; then
    [ -n "${batch_count}" ] || fail "--batches is required in batch mode."
    [ -n "${episodes_per_batch}" ] || fail "--episodes-per-batch is required in batch mode."
    is_positive_integer "${batch_count}" || fail "--batches must be a positive integer."
    is_positive_integer "${episodes_per_batch}" || fail "--episodes-per-batch must be a positive integer."
    is_positive_integer "${start_batch}" || fail "--start-batch must be a positive integer."
    [ "${start_batch}" -le "${batch_count}" ] || fail "--start-batch cannot exceed --batches."
    [ "${resume}" = false ] || fail "--resume is not supported in independent batch mode. Restart an incomplete batch instead."
fi

run_recording() {
    repo_id="$1"
    dataset_root="$2"
    num_episodes="$3"
    resume_dataset="$4"
    encoding_batch_size="$5"

    set -- "--display_data=${display}"
    if [ "${resume_dataset}" = true ]; then
        set -- "$@" --resume=true
    fi

    lerobot-record \
        --robot.type=so101_follower \
        --robot.use_degrees=true \
        "--robot.port=${FOLLOWER_PORT:?}" \
        "--robot.id=${FOLLOWER_ID:?}" \
        --robot.num_read_retries=10 \
        "--robot.cameras=${CAMERAS:?}" \
        --teleop.type=so101_leader \
        "--teleop.port=${LEADER_PORT:?}" \
        "--teleop.id=${LEADER_ID:?}" \
        "--dataset.repo_id=${repo_id}" \
        "--dataset.root=${dataset_root}" \
        --dataset.single_task='Find the orange wooden fish, pick up the mallet by its handle, tap the wooden fish exactly three times, return the mallet to its holder, and return the arm to its home pose.' \
        --dataset.fps=30 \
        --dataset.episode_time_s=35 \
        --dataset.reset_time_s=10 \
        "--dataset.num_episodes=${num_episodes}" \
        --dataset.push_to_hub=false \
        --dataset.no_stamp=true \
        --dataset.streaming_encoding=false \
        "--dataset.video_encoding_batch_size=${encoding_batch_size}" \
        --dataset.num_image_writer_processes=0 \
        --dataset.num_image_writer_threads_per_camera=4 \
        --dataset.rgb_encoder.vcodec=h264 \
        --dataset.rgb_encoder.preset=ultrafast \
        --dataset.rgb_encoder.crf=23 \
        --dataset.encoder_threads=2 \
        --display_mode=foxglove \
        --display_ip=127.0.0.1 \
        --display_port=8765 \
        --display_compressed_images=true \
        "$@"
}

validate_batch() {
    dataset_root="$1"
    expected_episodes="$2"
    actual_episodes="$({
        python -c 'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "meta/info.json").read_text())["total_episodes"])' "${dataset_root}"
    } 2>/dev/null)" || fail "Could not read batch metadata at ${dataset_root}."

    [ "${actual_episodes}" = "${expected_episodes}" ] || fail \
        "Batch stopped with ${actual_episodes}/${expected_episodes} episodes. It will not advance to the next batch."
}

if [ -z "${batch_count}" ]; then
    run_recording "${base_repo_id}" "${base_dataset_root}" "${default_num_episodes}" "${resume}" 10
    exit 0
fi

batch_index="${start_batch}"
while [ "${batch_index}" -le "${batch_count}" ]; do
    batch_suffix="$(printf '%03d' "${batch_index}")"
    repo_id="${base_repo_id}_batch_${batch_suffix}"
    dataset_root="${base_dataset_root}_batch_${batch_suffix}"

    [ ! -e "${dataset_root}" ] || fail \
        "Batch ${batch_index} already exists at ${dataset_root}. Move it aside or choose another --start-batch."

    echo "Starting independent batch ${batch_index}/${batch_count}: ${repo_id}"
    run_recording "${repo_id}" "${dataset_root}" "${episodes_per_batch}" false "${episodes_per_batch}"
    validate_batch "${dataset_root}" "${episodes_per_batch}"
    echo "Completed independent batch ${batch_index}/${batch_count}."

    batch_index=$((batch_index + 1))
done

echo "Completed all ${batch_count} independent batches (${episodes_per_batch} episodes each)."
