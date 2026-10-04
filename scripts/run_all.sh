#!/usr/bin/env bash
# Run a list of training jobs back to back, so a rented GPU never sits idle.
#
# Usage (from the repo root, ideally inside tmux):
#   bash scripts/run_all.sh jobs/main.txt /path/to/runs [extra flags for every job...]
#   e.g. bash scripts/run_all.sh jobs/main.txt ~/runs --micro_G 0 --tf32
#
# Each line of the jobs file is:  <run_name> <grpo.train flags...>   ('#' comments and blank lines are skipped)
# - A job that finished (marker <run_dir>/.done) is skipped, so the script can simply be restarted.
# - A job that was interrupted resumes from its checkpoint (grpo.train does that by itself).
# - A failed job is reported and the script moves on to the next one.
# - Each job's full output goes to <run_dir>/train.log.

set -u -o pipefail

JOBS_FILE=${1:?usage: run_all.sh <jobs_file> <ckpt_dir> [extra flags...]}
CKPT_DIR=${2:?usage: run_all.sh <jobs_file> <ckpt_dir> [extra flags...]}
shift 2
EXTRA=("$@")

done_jobs=(); skipped=(); failed=()

while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    run_dir="$CKPT_DIR/$name"
    if [[ -f "$run_dir/.done" ]]; then
        echo "=== skip $name (already finished)"
        skipped+=("$name"); continue
    fi
    mkdir -p "$run_dir"
    echo "=== start $name  $(date '+%F %T')"
    # $args is unquoted on purpose: it holds several flags that must be split into separate words.
    # < /dev/null: the job must not read the rest of the jobs file from stdin.
    if python -u -m grpo.train --run_name "$name" --ckpt_dir "$CKPT_DIR" $args "${EXTRA[@]}" \
            < /dev/null 2>&1 | tee -a "$run_dir/train.log"; then
        touch "$run_dir/.done"
        done_jobs+=("$name")
        echo "=== done  $name  $(date '+%F %T')"
    else
        failed+=("$name")
        echo "=== FAILED $name  $(date '+%F %T')  (see $run_dir/train.log)"
    fi
done < "$JOBS_FILE"

echo
echo "finished: ${#done_jobs[@]}   skipped: ${#skipped[@]}   failed: ${#failed[@]}"
[[ ${#failed[@]} -gt 0 ]] && printf '  failed: %s\n' "${failed[@]}"
[[ ${#failed[@]} -eq 0 ]]
