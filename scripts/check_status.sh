#!/bin/bash
# Status of every task in src/log/queue/diag-*/worker*.txt (2026-11-06 onwards), deduplicated across queue dirs:
#   DONE (finished log) | RUNNING / QUEUED (in an active job) | FAILED (error in its .out) |
#   STOPPED (started, job ended without finishing: walltime) | NOTRUN (job ended or never submitted)
#     bash scripts/check_status.sh            # run from the repo root
# Re-running the same submit_diag.sh command resubmits everything that is not DONE / RUNNING / QUEUED.
cd "$(dirname "$0")/.."
ACTIVE=$(for j in $(qselect -u "$USER" 2>/dev/null); do qstat -f "$j" | tr -d '\n\t ' | grep -o 'TASK_FILE=[^,]*' | cut -d= -f2; done)
declare -A ST WHERE; rank() { case $1 in DONE) echo 5;; RUNNING) echo 4;; QUEUED) echo 3;; FAILED|STOPPED) echo 2;; NOTRUN) echo 1;; *) echo 0;; esac; }
for w in src/log/queue/diag-2026*/worker*.txt; do
  q=$(dirname "$w"); act=0; grep -qxF "$w" <<< "$ACTIVE" && act=1
  while read -r n m d s rest; do
    [ -z "$n" ] && continue; k="$m $d $n"; o="$q/$m-$d-$n.out"
    if grep -l 'BEST ████' src/log/"$m-$d-$n"-*.log >/dev/null 2>&1; then st=DONE
    elif [ -f "$o" ] && grep -qE 'FAILED|Traceback|OutOfMemory|Killed|Error' "$o"; then st=FAILED
    elif [ -f "$o" ]; then [ $act = 1 ] && st=RUNNING || st=STOPPED
    else [ $act = 1 ] && st=QUEUED || st=NOTRUN; fi
    if [ "$(rank $st)" -gt "$(rank "${ST[$k]:-x}")" ]; then ST[$k]=$st; WHERE[$k]=$o; fi
  done < "$w"
done
echo "== tổng: $(for k in "${!ST[@]}"; do echo "${ST[$k]}"; done | sort | uniq -c | tr '\n' ' ')"
for s in RUNNING QUEUED FAILED STOPPED NOTRUN; do
  L=$(for k in "${!ST[@]}"; do [ "${ST[$k]}" = $s ] && echo "$k"; done | sort)
  [ -z "$L" ] && continue; echo; echo "== $s ($(echo "$L" | wc -l))"
  while read -r k; do
    if [ $s = FAILED ] || [ $s = STOPPED ]; then m=$(grep -E 'Error|Killed|walltime|exceeded' "${WHERE[$k]}" 2>/dev/null | tail -1); [ -z "$m" ] && m=$(tail -1 "${WHERE[$k]}" 2>/dev/null); echo "  $k   <- ${m:0:120}"; else echo "  $k"; fi
  done <<< "$L"
done
ls src/log/queue/diag-2026*/pending.sh 2>/dev/null | sed 's/^/pending: /'
