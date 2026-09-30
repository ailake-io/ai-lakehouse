#!/usr/bin/env bash
set -euo pipefail

binary=${1:?usage: check_multi_process_lock.sh BINARY STORE TABLE [PORT1] [PORT2]}
store=${2:?usage: check_multi_process_lock.sh BINARY STORE TABLE [PORT1] [PORT2]}
table=${3:?usage: check_multi_process_lock.sh BINARY STORE TABLE [PORT1] [PORT2]}
port_one=${4:-7711}
port_two=${5:-7712}
log_dir=${TMPDIR:-/tmp}/ailake-lock-test-$$
mkdir -p "$log_dir"
first_pid=""
second_pid=""

cleanup() {
  if [[ -n "$second_pid" ]] && kill -0 "$second_pid" 2>/dev/null; then
    kill "$second_pid" 2>/dev/null || true
  fi
  if [[ -n "$first_pid" ]] && kill -0 "$first_pid" 2>/dev/null; then
    kill "$first_pid" 2>/dev/null || true
  fi
  wait "$second_pid" 2>/dev/null || true
  wait "$first_pid" 2>/dev/null || true
  rm -rf "$log_dir"
}
trap cleanup EXIT

"$binary" --store "$store" serve "$table" --port "$port_one" >"$log_dir/first.log" 2>&1 &
first_pid=$!
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$port_one/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$first_pid" 2>/dev/null; then
    cat "$log_dir/first.log"
    exit 1
  fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$port_one/healthz" >/dev/null

set +e
"$binary" --store "$store" serve "$table" --port "$port_two" >"$log_dir/second.log" 2>&1 &
second_pid=$!
wait "$second_pid"
second_status=$?
set -e
if [[ "$second_status" -eq 0 ]]; then
  echo "second ailake serve unexpectedly acquired the table lock"
  cat "$log_dir/second.log"
  exit 1
fi
grep -q "already owns table" "$log_dir/second.log"
echo "multi-process serve lock: PASS"
