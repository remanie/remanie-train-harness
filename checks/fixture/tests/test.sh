#!/bin/bash
set -eu

mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
cd /tmp

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

pid_alive() {
    local pid
    pid=$(cat "$1")
    [[ "$pid" =~ ^[0-9]+$ ]] || fail "$1 holds no pid"
    kill -0 "$pid" 2>/dev/null
}

case "$REMANIE_CHECK_MODE" in
    completed)
        test "$(cat result.txt)" = after || fail "result.txt is not after"
        if grep -q OPENROUTER_API_KEY child-env.txt; then fail "a command saw OPENROUTER_API_KEY"; fi
        if pid_alive background.pid; then fail "the killed task survived"; fi
        pid_alive cleanup.pid || fail "the running task was stopped before verification"
        ;;
    timeout)
        pid_alive timeout.pid || fail "the running task was stopped before verification"
        ;;
    *) exit 1 ;;
esac

printf '1\n' > /logs/verifier/reward.txt
printf 'Verified %s fixture and background tasks\n' "$REMANIE_CHECK_MODE"
