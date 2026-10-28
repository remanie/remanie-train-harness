#!/bin/bash
set -eu

mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
cd /tmp

assert_stopped() {
    local pid
    pid=$(cat "$1")
    [[ "$pid" =~ ^[0-9]+$ ]]
    if kill -0 "$pid" 2>/dev/null; then
        printf 'Process %s from %s survived into verification\n' "$pid" "$1" >&2
        return 1
    fi
}

case "$REMANIE_CHECK_MODE" in
    completed)
        test "$(cat result.txt)" = after
        ! grep -q OPENROUTER_API_KEY child-env.txt
        assert_stopped background.pid
        assert_stopped cleanup.pid
        ;;
    timeout)
        assert_stopped timeout.pid
        ;;
    *) exit 1 ;;
esac

printf '1\n' > /logs/verifier/reward.txt
printf 'Verified %s fixture and process cleanup\n' "$REMANIE_CHECK_MODE"
