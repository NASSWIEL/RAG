#!/usr/bin/env bash
# Claude Code status line — uses Python for JSON parsing (no jq required)

input=$(cat)

# Parse all fields in one Python call
eval "$(echo "$input" | python -c "
import sys, json, os

data = json.load(sys.stdin)

model_id = (data.get('model') or {}).get('id', '')
if 'opus' in model_id:
    model = 'opus'
elif 'sonnet' in model_id:
    model = 'sonnet'
elif 'haiku' in model_id:
    model = 'haiku'
else:
    model = (data.get('model') or {}).get('display_name', 'unknown')

ctx = data.get('context_window') or {}
used_pct = ctx.get('used_percentage')
total_in  = ctx.get('total_input_tokens', 0) or 0
total_out = ctx.get('total_output_tokens', 0) or 0

cwd = (data.get('workspace') or {}).get('current_dir') or data.get('cwd', '')

print(f'MODEL={model}')
print(f'USED_PCT={used_pct if used_pct is not None else \"\"}')
print(f'TOTAL_IN={int(total_in)}')
print(f'TOTAL_OUT={int(total_out)}')
print(f'CWD={cwd}')
" 2>/dev/null)"

# --- Context bar ---
if [ -n "$USED_PCT" ]; then
  used_int=$(printf '%.0f' "$USED_PCT")
  filled=$(( used_int / 10 ))
  empty=$(( 10 - filled ))
  bar=""
  for i in $(seq 1 $filled); do bar="${bar}#"; done
  for i in $(seq 1 $empty);  do bar="${bar}-"; done
  ctx_block="[${bar}] ${used_int}%"
  if [ "$used_int" -ge 90 ]; then
    ctx_color='\033[31m'
  elif [ "$used_int" -ge 70 ]; then
    ctx_color='\033[33m'
  else
    ctx_color='\033[32m'
  fi
else
  ctx_block="[----------] --%"
  ctx_color='\033[32m'
fi

# --- Git branch ---
git_branch=""
if [ -n "$CWD" ]; then
  git_branch=$(git -C "$CWD" --no-optional-locks symbolic-ref --short HEAD 2>/dev/null)
fi

# --- Cost ---
case "$MODEL" in
  opus)   in_price="15";   out_price="75" ;;
  haiku)  in_price="0.80"; out_price="4"  ;;
  *)      in_price="3";    out_price="15" ;;
esac
cost=$(python -c "print(f'\${(${TOTAL_IN}*${in_price} + ${TOTAL_OUT}*${out_price})/1000000:.4f}')" 2>/dev/null || echo '$0.0000')

# --- Dir ---
dir_block="${CWD##*/}"
[ -z "$dir_block" ] && dir_block="?"

# --- Assemble ---
out="\033[36m${MODEL}\033[0m"
out="${out}  ${ctx_color}${ctx_block}\033[0m"
[ -n "$git_branch" ] && out="${out}  \033[35m(${git_branch})\033[0m"
out="${out}  \033[33m${cost}\033[0m"
out="${out}  \033[34m${dir_block}\033[0m"

printf "%b" "$out"
