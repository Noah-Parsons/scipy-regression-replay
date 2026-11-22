#!/bin/bash
# Build one isolated environment per SciPy release and run the replay corpus in each.
# Each line: python version, scipy version, numpy version (a NumPy of the same period).
set -u
cd "$(dirname "$0")"
mkdir -p envs out
while read -r py sv nv; do
  [ -z "$py" ] && continue
  [ -f "out/scipy-$sv.npz" ] && { echo "skip $sv"; continue; }
  env="envs/scipy-$sv"
  if [ ! -x "$env/bin/python" ]; then
    uv venv -q -p "$py" "$env" || { echo "FAIL venv $sv"; continue; }
    uv pip install -q -p "$env/bin/python" "scipy==$sv" "numpy==$nv" || { echo "FAIL install $sv"; continue; }
  fi
  ( ulimit -v 4000000; timeout 1200 "$env/bin/python" replay.py "out/scipy-$sv.npz" ) 2>&1 | tail -1
  rm -rf "$env"          # keep disk use low; the output is what matters
done <<'EOF'
3.8 1.3.3 1.17.5
3.8 1.4.1 1.18.5
3.8 1.5.4 1.19.5
3.8 1.6.3 1.20.3
3.8 1.7.3 1.21.6
3.8 1.8.1 1.22.4
3.10 1.9.3 1.23.5
3.10 1.10.1 1.24.4
3.11 1.11.4 1.26.4
3.11 1.12.0 1.26.4
3.11 1.13.1 1.26.4
3.12 1.14.1 2.0.2
3.12 1.15.3 2.2.6
3.13 1.16.3 2.3.5
3.13 1.17.1 2.3.5
3.13 1.18.1 2.4.6
EOF
echo ALLDONE
