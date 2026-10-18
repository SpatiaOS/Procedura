#!/usr/bin/env bash
# Local: bash scripts/install-mcp.sh --agent codex|claude|both
# Online: curl -fsSL https://raw.githubusercontent.com/SpatiaOS/Procedura/main/scripts/install-mcp.sh | bash -s -- --agent codex
set -euo pipefail

agent=both
destination=""
revision=main
no_deps=0
check_only=0
setup_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --agent|--dir|--ref|--openscad|--blender|--isaac)
      if [ "$#" -lt 2 ]; then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --agent) agent="$2" ;;
        --dir) destination="$2" ;;
        --ref) revision="$2" ;;
        *) setup_args+=("$1" "$2") ;;
      esac
      shift 2 ;;
    --no-deps) no_deps=1; shift ;;
    --check) check_only=1; shift ;;
    -h|--help)
      printf '%s\n' 'Install Procedura MCP for Codex, Claude Code or both.' \
        '  --agent codex|claude|both  Agents to configure (default both)' \
        '  --dir PATH                Clone/install location (online default ~/.local/share/Procedura)' \
        '  --ref REF                 Git branch/tag to clone (default main)' \
        '  --no-deps                 Use existing native tools, Pillow and Bun packages' \
        '  --check                   Check the current local installation without writes/downloads' \
        '  --openscad/--blender PATH Override native executable discovery' \
        '  --isaac PATH              Optional existing Isaac Sim directory'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$agent" in codex|claude|both) ;; *) printf 'Invalid agent: %s\n' "$agent" >&2; exit 2 ;; esac

source_path="${BASH_SOURCE[0]:-}"
repo=""
if [ -n "$source_path" ] && [ -f "$source_path" ]; then
  candidate="$(cd "$(dirname "$source_path")/.." && pwd)"
  if [ -f "$candidate/mcp/setup.py" ]; then repo="$candidate"; fi
fi
if [ -z "$destination" ] && [ -n "$repo" ]; then destination="$repo"; fi
destination="${destination:-$HOME/.local/share/Procedura}"

if [ ! -f "$destination/mcp/setup.py" ]; then
  if [ "$check_only" = 1 ]; then printf 'No MCP installation at %s\n' "$destination" >&2; exit 1; fi
  if [ -e "$destination" ]; then printf 'Destination exists but is not a Procedura MCP checkout: %s\n' "$destination" >&2; exit 1; fi
  command -v git >/dev/null || { printf 'git is required\n' >&2; exit 1; }
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 --branch "$revision" https://github.com/SpatiaOS/Procedura.git "$destination"
fi
repo="$(cd "$destination" && pwd)"
if [ "$check_only" = 0 ] && [ -n "$source_path" ] && [ -f "$source_path" ]; then
  expected="$(cd "$(dirname "$source_path")/.." && pwd)"
  if [ "$expected" != "$repo" ] && [ -f "$expected/mcp/setup.py" ]; then
    printf 'Use the installer inside the selected checkout: %s/scripts/install-mcp.sh\n' "$repo" >&2; exit 1
  fi
fi
python_cmd="${PROCEDURA_SETUP_PYTHON:-python3}"
command -v "$python_cmd" >/dev/null || { printf 'Python 3.11+ is required\n' >&2; exit 1; }
"$python_cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else "Python 3.11+ is required")'
if [ "$check_only" = 0 ] && [ "$no_deps" = 0 ]; then
  bash "$repo/scripts/install-deps.sh" --mcp
  export PATH="$HOME/.bun/bin:$PATH"
  # Keep Python packages inside this checkout, even on externally managed systems.
  if [ ! -x "$repo/.procedura/venv/bin/python3" ]; then
    "$python_cmd" -m venv "$repo/.procedura/venv"
  fi
  "$repo/.procedura/venv/bin/python3" -m pip install --disable-pip-version-check -r "$repo/mcp/requirements.txt"
fi
if [ -x "$repo/.procedura/venv/bin/python3" ]; then python_cmd="$repo/.procedura/venv/bin/python3"; fi
if [ "$check_only" = 1 ]; then setup_args+=(--check); fi
"$python_cmd" "$repo/mcp/setup.py" --agent "$agent" "${setup_args[@]}"
