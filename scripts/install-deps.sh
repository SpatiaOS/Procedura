#!/usr/bin/env bash
#
# Install everything Procedura needs, into $PREFIX (default ~/opt) plus the
# repo's own node_modules. Safe to re-run: anything already working is left
# alone, so this doubles as a "what am I missing?" check.
#
#   bash scripts/install-deps.sh              # everything
#   bash scripts/install-deps.sh --check      # report only, install nothing
#   bash scripts/install-deps.sh --no-blender # skip the ~350 MB download
#
# What it does NOT install: NVIDIA Isaac Sim (multi-GB, account-gated). It is
# only needed for `--motion` physics validation; everything else runs without.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/opt}"
DO_BLENDER=1
DO_OPENSCAD=1
DO_BUN=1
CHECK_ONLY=0
DO_ENV=1
DO_STUDIO=1

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --no-blender) DO_BLENDER=0 ;;
    --no-openscad) DO_OPENSCAD=0 ;;
    --no-bun) DO_BUN=0 ;;
    --mcp) DO_ENV=0; DO_STUDIO=0 ;;
    --prefix) PREFIX="$2"; shift ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# ── output ──────────────────────────────────────────────────────────────────
if [ -t 1 ]; then B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else B=""; DIM=""; G=""; Y=""; R=""; N=""; fi
step() { printf "\n${B}%s${N}\n" "$*"; }
ok()   { printf "  ${G}✓${N} %s\n" "$*"; }
warn() { printf "  ${Y}!${N} %s\n" "$*"; }
bad()  { printf "  ${R}✗${N} %s\n" "$*"; }
info() { printf "  ${DIM}%s${N}\n" "$*"; }

MISSING=()

# The repo's .env already records where the binaries live; honour it, so a
# configured machine reports "found" instead of re-downloading 350 MB.
if [ -f "$REPO/.env" ]; then
  for k in OPENSCAD_PATH PROCEDURA_BLENDER_PATH PROCEDURA_ISAACSIM_PATH; do
    v="$(sed -n "s/^[[:space:]]*$k=//p" "$REPO/.env" | tail -1 | tr -d '"'"'"'"' | tr -d "\r")"
    [ -n "$v" ] && export "$k=$v"
  done
fi

OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS/$ARCH" in
  Linux/x86_64) ;;
  *) warn "Automatic binary installs target Linux x86_64; this is $OS/$ARCH."
     info "Dependencies are still installed, but OpenSCAD and Blender are left to you:"
     [ "$OS" = "Darwin" ] && info "  brew install --cask openscad@snapshot blender"
     DO_BLENDER=0; DO_OPENSCAD=0 ;;
esac

# Every download goes through this. --http1.1 because an intercepting proxy
# will happily break an HTTP/2 stream mid-transfer (PROTOCOL_ERROR); -C -
# resumes a partial file so a retry is cheap.
fetch() {
  local prog="--silent --show-error"
  [ -t 1 ] && prog="--progress-bar"
  # shellcheck disable=SC2086
  curl -fL --http1.1 --retry 5 --retry-delay 2 --retry-all-errors -C - $prog -o "$2" "$1"
}

# ── 1. Bun ──────────────────────────────────────────────────────────────────
step "1/5  Bun"
if command -v bun >/dev/null 2>&1; then
  ok "bun $(bun --version) — $(command -v bun)"
elif [ "$DO_BUN" = 0 ] || [ "$CHECK_ONLY" = 1 ]; then
  bad "not installed"; MISSING+=("bun")
else
  info "installing from bun.sh…"
  curl -fsSL https://bun.sh/install | bash >/dev/null
  export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
  export PATH="$BUN_INSTALL/bin:$PATH"
  if command -v bun >/dev/null 2>&1; then
    ok "bun $(bun --version) installed"
    warn "add this to your shell profile: export PATH=\"\$HOME/.bun/bin:\$PATH\""
  else
    bad "install failed — see https://bun.sh"; MISSING+=("bun")
  fi
fi

# ── 2. Packages ─────────────────────────────────────────────────────────────
step "2/5  Packages"
if ! command -v bun >/dev/null 2>&1; then
  bad "skipped — bun is not available"
elif [ "$CHECK_ONLY" = 1 ]; then
  [ -d "$REPO/node_modules" ] && ok "root node_modules present" || { bad "root node_modules missing"; MISSING+=("bun install"); }
  [ -d "$REPO/web/node_modules" ] && ok "web node_modules present" || warn "web node_modules missing (Studio only)"
else
  ( cd "$REPO" && bun install --silent ) && ok "pipeline dependencies"
  if [ "$DO_STUDIO" = 1 ]; then
    ( cd "$REPO/web" && bun install --silent ) && ok "Studio dependencies"
  fi
fi

# ── 3. OpenSCAD (must be Manifold-capable) ──────────────────────────────────
# A pre-Manifold build does not fail, it just runs orders of magnitude slower,
# so "is openscad installed" is the wrong question — "does it have --backend" is.
manifold_ok() { [ -x "$1" ] && "$1" --help 2>&1 | grep -q -- "--backend"; }
find_openscad() {
  for c in "${OPENSCAD_PATH:-}" "$PREFIX/openscad" /usr/local/bin/openscad /opt/openscad/openscad "$(command -v openscad 2>/dev/null || true)"; do
    [ -n "$c" ] && manifold_ok "$c" && { echo "$c"; return 0; }
  done
  return 1
}

step "3/5  OpenSCAD (Manifold backend)"
if found="$(find_openscad)"; then
  ok "$found"
  if command -v openscad >/dev/null 2>&1 && ! manifold_ok "$(command -v openscad)"; then
    info "note: $(command -v openscad) on your PATH is pre-Manifold; Procedura will prefer the one above"
  fi
elif [ "$DO_OPENSCAD" = 0 ] || [ "$CHECK_ONLY" = 1 ]; then
  bad "no Manifold-capable build found"; MISSING+=("OpenSCAD")
else
  info "resolving the newest development snapshot…"
  SNAP="$(curl -fsSL https://files.openscad.org/snapshots/ \
          | grep -oE 'OpenSCAD-[0-9]{4}\.[0-9]{2}\.[0-9]{2}-x86_64\.AppImage' | sort -u | tail -1 || true)"
  if [ -z "$SNAP" ]; then
    bad "could not reach files.openscad.org — install a 2023.03+ build yourself and set OPENSCAD_PATH"
    MISSING+=("OpenSCAD")
  else
    mkdir -p "$PREFIX"
    info "$SNAP (~30 MB)"
    if ! fetch "https://files.openscad.org/snapshots/$SNAP" "$PREFIX/openscad.AppImage"; then
      bad "download failed — re-run to resume, or install a 2023.03+ build and set OPENSCAD_PATH"
      MISSING+=("OpenSCAD")
      SNAP=""
    fi
    [ -n "$SNAP" ] && chmod +x "$PREFIX/openscad.AppImage"
    # AppImages need FUSE. Where it is absent (containers, minimal hosts),
    # extract once and point a tiny wrapper at the extracted AppRun instead.
    if [ -z "$SNAP" ]; then :
    elif "$PREFIX/openscad.AppImage" --help >/dev/null 2>&1; then
      ln -sf "$PREFIX/openscad.AppImage" "$PREFIX/openscad"
    else
      info "no FUSE — extracting the AppImage instead"
      ( cd "$PREFIX" && rm -rf openscad.extracted squashfs-root \
        && ./openscad.AppImage --appimage-extract >/dev/null && mv squashfs-root openscad.extracted )
      printf '#!/bin/sh\nexec "%s/openscad.extracted/AppRun" "$@"\n' "$PREFIX" > "$PREFIX/openscad"
      chmod +x "$PREFIX/openscad"
    fi
    if [ -z "$SNAP" ]; then :
    elif manifold_ok "$PREFIX/openscad"; then ok "$PREFIX/openscad"
    else bad "installed but --backend is missing; set OPENSCAD_PATH to a Manifold build"; MISSING+=("OpenSCAD"); fi
  fi
fi

# ── 4. Blender ──────────────────────────────────────────────────────────────
find_blender() {
  for c in "${PROCEDURA_BLENDER_PATH:-}" "$PREFIX/blender/blender" /usr/local/bin/blender /opt/blender/blender "$(command -v blender 2>/dev/null || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

step "4/5  Blender"
if found="$(find_blender)"; then
  ok "$found"
elif [ "$DO_BLENDER" = 0 ] || [ "$CHECK_ONLY" = 1 ]; then
  warn "not found — renders, materials and 3D feedback stay off"; MISSING+=("Blender")
else
  info "resolving the newest release…"
  SERIES="$(curl -fsSL https://download.blender.org/release/ \
            | grep -oE 'Blender[0-9]+\.[0-9]+/' | tr -d '/' | sort -u -V | tail -1 || true)"
  TARBALL=""
  [ -n "$SERIES" ] && TARBALL="$(curl -fsSL "https://download.blender.org/release/$SERIES/" \
      | grep -oE "blender-[0-9.]+-linux-x64\.tar\.xz" | sort -u -V | tail -1 || true)"
  if [ -z "$TARBALL" ]; then
    bad "could not reach download.blender.org — install Blender and set PROCEDURA_BLENDER_PATH"
    MISSING+=("Blender")
  else
    mkdir -p "$PREFIX"
    info "$TARBALL (~350 MB)"
    if ! fetch "https://download.blender.org/release/$SERIES/$TARBALL" "$PREFIX/$TARBALL"; then
      bad "download failed — re-run to resume from where it stopped"
      MISSING+=("Blender")
    else
      info "extracting…"
      rm -rf "$PREFIX/blender"; mkdir -p "$PREFIX/blender"
      tar -xJf "$PREFIX/$TARBALL" -C "$PREFIX/blender" --strip-components=1
      rm -f "$PREFIX/$TARBALL"
      if [ -x "$PREFIX/blender/blender" ]; then ok "$PREFIX/blender/blender"
      else bad "extraction failed"; MISSING+=("Blender"); fi
    fi
  fi
fi

# ── 5. Configuration ────────────────────────────────────────────────────────
step "5/5  Configuration"
if [ "$DO_ENV" = 0 ]; then
  ok "MCP uses the coding agent's model access; no API .env is needed"
elif [ -f "$REPO/.env" ]; then
  ok ".env exists (left untouched)"
elif [ "$CHECK_ONLY" = 1 ]; then
  warn ".env missing — copy .env.example and add your endpoint"
else
  cp "$REPO/.env.example" "$REPO/.env"
  ok "created .env from .env.example"
fi
if [ "$DO_ENV" = 0 ]; then
  :
elif grep -qE '^\s*(OPENAI_API_KEY|GEMINI_API_KEY)=\S' "$REPO/.env" 2>/dev/null; then
  ok "an LLM key is set"
else
  warn "no LLM key yet — add OPENAI_API_KEY (or GEMINI_API_KEY) to .env"
  MISSING+=("LLM key")
fi

# ── summary ─────────────────────────────────────────────────────────────────
printf "\n${B}Summary${N}\n"
if [ ${#MISSING[@]} -eq 0 ]; then
  ok "everything is in place"
  if [ "$DO_ENV" = 1 ]; then
    printf "\n  Try it:\n    ${DIM}bun run scripts/procedura.ts -o outputs/stool --prompt \"a three-legged wooden stool\"${N}\n"
    printf "  Or open the Studio:\n    ${DIM}cd web && bun run start${N}\n\n"
  fi
else
  warn "still missing: ${MISSING[*]}"
  printf "\n  Re-run this script after fixing those, or ${DIM}bash scripts/install-deps.sh --check${N} to re-test.\n\n"
fi

# Isaac is never auto-installed; say so once, at the end, so it is not noise.
if ! [ -x "${PROCEDURA_ISAACSIM_PATH:-$HOME/isaacsim}/python.sh" ]; then
  info "Isaac Sim not found — only needed for --motion validation. See developer.nvidia.com/isaac-sim"
fi
