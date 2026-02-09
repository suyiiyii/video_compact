#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TAG="autobuild-2026-02-08-12-58"
DEFAULT_ASSET=""
DEFAULT_DEST=".tools/ffmpeg-libvmaf"

tag="${DEFAULT_TAG}"
asset="${DEFAULT_ASSET}"
dest="${DEFAULT_DEST}"

usage() {
  cat <<'EOF'
Install an FFmpeg binary with libvmaf support from BtbN releases.

Usage:
  ./scripts/install_ffmpeg_libvmaf.sh [--tag TAG] [--asset ASSET] [--dest DIR]

Defaults:
  --tag    autobuild-2026-02-08-12-58
  --asset  (optional) exact asset name in selected tag
  --dest   .tools/ffmpeg-libvmaf

After install, export:
  VIDEO_COMPACT_FFMPEG_BIN=<...>/bin/ffmpeg
  VIDEO_COMPACT_FFPROBE_BIN=<...>/bin/ffprobe
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="$2"
      shift 2
      ;;
    --asset)
      asset="$2"
      shift 2
      ;;
    --dest)
      dest="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer currently supports Linux only." >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This installer currently supports x86_64 only." >&2
  exit 1
fi

tmp_dir="$(mktemp -d -t ffmpeg-libvmaf-XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

mapfile -t release_info < <(python3 - "${tag}" "${asset}" <<'PY'
import json
import re
import sys
import urllib.request

tag = sys.argv[1]
requested_asset = sys.argv[2].strip()
api = f"https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/{tag}"
with urllib.request.urlopen(api) as resp:
    release = json.load(resp)

assets = release.get("assets", [])
if not assets:
    raise SystemExit(f"No assets found for tag: {tag}")

selected = None
if requested_asset:
    for item in assets:
        if item.get("name") == requested_asset:
            selected = item
            break
    if selected is None:
        raise SystemExit(f"Asset not found in tag {tag}: {requested_asset}")
else:
    patterns = [
        r"^ffmpeg-N-.*-linux64-gpl\.tar\.xz$",
        r"^ffmpeg-master-latest-linux64-gpl\.tar\.xz$",
        r".*-linux64-gpl\.tar\.xz$",
    ]
    for pattern in patterns:
        for item in assets:
            name = item.get("name", "")
            if re.match(pattern, name):
                selected = item
                break
        if selected is not None:
            break
    if selected is None:
        raise SystemExit(f"No linux64 gpl tar.xz asset found in tag {tag}")

checksums = next((a for a in assets if a.get("name") == "checksums.sha256"), None)
checksums_url = checksums["browser_download_url"] if checksums else (
    f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{tag}/checksums.sha256"
)

print(selected["name"])
print(selected["browser_download_url"])
print(checksums_url)
PY
)

asset="${release_info[0]}"
asset_url="${release_info[1]}"
checksums_url="${release_info[2]}"

echo "Downloading ${asset_url}"
curl -fsSL "${asset_url}" -o "${tmp_dir}/${asset}"
curl -fsSL "${checksums_url}" -o "${tmp_dir}/checksums.sha256"

if ! grep -E "[[:space:]]${asset}$" "${tmp_dir}/checksums.sha256" > "${tmp_dir}/asset.sha256"; then
  echo "Checksum entry for ${asset} not found in checksums.sha256" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${tmp_dir}" && sha256sum -c asset.sha256)
elif command -v shasum >/dev/null 2>&1; then
  expected="$(awk '{print $1}' "${tmp_dir}/asset.sha256")"
  actual="$(shasum -a 256 "${tmp_dir}/${asset}" | awk '{print $1}')"
  if [[ "${expected}" != "${actual}" ]]; then
    echo "Checksum mismatch for ${asset}" >&2
    exit 1
  fi
else
  echo "Missing checksum tool: need sha256sum or shasum" >&2
  exit 1
fi

mkdir -p "${dest}"
tar -xf "${tmp_dir}/${asset}" -C "${dest}"

install_root="$(find "${dest}" -maxdepth 1 -type d -name 'ffmpeg-*' | sort | tail -n 1)"
if [[ -z "${install_root}" ]]; then
  echo "Unable to locate extracted ffmpeg directory in ${dest}" >&2
  exit 1
fi

ffmpeg_bin="${install_root}/bin/ffmpeg"
ffprobe_bin="${install_root}/bin/ffprobe"
if [[ ! -x "${ffmpeg_bin}" || ! -x "${ffprobe_bin}" ]]; then
  echo "ffmpeg/ffprobe binary not found after extraction" >&2
  exit 1
fi

if ! "${ffmpeg_bin}" -hide_banner -filters | grep -q "libvmaf"; then
  echo "Downloaded ffmpeg does not include libvmaf filter" >&2
  exit 1
fi

echo
echo "Installed to: ${install_root}"
echo "libvmaf filter: OK"
echo
echo "Use in current shell:"
echo "  export VIDEO_COMPACT_FFMPEG_BIN=\"${ffmpeg_bin}\""
echo "  export VIDEO_COMPACT_FFPROBE_BIN=\"${ffprobe_bin}\""
