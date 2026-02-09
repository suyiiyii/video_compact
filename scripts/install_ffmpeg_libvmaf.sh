#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TAG="autobuild-2026-02-08-12-58"
DEFAULT_ASSET="ffmpeg-master-latest-linux64-gpl.tar.xz"
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
  --asset  ffmpeg-master-latest-linux64-gpl.tar.xz
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

base_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/${tag}"
asset_url="${base_url}/${asset}"
checksums_url="${base_url}/checksums.sha256"

tmp_dir="$(mktemp -d -t ffmpeg-libvmaf-XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

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
