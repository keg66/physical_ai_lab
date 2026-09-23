#!/usr/bin/env bash
# Set up the local Kev System One server used by kev_mini_lab.py.
#
# Usage:
#   ./system1/setup_kev.sh
#   ./system1/setup_kev.sh --serve
#
# Optional environment variables:
#   KEV_DIR       Checkout location (default: system1/.kev)
#   KEV_REF       Git ref to install (default: main)
#   KEV_PYTHON    Python version for Kev (default: 3.13)
#   KEV_RUN       Model/checkpoint (default: jaredpalmer/kev-4b)
#   KEV_PORT      Local server port (default: 8009)
#   KEV_DTYPE     Inference dtype, e.g. bf16 or fp32 (default: bf16)

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
kev_dir="${KEV_DIR:-$script_dir/.kev}"
tools_dir="$script_dir/.tools"
kev_ref="${KEV_REF:-main}"
kev_python="${KEV_PYTHON:-3.13}"
kev_run="${KEV_RUN:-jaredpalmer/kev-0.8b}"
kev_port="${KEV_PORT:-8009}"
kev_dtype="${KEV_DTYPE:-bf16}"

usage() {
    sed -n '2,14p' "$0"
}

serve=false
case "${1:-}" in
    "") ;;
    --serve) serve=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
esac

for command in git curl; do
    command -v "$command" >/dev/null || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done

if command -v uv >/dev/null; then
    uv_cmd="$(command -v uv)"
else
    uv_cmd="$tools_dir/uv"
    if [[ ! -x "$uv_cmd" ]]; then
        echo "Installing uv in $tools_dir ..."
        mkdir -p "$tools_dir"
        # This is Astral's official installer.  Keeping it in system1/.tools
        # avoids modifying the user's shell profile or global PATH.
        curl -LsSf https://astral.sh/uv/install.sh | \
            env UV_INSTALL_DIR="$tools_dir" UV_NO_MODIFY_PATH=1 sh
    fi
fi

if [[ -e "$kev_dir" && ! -d "$kev_dir/.git" ]]; then
    echo "KEV_DIR exists but is not a Kev git checkout: $kev_dir" >&2
    exit 1
fi

if [[ -d "$kev_dir/.git" ]]; then
    echo "Updating Kev ($kev_ref) ..."
    git -C "$kev_dir" fetch --depth 1 origin "$kev_ref"
    git -C "$kev_dir" checkout --detach FETCH_HEAD
else
    echo "Cloning Kev ($kev_ref) ..."
    git clone --depth 1 --branch "$kev_ref" \
        https://github.com/jaredpalmer/kev.git "$kev_dir"
fi

echo "Installing Kev server dependencies with Python $kev_python ..."
"$uv_cmd" python install "$kev_python"
"$uv_cmd" sync --directory "$kev_dir" --extra serve --python "$kev_python"

echo
echo "Kev is ready. Start the server in one terminal:"
echo "  $0 --serve"
echo "Then, from another terminal, run:"
echo "  python $project_dir/system1/kev_mini_lab.py"

if [[ "$serve" == true ]]; then
    echo "Starting Kev on http://127.0.0.1:$kev_port ..."
    exec env KEV_DTYPE="$kev_dtype" \
        "$uv_cmd" run --directory "$kev_dir" --extra serve \
        python -m kev.serve --run "$kev_run" --port "$kev_port"
fi
