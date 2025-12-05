#!/usr/bin/env bash
set -euo pipefail

# Package source code path
PACKAGE_PATH="mtopt"

# Sphinx docs directories
DOCS_SOURCE_DIR="docs/source"
DOCS_BUILD_DIR="docs/_build/html"
API_OUTPUT_SUBDIR="api"   # generated .rst files go into docs/source/api

# Python binary to use (can override by: PYTHON=python3 ./generate_docs.sh)
PYTHON_BIN="${PYTHON:-python}"

# Derived paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PACKAGE_ABS="${PROJECT_ROOT}/${PACKAGE_PATH}"
DOCS_SOURCE_ABS="${PROJECT_ROOT}/${DOCS_SOURCE_DIR}"
DOCS_BUILD_ABS="${PROJECT_ROOT}/${DOCS_BUILD_DIR}"
API_OUTPUT_DIR="${DOCS_SOURCE_ABS}/${API_OUTPUT_SUBDIR}"

# Sanity checks
if [[ ! -d "${PACKAGE_ABS}" ]]; then
    echo "ERROR: Package directory not found: ${PACKAGE_ABS}"
    echo "       Please set PACKAGE_PATH correctly at the top of this script."
    exit 1
fi

if [[ ! -f "${DOCS_SOURCE_ABS}/conf.py" ]]; then
    echo "ERROR: Sphinx conf.py not found in ${DOCS_SOURCE_ABS}."
    echo "       Run 'sphinx-quickstart docs' first and configure conf.py."
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import sphinx" >/dev/null 2>&1; then
    echo "ERROR: Sphinx does not seem to be installed for '${PYTHON_BIN}'."
    echo "       Try: ${PYTHON_BIN} -m pip install sphinx sphinx-autodoc-typehints"
    exit 1
fi

# Generate .rst API docs with sphinx-apidoc
echo "==> Generating .rst API docs from ${PACKAGE_PATH} into ${API_OUTPUT_DIR}"
mkdir -p "${API_OUTPUT_DIR}"

# Optionally clean previous generated .rst files in that directory
find "${API_OUTPUT_DIR}" -maxdepth 1 -name "*.rst" -delete || true

# Generate API docs (.rst files)
"${PYTHON_BIN}" -m sphinx.ext.apidoc \
    -o "${API_OUTPUT_DIR}" \
    "${PACKAGE_ABS}" \
    -f

# Build HTML docs with sphinx-build
echo "==> Building HTML docs into ${DOCS_BUILD_ABS}"
mkdir -p "${DOCS_BUILD_ABS}"
"${PYTHON_BIN}" -m sphinx \
    -b html \
    "${DOCS_SOURCE_ABS}" \
    "${DOCS_BUILD_ABS}"
echo "==> Done."
echo "    - Generated .rst files: ${API_OUTPUT_DIR}"
echo "    - HTML documentation:   ${DOCS_BUILD_ABS}"
