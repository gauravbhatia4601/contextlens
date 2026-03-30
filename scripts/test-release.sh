#!/bin/bash
# Test release to TestPyPI

set -e

echo "🧪 Testing release on TestPyPI..."
echo ""

# Build package
echo "📦 Building package..."
rm -rf dist/ build/
python -m build

# Upload to TestPyPI
echo "📤 Uploading to TestPyPI..."
twine upload --repository testpypi dist/*

echo ""
echo "✅ Uploaded to TestPyPI!"
echo ""
echo "Test installation:"
echo "  pip install --index-url https://test.pypi.org/simple/ contextlens"
echo ""
