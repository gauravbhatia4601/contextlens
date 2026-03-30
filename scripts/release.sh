#!/bin/bash
# Release script for ContextLens PyPI package

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.3.0"
    exit 1
fi

echo "🚀 Releasing ContextLens v$VERSION"
echo ""

# Check we're on main branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "❌ Error: Must be on main branch to release"
    exit 1
fi

# Update version in pyproject.toml
echo "📝 Updating version to $VERSION..."
sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml

# Commit version bump
git add pyproject.toml
git commit -m "Bump version to $VERSION for release"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo ""
echo "📦 Building package..."
rm -rf dist/ build/
python -m build

echo ""
echo "🧪 Testing installation..."
pip install --force-reinstall dist/*.whl >/dev/null 2>&1
if command -v contextlens &> /dev/null; then
    echo "✅ Installation test passed"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "🚀 Ready to publish to PyPI!"
echo ""
echo "Next steps:"
echo "1. Review changes: git show HEAD"
echo "2. Push to GitHub: git push && git push --tags"
echo "3. Publish to TestPyPI: twine upload --repository testpypi dist/*"
echo "4. Test from TestPyPI: pip install --index-url https://test.pypi.org/simple/ contextlens"
echo "5. Publish to PyPI: twine upload dist/*"
echo ""
echo "⚠️  Do NOT push until you've tested locally!"
