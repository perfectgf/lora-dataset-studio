"""Plugin source unit tests: no LDS app or personal runtime is booted."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Pure graph and frame-cache contracts use the real public SDK, without
# constructing an app. Standalone plugin collection does not load backend's
# conftest, so expose that package explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'backend'))
