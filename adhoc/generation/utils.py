from __future__ import annotations

from datetime import datetime
from pathlib import Path

from adhoc.utils.repo_paths import results_subdir


def announce_output(path: str | Path, count: int | None, kind: str = "gif") -> None:
    p = Path(path)
    if count is None or count < 0:
        print(f"'{p}' 폴더에 생성될 {kind} 개수는 실행 중 계산됩니다.")
    else:
        print(f"'{p}' 폴더에 {count}개의 {kind}가 저장될 것입니다.")


def dated_results_html_path(html_subdir: str, stem: str) -> Path:
    """One index HTML per day under data/results/html/{subdir}/ (overwrites same calendar day)."""
    d = results_subdir("html") / html_subdir
    d.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    return d / f"{stem}_{day}.html"
