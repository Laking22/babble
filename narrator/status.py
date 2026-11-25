"""Reading progress out of status text ("Line 3 of 20", "Chapter 2 of 5", "page 1 of 4", "40%"), for the page's
progress bars and the API's task progress alike."""
import re


def fraction(text):
    """How far along a status message says it is (0..1), or None."""
    head = (text or "").split("\n", 1)[0]
    if head.startswith("Done"):
        return 1.0
    num = lambda s: int(s.replace(",", ""))  # noqa: E731
    line = re.search(r"Line ([\d,]+) of ([\d,]+)", head) or re.search(r"Speaking \(([\d,]+)/([\d,]+)\)", head)
    inner = max(0.0, min(1.0, (num(line[1]) - 1) / num(line[2]))) if line and num(line[2]) else None
    chapter = re.search(r"Chapter (\d+)\s*(?:of|/)\s*(\d+)", head)
    if chapter and int(chapter[2]):
        if "already made" in head or "Reusing" in head:
            inner = 1.0
        return max(0.0, min(1.0, (int(chapter[1]) - 1 + (inner or 0.0)) / int(chapter[2])))
    if inner is not None:
        return inner
    m = re.search(r"([\d,]+)\s*(?:of|/)\s*([\d,]+)", head)
    if m and num(m[2]):
        return max(0.0, min(1.0, num(m[1]) / num(m[2])))
    m = re.search(r"(\d+(?:\.\d+)?)\s?%", head)
    return max(0.0, min(1.0, float(m[1]) / 100)) if m else None


class Status(str):
    """A status line that also says how far along it is (0..1), when its text alone can't: "Stitching the audio
    together" has no numbers in it. Used as a plain string everywhere else."""
    def __new__(cls, text, progress=None):
        s = super().__new__(cls, text)
        s.progress = progress
        return s


def progress_of(status):
    """How far along a status is: what it says it is (Status), else what its text says."""
    p = getattr(status, "progress", None)
    return p if p is not None else fraction(status)
