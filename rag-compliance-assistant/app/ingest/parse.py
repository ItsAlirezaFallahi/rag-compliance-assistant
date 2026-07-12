"""PDF -> list of Sections, using each document's own heading structure.

This is the "structure-aware" half of structure-aware chunking. Instead of
slicing the document into fixed 500-token windows (which routinely cuts a
regulatory requirement in half and glues unrelated topics together), we:

  1. Extract raw text page by page (pypdf).
  2. Normalize whitespace and strip page-number artifacts.
  3. Walk the lines and start a new Section whenever a line matches the
     document's heading pattern.

Each document gets a DocProfile because regulators don't share a style
guide: SR 11-7 uses Roman-numeral headings ("V. Model Validation"), while
NIST AI RMF 1.0 uses decimal-numbered headings ("5.1 Govern") plus
appendices.

Honest caveat (worth saying in an interview too): heading detection from
extracted PDF text is heuristic. Run `python -m app.cli inspect <slug>`
after ingestion to eyeball the detected sections and tune the regexes.
"""

import re
from dataclasses import dataclass, field

from pypdf import PdfReader


@dataclass
class Section:
    section_path: str      # short breadcrumb used in metadata/citations
    title: str
    text: str


@dataclass
class DocProfile:
    slug: str
    title: str
    source_url: str
    # A line matching any of these regexes starts a new section.
    heading_patterns: list[re.Pattern] = field(default_factory=list)
    # Lines matching these are dropped entirely (headers/footers/noise).
    noise_patterns: list[re.Pattern] = field(default_factory=list)


PROFILES: dict[str, DocProfile] = {
    "sr-11-7": DocProfile(
        slug="sr-11-7",
        title="SR 11-7: Supervisory Guidance on Model Risk Management",
        source_url="https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf",
        heading_patterns=[
            # "I. Introduction", "V. Model Validation", "VIII. Conclusion"
            re.compile(r"^(?P<num>[IVX]{1,5})\.\s+(?P<title>[A-Z][^\n]{2,80})$"),
        ],
        noise_patterns=[
            re.compile(r"^Page \d+ of \d+$", re.I),
            re.compile(r"^\d{1,3}$"),  # bare page numbers
        ],
    ),
    "nist-ai-rmf": DocProfile(
        slug="nist-ai-rmf",
        title="NIST AI Risk Management Framework (AI RMF 1.0)",
        source_url="https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        heading_patterns=[
            # "1 Framing Risk", "1.2.1 Risk Measurement", "5.1 Govern"
            re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{2,80})$"),
            # "Part 1: Foundational Information", "Appendix A: ..."
            re.compile(r"^(?P<num>Part \d|Appendix [A-D])[:.]?\s*(?P<title>[^\n]{0,80})$"),
        ],
        noise_patterns=[
            re.compile(r"^NIST AI 100-1.*$", re.I),
            re.compile(r"^AI Risk Management Framework$", re.I),
            re.compile(r"^\d{1,3}$"),
        ],
    ),
}


def extract_lines(pdf_path: str, profile: DocProfile) -> list[str]:
    reader = PdfReader(pdf_path)
    lines: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            if any(p.match(line) for p in profile.noise_patterns):
                continue
            lines.append(line)
    return lines


def match_heading(line: str, profile: DocProfile) -> str | None:
    """Return the normalized heading string if `line` is a heading."""
    for pattern in profile.heading_patterns:
        m = pattern.match(line)
        if m:
            num = m.groupdict().get("num", "").strip()
            title = (m.groupdict().get("title") or "").strip().rstrip(".")
            return f"{num}. {title}".strip(". ") if title else num
    return None


def parse_pdf(pdf_path: str, slug: str) -> list[Section]:
    profile = PROFILES[slug]
    lines = extract_lines(pdf_path, profile)

    sections: list[Section] = []
    current_title = "Front Matter"
    buffer: list[str] = []

    def flush():
        text = " ".join(buffer).strip()
        if text:
            sections.append(
                Section(section_path=current_title, title=current_title, text=text)
            )

    for line in lines:
        heading = match_heading(line, profile)
        if heading:
            flush()
            current_title = heading
            buffer = []
        else:
            buffer.append(line)
    flush()

    return sections
