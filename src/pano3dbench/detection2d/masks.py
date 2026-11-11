from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Interval = tuple[int, int]


@dataclass(frozen=True)
class PanoramaMask:
    height: int
    width: int
    intervals: tuple[Interval, ...]

    @classmethod
    def from_rle(cls, rle: Mapping[str, Any]) -> PanoramaMask:
        height, width = (int(value) for value in rle["size"])
        counts = rle["counts"]
        if isinstance(counts, (str, bytes)):
            raise ValueError("Compressed RLE is not supported; pass uncompressed counts")
        intervals: list[Interval] = []
        position = 0
        for index, count in enumerate(counts):
            count = int(count)
            if count < 0:
                raise ValueError("RLE counts must not be negative")
            if index % 2 == 1 and count:
                intervals.append((position, position + count))
            position += count
        if position != height * width:
            raise ValueError(
                f"RLE covers {position} pixels, the image has {height * width}"
            )
        return cls(height, width, tuple(intervals))

    def to_rle(self) -> dict[str, Any]:
        counts: list[int] = []
        position = 0
        for start, end in self.intervals:
            counts.extend((start - position, end - start))
            position = end
        counts.append(self.height * self.width - position)
        return {"size": [self.height, self.width], "counts": counts}

    @property
    def area(self) -> int:
        return sum(end - start for start, end in self.intervals)

    def rolled(self, columns: int) -> PanoramaMask:
        total = self.height * self.width
        shift = (columns % self.width) * self.height
        pieces: list[Interval] = []
        for start, end in self.intervals:
            start, end = start + shift, end + shift
            if start >= total:
                pieces.append((start - total, end - total))
            elif end > total:
                pieces.extend(((start, total), (0, end - total)))
            else:
                pieces.append((start, end))
        return PanoramaMask(self.height, self.width, _merged(pieces))

    def union(self, other: PanoramaMask) -> PanoramaMask:
        self._require_same_size(other)
        return PanoramaMask(
            self.height, self.width, _merged((*self.intervals, *other.intervals))
        )

    def intersection_area(self, other: PanoramaMask) -> int:
        self._require_same_size(other)
        first, second = self.intervals, other.intervals
        i = j = total = 0
        while i < len(first) and j < len(second):
            low = max(first[i][0], second[j][0])
            high = min(first[i][1], second[j][1])
            if high > low:
                total += high - low
            if first[i][1] <= second[j][1]:
                i += 1
            else:
                j += 1
        return total

    def iou(self, other: PanoramaMask) -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def occupied_columns(self) -> tuple[Interval, ...]:
        return _merged(
            (start // self.height, (end - 1) // self.height + 1)
            for start, end in self.intervals
        )

    def row_range(self) -> Interval | None:
        if not self.intervals:
            return None
        low, high = self.height, -1
        for start, end in self.intervals:
            if start // self.height == (end - 1) // self.height:
                low = min(low, start % self.height)
                high = max(high, (end - 1) % self.height)
            else:
                low, high = 0, self.height - 1
        return low, high

    def _require_same_size(self, other: PanoramaMask) -> None:
        if (self.height, self.width) != (other.height, other.width):
            raise ValueError("Masks of different image sizes cannot be compared")


def _merged(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    result: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


@dataclass(frozen=True)
class PanoramaBox:
    x: int
    y: int
    width: int
    height: int
    image_width: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def crosses_seam(self) -> bool:
        return self.x + self.width > self.image_width

    def iou(self, other: PanoramaBox) -> float:
        overlap_x = min(
            min(self.width, other.width),
            sum(
                max(
                    0,
                    min(self.x + self.width, other.x + shift + other.width)
                    - max(self.x, other.x + shift),
                )
                for shift in (-self.image_width, 0, self.image_width)
            ),
        )
        overlap_y = max(
            0,
            min(self.y + self.height, other.y + other.height) - max(self.y, other.y),
        )
        intersection = overlap_x * overlap_y
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def box_of(mask: PanoramaMask) -> PanoramaBox | None:
    rows = mask.row_range()
    if rows is None:
        return None
    columns = mask.occupied_columns()
    x, width = _covering_arc(columns, mask.width)
    return PanoramaBox(x, rows[0], width, rows[1] - rows[0] + 1, mask.width)


def _covering_arc(columns: Sequence[Interval], width: int) -> Interval:
    gaps: list[Interval] = []
    for (_, end), (start, _) in zip(columns, columns[1:]):
        gaps.append((end, start - end))
    wrap = columns[0][0] + width - columns[-1][1]
    gaps.append((columns[-1][1] % width, wrap))
    gap_start, gap_length = max(gaps, key=lambda gap: (gap[1], -gap[0]))
    if gap_length <= 0:
        return 0, width
    return (gap_start + gap_length) % width, width - gap_length
