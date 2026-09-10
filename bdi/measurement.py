"""Calibration validation and geometry measurement.

Physical values are deliberately unavailable unless the supplied calibration is
explicitly valid.  The module supports either an isotropic local scale or a
pixel-to-plane homography; it never attempts to infer scale from image content.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


VALID_METHODS = {
    "reference_marker",
    "camera_planar_calibration",
    "UAV_GSD",
    "registered_3d",
}


@dataclass(frozen=True)
class Calibration:
    method: str
    unit: str
    uncertainty: float
    millimeters_per_pixel: float | None = None
    homography_pixel_to_mm: tuple[tuple[float, float, float], ...] | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Calibration":
        if value.get("scale_status") != "valid":
            raise ValueError("Calibration scale_status must be 'valid'")
        method = str(value.get("scale_method", ""))
        if method not in VALID_METHODS:
            raise ValueError(f"Unsupported valid scale method: {method!r}")
        unit = str(value.get("unit", "mm"))
        if unit not in {"mm", "cm"}:
            raise ValueError("Calibration unit must be 'mm' or 'cm'")
        uncertainty = _positive_number(value.get("uncertainty"), "uncertainty", allow_zero=True)

        scale = value.get("millimeters_per_pixel")
        homography = value.get("homography_pixel_to_mm")
        if (scale is None) == (homography is None):
            raise ValueError(
                "A valid calibration must provide exactly one of "
                "millimeters_per_pixel or homography_pixel_to_mm"
            )
        if scale is not None:
            scale = _positive_number(scale, "millimeters_per_pixel")
            matrix = None
        else:
            if not isinstance(homography, list) or len(homography) != 3:
                raise ValueError("homography_pixel_to_mm must be a 3x3 array")
            rows: list[tuple[float, float, float]] = []
            for row in homography:
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError("homography_pixel_to_mm must be a 3x3 array")
                numbers = tuple(float(item) for item in row)
                if not all(math.isfinite(item) for item in numbers):
                    raise ValueError("Homography entries must be finite")
                rows.append(numbers)
            matrix = tuple(rows)
            if abs(_determinant_3x3(matrix)) < 1e-12:
                raise ValueError("homography_pixel_to_mm must be non-singular")
            scale = None

        if method == "camera_planar_calibration" and matrix is None:
            raise ValueError("camera_planar_calibration requires a homography")
        return cls(method, unit, uncertainty, scale, matrix)


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    lower_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not lower_ok:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {comparator} number")
    return number


def _determinant_3x3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _transform(
    point: tuple[float, float], matrix: tuple[tuple[float, float, float], ...]
) -> tuple[float, float]:
    x, y = point
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-12:
        raise ValueError("Box intersects the homography horizon")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _polygon_area(points: Iterable[tuple[float, float]]) -> float:
    values = list(points)
    return abs(
        sum(
            values[index][0] * values[(index + 1) % len(values)][1]
            - values[(index + 1) % len(values)][0] * values[index][1]
            for index in range(len(values))
        )
    ) / 2.0


def measure_box(
    box_xywh: list[float], calibration: Calibration | None = None
) -> dict[str, Any]:
    """Return schema-compatible pixel and, when valid, physical measurements."""
    if len(box_xywh) != 4:
        raise ValueError("box_xywh must have four values")
    x, y, width, height = (float(item) for item in box_xywh)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("box_xywh must have non-negative origin and positive dimensions")

    result: dict[str, Any] = {
        "scale_status": "not_provided",
        "scale_method": "none",
        "pixel": {
            "length": round(max(width, height), 4),
            "max_width": round(min(width, height), 4),
            "area": round(width * height, 4),
        },
    }
    if calibration is None:
        return result

    if calibration.millimeters_per_pixel is not None:
        scale = calibration.millimeters_per_pixel
        length_mm = max(width, height) * scale
        width_mm = min(width, height) * scale
        area_mm2 = width * height * scale * scale
    else:
        assert calibration.homography_pixel_to_mm is not None
        corners_px = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        corners_mm = [
            _transform(point, calibration.homography_pixel_to_mm) for point in corners_px
        ]
        sides = [
            _distance(corners_mm[index], corners_mm[(index + 1) % 4])
            for index in range(4)
        ]
        length_mm = max(sides)
        width_mm = min(sides)
        area_mm2 = _polygon_area(corners_mm)

    divisor = 10.0 if calibration.unit == "cm" else 1.0
    result.update(
        {
            "scale_status": "valid",
            "scale_method": calibration.method,
            "physical": {
                "unit": calibration.unit,
                "length": round(length_mm / divisor, 4),
                "max_width": round(width_mm / divisor, 4),
                "area": round(area_mm2 / (divisor * divisor), 4),
                "uncertainty": round(calibration.uncertainty / divisor, 4),
            },
        }
    )
    return result

