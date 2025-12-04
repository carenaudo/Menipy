# Results Tab Enhancement Plan

## Overview
Transform the results panel from a single-measurement display to a multi-measurement table where files/measurements are rows and results are columns, supporting measurements from different pipeline types.

## Current State Analysis

**Results Storage:**
- Results stored in `Context.results` as key-value pairs
- Each pipeline run overwrites previous results (single measurement mode)
- No historical data persistence

**Results Display:**
- `ResultsPanel` uses 2-column table (Parameter | Value)
- Single measurement display only
- No file/measurement identifiers

**Data Sources for Identifiers:**
- `Context.image_path` - file path if loaded from file
- `Context.timings_ms` - could include pipeline start time
- System timestamp when pipeline completes
- Sequential measurement numbers

## Pipeline-Specific Result Sets

**Sessile Drop Results:**
- `diameter_mm`, `height_mm`, `volume_uL`, `contact_angle_deg`, `surface_tension_mN_m`
- `theta_left_deg`, `theta_right_deg`, `contact_surface_mm2`, `drop_surface_mm2`
- `baseline_tilt_deg`, `method`, `uncertainty_deg`

**Pendant Drop Results:**
- `diameter_mm`, `height_mm`, `volume_uL`, `surface_tension_mN_m`
- `beta`, `s1`, `r0_mm`, `needle_surface_mm2`, `drop_surface_mm2`

**Oscillating Drop Results:**
- `R0_mm`, `f0_Hz`, `r0_eq_px`, `residuals`

**Capillary Rise Results:**
- TBD (need to examine implementation)

## Proposed Architecture

### 1. New Data Structures

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

class MeasurementResult(BaseModel):
    id: str  # "20251028_143052_001"
    timestamp: datetime
    pipeline: str  # "sessile", "pendant", "oscillating"
    file_path: Optional[str]
    file_name: Optional[str]  # Display name
    results: Dict[str, Any]  # Raw results dict from pipeline

class ResultsHistory:
    def __init__(self):
        self.measurements: List[MeasurementResult] = []
        self.max_history = 100  # Configurable limit

    def add_measurement(self, measurement: MeasurementResult):
        self.measurements.insert(0, measurement)  # Most recent first
        if len(self.measurements) > self.max_history:
            self.measurements.pop()

    def get_table_data(self, pipeline_filter: Optional[str] = None) -> Tuple[List[str], List[List[Any]]]:
        """Return (headers, rows) for table display, optionally filtered by pipeline"""
        # Implementation details in ResultsPanel section
```

### 2. Dynamic Column Management

Instead of fixed columns, the results table will:

1. **Auto-detect columns** from all stored measurements
2. **Pipeline-filtered views** - show only relevant columns for current pipeline
3. **Column prioritization** - common metrics first, then pipeline-specific
4. **Missing data handling** - show empty cells for measurements without certain metrics

### 3. Enhanced ResultsPanel

- **Dynamic column generation** based on available measurements
- **Pipeline filtering** dropdown/button to show only relevant columns
- **Column sorting** and reordering
- **Export options** - CSV with all data, filtered views
- **Measurement details** - click row to see full results

### 4. Pipeline Controller Integration

- Generate unique measurement IDs (timestamp-based)
- Extract file identifiers from context
- Store results in history manager
- Update UI with new table format

### 5. Data Persistence

- Store all measurements regardless of pipeline
- JSON format with measurement metadata
- Automatic cleanup of old measurements (configurable limit)

## Implementation Plan

### Phase 1: Core Data Structures
1. Create `MeasurementResult` and `ResultsHistory` classes
2. Add to `menipy.models.results` module
3. Create persistence layer (JSON file storage)

### Phase 2: Results Panel Redesign
1. Modify `ResultsPanel.update()` to accept measurement history
2. Implement multi-column table layout with dynamic columns
3. Add pipeline filtering controls
4. Implement export functionality

### Phase 3: Pipeline Integration
1. Update `PipelineController` to collect measurement metadata
2. Generate measurement IDs and timestamps
3. Store results in history manager
4. Update UI callbacks

### Phase 4: UI Enhancements
1. Add clear history/export buttons to results panel
2. Update status messages to show measurement count
3. Add measurement numbering in overlay

## Key Design Decisions

**Measurement ID Format:** `{timestamp}_{sequence:03d}`
- Example: `20251028_143052_001`

**File Identifier Priority:**
1. `Context.image_path` basename if available
2. "Live Capture" + timestamp for camera input
3. "Measurement {sequence}" as fallback

**Column Ordering Priority:**
1. **Universal metrics:** File/Time, Pipeline, Diameter, Height, Volume
2. **Common physics:** Surface Tension, Contact Angle
3. **Pipeline-specific:** Beta, S1, F0, etc.
4. **Metadata:** Method, Uncertainties, Confidence scores

**Pipeline Filtering:**
- Default: Show all measurements with all available columns
- Filtered: Show only measurements from selected pipeline with relevant columns
- Smart defaults: Hide empty columns when filtering

**Data Persistence:**
- Store all measurements regardless of pipeline
- JSON format with measurement metadata
- Automatic cleanup of old measurements (configurable limit)

## UI Controls

- Pipeline filter dropdown
- Clear history button
- Export current view button
- Measurement count display
- Column show/hide options

## Migration Strategy

1. **Backward Compatibility:** Keep existing single-measurement display as fallback
2. **Gradual Rollout:** Add history feature alongside existing functionality
3. **Data Migration:** Convert existing single results to measurement history format
4. **User Preferences:** Allow users to switch between single/multi-measurement views

## Testing Strategy

1. **Unit Tests:** Test data structures and history management
2. **Integration Tests:** Test pipeline integration and UI updates
3. **User Acceptance:** Verify usability with different pipeline types
4. **Performance:** Test with large measurement histories (100+ measurements)

## Success Criteria

- Users can view multiple measurements in a table format
- Results from different pipelines display correctly
- Export functionality works for analysis in external tools
- Performance remains acceptable with measurement history
- UI is intuitive and doesn't confuse existing workflows