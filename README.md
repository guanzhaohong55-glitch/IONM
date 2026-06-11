# IONM JSON Waveform Viewer

Qt desktop viewer for Cadwell-style intraoperative neurophysiological monitoring
(IONM) JSON exports.

The app scans local case JSON files, parses waveform trials, converts raw sample
values into physical voltage, and visualizes each trial with event timing.

## Modules

- `ionm_viewer.py`: desktop waveform viewer for Cadwell-style IONM JSON exports.
- `anesthesia_record_extraction/`: anesthesia-record PDF/image extraction tools for blood pressure, heart rate, SpO2, EtCO2, and manual review packages.

## Data Structure

The expected export shape is:

- `Cases`: case list. The current dataset usually has one case per JSON file.
- `Cases[].Events`: surgical timeline events with `Timestamp`, `Type`,
  `Message`, and `Deleted`.
- `Cases[].Modes`: monitoring modes such as Lower SSEP, MEP, TOF, and DNEP.
- `Modes[].Trials`: repeated acquisitions/stimulation trials for a mode.
- `Trials[].Traces`: channel waveforms for one trial.

Important trace fields:

- `TraceData`: raw waveform samples.
- `TraceDataScalar`: multiplier that converts raw samples to volts.
- `TraceDataLength`: sample count, commonly 640 in this dataset.
- `Sweep`: response window length in seconds, commonly 0.1s for SSEP/MEP and
  0.02s for TOF.
- `Channel`: channel or lead information.
- `Cursors`: latency/amplitude markers such as `N37`, `P45`, `P`, and `T`.

Voltage conversion:

```text
voltage_v = TraceData * TraceDataScalar
```

The viewer displays SSEP-like traces in `uV`/`μV` scale and MEP/EMG/TOF traces
in `mV` scale. The x-axis response window is calculated from `Sweep` and
`TraceDataLength`.

## Run

Recommended on Windows:

```powershell
.\run_viewer.bat
```

Manual setup:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ionm_viewer.py
```

For anesthesia-record extraction, see `anesthesia_record_extraction/README.md`.

## Privacy

Do not publish identifiable medical data. The repository `.gitignore` excludes
patient exports and generated runtime files by default, including JSON, PDF,
Excel, Word, image exports, `.exportzip`, and `.venv`.
