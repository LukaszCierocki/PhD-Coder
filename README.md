# PhD Coder: Audio Steganography Toolkit

## Project Overview
PhD Coder is an advanced toolkit for experimenting, testing, and optimizing steganographic methods in audio signals. It enables automated testing of various steganographic algorithms, analysis of their effectiveness and robustness against attacks.

## Main Features
- **Steganography Method Testing**: Automated tests on various audio carriers and with different parameters.
- **Result Analysis**: Automatic analysis of method effectiveness, report generation, charts, and statistics.
- **Attacks on Steganography**: Testing robustness against common attacks (noise, filtering, MP3 compression, etc.).
- **Support for Multiple Methods**: Includes various algorithms (SVD, MDCT, DCT, etc.) and their variants.

## Directory Structure (Server View)
```
PhD Coder/
├── coder.py                # Main test runner
├── requirements.txt        # Python dependencies
├── Methods/                # Steganography method implementations
├── lib/                    # Utilities (analysis, attacks, logger, etc.)
├── Carriers/               # Audio files used as carriers (tracked by Git)
└── .gitignore              # Git ignore rules
```

## Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests**
   ```bash
   python coder.py
   ```
   You can provide a config file:
   ```bash
   python coder.py --config test_config.json
   ```
3. **Parameter optimization**
   Example usage:
   ```bash
   python rosenbrock_optymization.py
   ```

## Key Files
- `coder.py` – main test runner and analysis script
- `requirements.txt` – required Python libraries
- `Methods/` – steganography algorithm implementations
- `lib/` – utilities for analysis, attacks, logging, secrets
- `Carriers/` – audio carriers for experiments

## Adding a New Method
1. Add your implementation file to the `Methods/` directory.
2. Import the class in `coder.py`.
3. Add the method to the `methods` list with appropriate parameters.

## Results & Analysis
- Test results are saved in the `Results/` directory (not present on the server).
- Analyses and reports are generated automatically after tests.
- Results can be grouped by various criteria (e.g., method, attack type, parameters).

## Supported Attacks
- Noise
- Filtering (highpass, lowpass)
- MP3 compression
- Resampling
- Clipping
- Jitter
- Requantize

## Requirements
- Python 3.8+
- Libraries from `requirements.txt` (e.g., numpy, scipy, pandas, matplotlib, alive-progress)

## Tips
- Audio files in `Carriers/` are tracked by Git and available on the server.
- You can easily extend the project with new methods, attacks, and analyses.

## Contact
This project was developed as part of a PhD thesis. For questions or collaboration, please contact the author.
