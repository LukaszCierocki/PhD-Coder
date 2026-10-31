"""
PhD Coder - steganography test runner.

    python coder.py                                   # preset from RUN_SETTINGS['TEST']
    python coder.py --test CAPACITY_SWEEP             # data for the "<metric> vs Capacity" charts
    python coder.py --test FOR_CLASSIFIER --steganalysis
    python coder.py --config ~/Results/<run>/test_config_<time>.json   # replay a saved run
    python coder.py --plots-from ~/Results/<run1> ~/Results/<run2> --out ~/Results/Plots
    python coder.py --steganalysis-from ~/Results/<run>
    python coder.py --predict suspicious.wav --models ~/Results/<run>/Steganalysis/ML/models
    python coder.py --list-methods

See README.md and docs/ for details.
"""

import argparse
import json
import logging
import os
import sys
import traceback
import zlib
from contextlib import contextmanager
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # batch runner - never open windows

import pandas as pd

from lib.Analysis import Analysis, test_label_from_filename
from lib.Attacks import Attacks, no_attacks
from lib.Registry import available_methods, canonical_name, create_method
from lib.Secret import Secret
from lib.Tools import Tools
from lib import Plots
from lib import Steganalysis

try:
    from alive_progress import alive_bar
except ImportError:  # pragma: no cover
    alive_bar = None


# --------------------------------------------------------------------------- #
# Settings (defaults - every value can be overridden from the command line)
# --------------------------------------------------------------------------- #

RUN_SETTINGS = {
    'TEST': 'DEBUG',             # preset name, see PRESETS below
    'LOG_LEVEL': 'INFO',         # INFO / DEBUG (file log)
    'ONLY_EMBED': False,         # only produce stegobjects (e.g. for steganalysis)
    'PER_FILE_PLOTS': True,      # waveform / spectrogram / singular value charts per stegobject
    'SUMMARY_PLOTS': True,       # "<metric> vs Capacity" charts after the run
    'PALETTE': 'husl',           # 'husl' (thesis figures) or 'accessible' (colour-blind safe)
    'CI': False,                 # 95% confidence bands in the summary charts
    'STEGANALYSIS': False,       # statistical + ML steganalysis after the run
    'STEGANALYSIS_CONFIG': {},   # overrides of lib.Steganalysis.DEFAULT_CONFIG (e.g. {'PERM_B': 500})
    'SEED': None,                # int -> reproducible secrets and attacks
    'MAX_CARRIERS': None,        # use only the first N carriers (sorted by name)
}

# Common parameters of the frame-based methods.
BASE_PARAMS = {
    'REDUCED': False, 'ADAPTIVE': False, 'ALPHA': 0.7, 'BETA': 0.7, 'TYPE_OF_VAD': 'rVAD',
    'WHERE_TO_HIDE': 'VOICED', 'FREQ_L': 1500, 'FREQ_H': 2500, 'FRAME_LENGTH': 20,
    'HOP_DURATION_MS': 20, 'LEVELS': 1, 'HANKEL_ROWS': 2,
}

# Method set of the thesis comparison: (registry name, parameter overrides).
# 'MDCT': True -> Wen, MDCT frames twice as long as the hop (50% overlap).
THESIS_METHODS = [
    ('MDCT_SVD_WEN', {'ADAPTIVE': False, 'MDCT': True}),
    ('DIFFERENTIAL_SVD_XUE', {'ADAPTIVE': False}),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V1', {'ADAPTIVE': False}),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_V2', {'ADAPTIVE': True}),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V1', {'ADAPTIVE': False}),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V2', {'ADAPTIVE': True}),
    ('DIFFERENTIAL_SVD_PEJAS_CIEROCKI_MULTI_PC_V3', {'ADAPTIVE': True, 'LEVELS': 8}),
]

# Nominal capacities of the thesis charts: 1 bit per hop -> hop = 1000 / bps [ms]
CAPACITY_SWEEP_BPS = [25, 31.25, 50, 62.5, 80, 100]


def params(hop_ms=20, **overrides):
    """
    Parameters for one method. ``MDCT=True`` -> FRAME_LENGTH = 2 x hop
    (Wen), otherwise FRAME_LENGTH = hop.
    """
    mdct = overrides.pop('MDCT', False)
    if float(hop_ms).is_integer():
        hop_ms = int(hop_ms)  # keeps method names as before (FL_20, not FL_20.0)
    p = dict(BASE_PARAMS)
    p['HOP_DURATION_MS'] = hop_ms
    p['FRAME_LENGTH'] = 2 * hop_ms if mdct else hop_ms
    p.update(overrides)
    return p


def thesis_methods(logger, hop_ms=20):
    return [create_method(name, params(hop_ms, **dict(overrides)), logger) for name, overrides in THESIS_METHODS]


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

def build_for_classifier(logger):
    """Stegobjects of all thesis methods (50 bps) + steganalysis."""
    return {
        'methods': thesis_methods(logger, hop_ms=20),
        'attacks': no_attacks(),
        'settings': {'STEGANALYSIS': True, 'PER_FILE_PLOTS': False},
    }


def build_debug(logger):
    """Quick check of one method."""
    return {
        'methods': [create_method('DIFFERENTIAL_SVD_XUE', params(20), logger)],
        'attacks': no_attacks(),
        'settings': {},
    }


def build_capacity_sweep(logger):
    """All thesis methods at 25 ... 100 bps (data for the capacity charts)."""
    methods = []
    for bps in CAPACITY_SWEEP_BPS:
        methods.extend(thesis_methods(logger, hop_ms=1000.0 / bps))
    return {
        'methods': methods,
        'attacks': no_attacks(),
        'settings': {'PER_FILE_PLOTS': False},
    }


def build_robustness(logger):
    """All thesis methods at 50 bps with all attacks (BER per attack charts)."""
    return {
        'methods': thesis_methods(logger, hop_ms=20),
        'attacks': {name: True for name in no_attacks()},
        'settings': {'PER_FILE_PLOTS': False},
    }


PRESETS = {
    'DEBUG': build_debug,
    'FOR_CLASSIFIER': build_for_classifier,
    'CAPACITY_SWEEP': build_capacity_sweep,
    'ROBUSTNESS': build_robustness,
}


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

@contextmanager
def progress(total):
    if alive_bar is None:
        yield lambda: None
    else:
        with alive_bar(total) as bar:
            yield bar


def derive_seed(base_seed, method_name, carrier_name):
    """Stable per (method, carrier) seed derived from the run seed."""
    if base_seed is None:
        return None
    return zlib.crc32(f"{base_seed}|{method_name}|{carrier_name}".encode()) & 0xFFFFFFFF


def _is_copy_of(filename, test_name):
    """True for the clean stegobject and for attacked copies '<prefix>_<test_name>'."""
    return filename == test_name or (filename.endswith(test_name) and filename[-len(test_name) - 1] == '_')


def run_experiment(tools, jobs, attacks_config, settings):
    """
    Runs embedding -> attacks -> extraction -> analysis for every job.

    jobs: list of {'method': Method, 'carrier': Carrier, 'seed': int | None}
    Returns a dict with the analysis object, performed tests, failures,
    steganalysis pairs and secrets.
    """
    logger = tools.get_logger()
    test_dir = tools.get_current_test_directory()
    only_embed = settings['ONLY_EMBED']

    attacks = Attacks(logger=logger, seed=settings['SEED'])
    analysis = Analysis(logger=logger, plots=settings['PER_FILE_PLOTS'])

    performed_tests, failures, pairs, secrets_config = [], [], [], []

    with progress(len(jobs)) as bar:
        for job in jobs:
            method, carrier = job['method'], job['carrier']
            label = f"{method.name} / {carrier.name}"
            try:
                logger.info(f"Running test for method: {method.name}, carrier: {carrier.name}")

                method_directories = tools.prepare_method_directory(test_dir, method.name, only_embed)
                method.set_method_directories(method_directories)
                method.set_carrier(carrier)

                secret = Secret({'type': 'MAX_CAP', 'size': method.get_capacity(),
                                 'save_directory': method_directories['Secrets'], 'only_embed': only_embed,
                                 'seed': job.get('seed')}, logger=logger)
                method.set_secret(secret)

                stegobject = method.embedding()
                key = method.get_extraction_key()

                stego_dir = method_directories['Stegobjects']

                if only_embed:
                    stego_name = carrier.name
                    tools.save_audio_file(stego_dir, {'name': stego_name + '.wav', 'content': stegobject,
                                                      'fs': carrier.fs})
                    tools.save_extraction_key(key, method_directories, stego_name)
                    if carrier.path:
                        pairs.append({'method': method.name, 'label': method.normal_name, 'carrier': carrier.name,
                                      'carrier_path': carrier.path,
                                      'stego_path': os.path.join(stego_dir, stego_name + '.wav')})
                    bar()
                    continue

                secrets_config.append(secret.to_metadata())

                test_name = method.name + '_' + secret.name + '_' + carrier.name
                tools.save_audio_file(stego_dir, {'name': test_name + '.wav', 'content': stegobject, 'fs': carrier.fs})
                key_path = tools.save_extraction_key(key, method_directories, test_name)

                if carrier.path:
                    pairs.append({'method': method.name, 'label': method.normal_name, 'carrier': carrier.name,
                                  'carrier_path': carrier.path,
                                  'stego_path': os.path.join(stego_dir, test_name + '.wav')})

                attacks.perform_attacks(stego_dir, test_name, attacks_config)

                paths, filenames = tools.find_files_with_string_in_name(stego_dir, test_name, recursive=False)

                for stegobject_path, filename in zip(paths, filenames):
                    if not _is_copy_of(filename, test_name):
                        continue
                    try:
                        loaded_stegobject = tools.load_audio_file(stegobject_path)
                        extracted_data = method.extraction(loaded_stegobject, key=key)
                        tools.save_extracted_secret(extracted_data, method_directories, filename)
                        analysis.analyze(method, carrier, secret, loaded_stegobject, extracted_data,
                                         test_label=test_label_from_filename(filename, test_name))
                    except Exception as e:
                        logger.error(f"Error during extraction/analysis for {test_name}, file: {filename}: {e}")
                        logger.error(traceback.format_exc())
                        failures.append({'test': label, 'file': filename, 'error': str(e)})

                performed_tests.append({
                    "method": {"name": method.__class__.__name__, "params": method.method_arguments},
                    "carrier": carrier.name,
                    "secret": {"seed": secret.seed, "type": secret.type, "size": secret.size},
                    "test_name": test_name,
                    "key": os.path.relpath(key_path, test_dir),
                })

            except Exception as e:
                logger.error(f"Test failed for method: {method.name}, carrier: {carrier.name}: {e}")
                logger.error(traceback.format_exc())
                failures.append({'test': label, 'file': None, 'error': str(e)})

            bar()

    return {'analysis': analysis, 'performed_tests': performed_tests, 'failures': failures,
            'pairs': pairs, 'secrets': secrets_config}


def finish_run(tools, result, settings, methods_config, carriers_config, attacks_config):
    """CSV files, console tables, config JSON, summary charts, steganalysis."""
    test_dir = tools.get_current_test_directory()
    logger = tools.get_logger()
    analysis = result['analysis']
    written = {'test_dir': test_dir}

    if result['pairs']:
        written['pairs'] = Steganalysis.save_pairs(result['pairs'], os.path.join(test_dir, 'steganalysis_pairs.json'))

    if not settings['ONLY_EMBED'] and analysis.methods_analysis:
        written['csv'] = analysis.save_methods_analysis_to_csv(test_dir)
        analysis.print_results()
        group_by = ["NAME", "TEST", "VAD", 'LEVELS', 'ALPHA', 'BETA', 'CAP. [bps]', 'ADAPTIVE', 'REDUCED']
        analysis.analyze_by_multiple_groups(group_by, filters=None)
        written['grouped_csv'] = analysis.save_grouped_to_csv(group_by, test_dir)

        if settings['SUMMARY_PLOTS']:
            try:
                df = Plots.load_results(written['csv'])
                written['charts'] = Plots.plot_capacity_charts(df, os.path.join(test_dir, 'Plots'),
                                                               metrics=list(Plots.CHARTS),
                                                               palette=settings['PALETTE'], ci=settings['CI'])
            except Exception as e:
                logger.error(f"Summary charts failed: {e}")
                logger.error(traceback.format_exc())

    config = {
        "created": datetime.now().isoformat(timespec='seconds'),
        "settings": settings,
        "methods": methods_config,
        "carriers": carriers_config,
        "secrets": result['secrets'],
        "attacks": attacks_config,
        "performed_tests": result['performed_tests'],
        "failures": result['failures'],
    }
    config_path = os.path.join(test_dir, f'test_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(config, f, indent=4, default=str)
    written['config'] = config_path

    if settings['STEGANALYSIS'] and result['pairs']:
        try:
            Steganalysis.run_steganalysis(result['pairs'], os.path.join(test_dir, 'Steganalysis'),
                                          settings['STEGANALYSIS_CONFIG'], logger)
            written['steganalysis'] = os.path.join(test_dir, 'Steganalysis')
        except Exception as e:
            logger.error(f"Steganalysis failed: {e}")
            logger.error(traceback.format_exc())

    print(f"\nResults: {test_dir}")
    if result['failures']:
        print(f"WARNING: {len(result['failures'])} test(s) failed - see the log and 'failures' in {config_path}")
    written['failures'] = result['failures']
    return written


def make_tools(settings, results_root=None, carriers_dir=None):
    log_level = logging.DEBUG if str(settings['LOG_LEVEL']).upper() == 'DEBUG' else logging.INFO
    return Tools(log_level, results_root=results_root, carriers_dir=carriers_dir)


def run_and_save_test(settings=None, results_root=None, carriers_dir=None):
    """
    Runs a preset. ``settings`` overrides RUN_SETTINGS; values listed in
    ``settings['_explicit']`` (command line) also override the preset's own
    defaults.
    """
    settings = dict(RUN_SETTINGS, **(settings or {}))
    explicit = settings.pop('_explicit', None)
    if explicit is None:
        explicit = {k: v for k, v in settings.items() if k in RUN_SETTINGS and v != RUN_SETTINGS[k]}

    preset_name = settings['TEST']
    if preset_name not in PRESETS:
        raise SystemExit(f"Unknown test preset {preset_name!r}. Available: {sorted(PRESETS)}")

    tools = make_tools(settings, results_root, carriers_dir)
    logger = tools.get_logger()
    test_id = tools.get_test_id()
    logger.info(f"Starting test with ID: {test_id}")

    preset = PRESETS[preset_name](logger)
    for key, value in preset.get('settings', {}).items():
        if key not in explicit:
            settings[key] = value

    carriers = tools.load_carriers(limit=settings['MAX_CARRIERS'])
    if not carriers:
        raise SystemExit(f"No carriers found in {tools.carriers_dir}.")

    methods = preset['methods']
    jobs = [{'method': method, 'carrier': carrier, 'seed': derive_seed(settings['SEED'], method.name, carrier.name)}
            for method in methods for carrier in carriers]

    result = run_experiment(tools, jobs, preset['attacks'], settings)

    print(f"TEST ID: {test_id}")
    return finish_run(tools, result, settings,
                      [{"name": m.__class__.__name__, "params": m.method_arguments} for m in methods],
                      [c.to_metadata() for c in carriers], preset['attacks'])


def run_test_from_config(config_path, settings=None, results_root=None, carriers_dir=None):
    """
    Replays a saved run: same methods, parameters, carriers, secret seeds and
    attacks. (Random attacks are reproducible only if the original run had a
    SEED.) Works with configuration files of older versions as well.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    settings = dict(RUN_SETTINGS, **{k: v for k, v in (config.get('settings') or {}).items() if k in RUN_SETTINGS},
                    **(settings or {}))
    settings.pop('_explicit', None)
    settings['ONLY_EMBED'] = False

    tools = make_tools(settings, results_root, carriers_dir)
    logger = tools.get_logger()
    logger.info(f"Replaying {config_path}")

    jobs, carriers_cache = [], {}
    for test in config["performed_tests"]:
        method = create_method(canonical_name(test["method"]["name"]), test["method"]["params"], logger)
        carrier_name = test["carrier"]
        if carrier_name not in carriers_cache:
            carriers_cache[carrier_name] = tools.load_carrier(carrier_name)
        jobs.append({'method': method, 'carrier': carriers_cache[carrier_name], 'seed': test["secret"]["seed"]})

    attacks_config = dict(no_attacks(), **config.get("attacks", {}))
    result = run_experiment(tools, jobs, attacks_config, settings)

    return finish_run(tools, result, settings, config.get("methods", []),
                      [c.to_metadata() for c in carriers_cache.values()], attacks_config)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Steganography test runner",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--config", type=str, help="replay a saved test_config_*.json")
    parser.add_argument("--test", type=str, help=f"preset: {', '.join(PRESETS)}")
    parser.add_argument("--log-level", choices=['INFO', 'DEBUG'])
    parser.add_argument("--only-embed", action='store_true', default=None, help="only create stegobjects")
    parser.add_argument("--no-file-plots", action='store_true', help="skip per-stegobject charts")
    parser.add_argument("--no-summary-plots", action='store_true', help="skip '<metric> vs Capacity' charts")
    parser.add_argument("--steganalysis", action='store_true', default=None, help="run steganalysis after the test")
    parser.add_argument("--perm-b", type=int, help="permutations of the statistical tests (default 200)")
    parser.add_argument("--carriers", type=int, help="use only the first N carriers")
    parser.add_argument("--carriers-dir", type=str, help="carrier directory (default ./Carriers)")
    parser.add_argument("--results-root", type=str, help="results directory (default $PHD_CODER_RESULTS or ~/Results)")
    parser.add_argument("--seed", type=int, help="seed for secrets and attacks")
    parser.add_argument("--palette", choices=['husl', 'accessible'], help="chart palette")
    parser.add_argument("--ci", action='store_true', default=None, help="95%% confidence bands in charts")

    parser.add_argument("--plots-from", nargs='+', metavar='PATH',
                        help="draw capacity charts from analysis CSV files / result directories")
    parser.add_argument("--metrics", nargs='+', help="metrics for --plots-from (default: all)")
    parser.add_argument("--out", type=str, help="output directory for --plots-from / --steganalysis-from")
    parser.add_argument("--steganalysis-from", type=str, metavar='RESULTS_DIR',
                        help="run steganalysis on an existing results directory")
    parser.add_argument("--predict", type=str, metavar='WAV', help="classify one file with saved models")
    parser.add_argument("--models", type=str, help="models directory for --predict")
    parser.add_argument("--classifier", default='SVM', help="classifier for --predict (SVM / FRBB)")
    parser.add_argument("--list-methods", action='store_true', help="list importable methods")
    return parser.parse_args(argv)


def settings_from_args(args):
    explicit = {
        'TEST': args.test,
        'LOG_LEVEL': args.log_level,
        'ONLY_EMBED': args.only_embed,
        'PER_FILE_PLOTS': False if args.no_file_plots else None,
        'SUMMARY_PLOTS': False if args.no_summary_plots else None,
        'STEGANALYSIS': args.steganalysis,
        'SEED': args.seed,
        'MAX_CARRIERS': args.carriers,
        'PALETTE': args.palette,
        'CI': args.ci,
    }
    settings = {k: v for k, v in explicit.items() if v is not None}
    if args.perm_b is not None:
        settings['STEGANALYSIS_CONFIG'] = dict(RUN_SETTINGS['STEGANALYSIS_CONFIG'], PERM_B=args.perm_b)
    settings['_explicit'] = {k for k, v in explicit.items() if v is not None}
    return settings


def main(argv=None):
    args = parse_args(argv)

    if args.list_methods:
        for name in available_methods():
            print(name)
        return 0

    if args.plots_from:
        df = Plots.load_results(args.plots_from)
        out = args.out or os.path.join(os.getcwd(), 'Plots')
        charts = Plots.plot_capacity_charts(df, out, metrics=args.metrics or list(Plots.CHARTS),
                                            palette=args.palette or RUN_SETTINGS['PALETTE'], ci=bool(args.ci))
        print(f"{len(charts)} chart(s) written to {out}")
        return 0

    if args.steganalysis_from:
        carriers_dir = args.carriers_dir or os.path.join(os.getcwd(), 'Carriers')
        pairs = Steganalysis.pairs_from_results_dir(args.steganalysis_from, carriers_dir)
        if not pairs:
            print(f"No carrier/stegobject pairs found in {args.steganalysis_from}")
            return 1
        out = args.out or os.path.join(args.steganalysis_from, 'Steganalysis')
        config = dict(RUN_SETTINGS['STEGANALYSIS_CONFIG'])
        if args.perm_b is not None:
            config['PERM_B'] = args.perm_b
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        Steganalysis.run_steganalysis(pairs, out, config, logging.getLogger('steganalysis'))
        print(f"Steganalysis of {len(pairs)} pairs written to {out}")
        return 0

    if args.predict:
        if not args.models:
            print("--predict needs --models <dir> (e.g. <run>/Steganalysis/ML/models)")
            return 1
        report = Steganalysis.predict_file(args.predict, args.models, classifier=args.classifier)
        print(f"\nFORENSIC ANALYSIS REPORT - {os.path.basename(args.predict)}\n")
        for entry in report.values():
            details = ', '.join(f"{k}: {v}" for k, v in entry['details'].items())
            print(f"{entry['label']:22s} {entry['verdict']:10s} ({entry['votes']}/{entry['models']})  {details}")
        return 0

    settings = settings_from_args(args)
    if args.config:
        settings.pop('_explicit', None)
        run_test_from_config(args.config, settings, args.results_root, args.carriers_dir)
    else:
        run_and_save_test(settings, args.results_root, args.carriers_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
