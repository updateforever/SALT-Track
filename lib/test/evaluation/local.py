from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def _project_root() -> Path:
    # local.py -> evaluation -> test -> lib -> repo root
    return Path(__file__).resolve().parents[3]


def _dataset_root() -> Path:
    return Path("/root/user-data/PUBLIC_DATASETS")


def local_env_settings():
    settings = EnvSettings()

    prj_dir = _project_root()
    data_root = _dataset_root()
    output_root = prj_dir / "output"

    settings.prj_dir = str(prj_dir)

    # Model / checkpoint paths
    settings.checkpoints_path = str(output_root / "checkpoints")
    settings.network_path = settings.checkpoints_path

    # Evaluation output paths
    settings.save_dir = str(output_root)
    settings.results_path = str(output_root / "test" / "tracking_results")
    settings.result_plot_path = str(output_root / "test" / "result_plots")
    settings.segmentation_path = str(output_root / "test" / "segmentation_results")

    # Dataset roots
    settings.davis_dir = str(data_root / "davis")
    settings.got10k_lmdb_path = str(data_root / "got10k_lmdb")
    settings.got10k_path = str(data_root / "got10k")
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.itb_path = str(data_root / "itb")
    settings.lasot_extension_subset_path = str(data_root / "lasot_extension_subset")
    settings.lasot_lmdb_path = str(data_root / "lasot_lmdb")
    settings.lasotlang_path = str(data_root / "lasot")
    settings.nfs_path = str(data_root / "nfs")
    settings.otb_lang_path = str(data_root / "OTB_sentences")
    settings.otb_path = str(data_root / "otb")
    settings.tc128_path = str(data_root / "TC128")
    settings.tn_packed_results_path = ""
    settings.tnl2k_path = str(data_root / "TNL2K" / "TNL2K_test_subset")
    settings.tpl_path = ""
    settings.trackingnet_path = str(data_root / "trackingnet")

    return settings  
