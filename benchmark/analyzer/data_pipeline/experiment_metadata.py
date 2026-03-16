from __future__ import annotations

import logging
import os
from typing import Any, Dict


logger = logging.getLogger(__name__)


def _flatten_dict(d: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dict into dot-path keys."""
    result = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten_dict(v, new_key, sep))
            else:
                result[new_key] = v
    return result


class ExperimentMetadata:
    """Extracts a flat, dot-addressable metadata dict from an experiment."""

    @staticmethod
    def extract(exp: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"filename": getattr(exp, "_source_filename", "") or "",
                                "exp_name": getattr(exp, "name", "") or "", "exp_id": getattr(exp, "id", "") or "",
                                "num_measured": len(getattr(exp, "measured_configurations", []))}

        desc = getattr(exp, "description", None)
        if isinstance(desc, dict):
            meta.update(_flatten_dict(desc, prefix="description"))
        elif desc is not None:
            logger.warning(
                "Experiment '%s' has non-dict description of type '%s'",
                meta.get("exp_name") or meta.get("exp_id") or "unknown",
                type(desc).__name__,
            )

        ExperimentMetadata._add_aliases(meta, exp)

        return meta

    @staticmethod
    def _add_aliases(meta: Dict[str, Any], exp: Any) -> None:
        desc = getattr(exp, "description", None)
        desc_dict = desc if isinstance(desc, dict) else {}

        task_name = meta.get("description.TaskConfiguration.TaskName", "")
        meta["task_name"] = str(task_name) if task_name else ""

        instance_path = (
            meta.get("description.TaskConfiguration.Scenario.problem_initialization_parameters.instance")
            or meta.get("description.DomainDescription.instance")
            or ""
        )
        meta["problem_instance_path"] = str(instance_path)
        meta["problem_instance"] = os.path.basename(str(instance_path)) if instance_path else ""

        hp = meta.get("description.TaskConfiguration.Scenario.Hyperparameters", "")
        meta["hyperparams_mode"] = str(hp) if hp else ""

        predictor = desc_dict.get("Predictor", {}) if isinstance(desc_dict.get("Predictor", {}), dict) else {}
        models = predictor.get("models", []) if isinstance(predictor.get("models", []), list) else []
        meta["model_types"] = [m.get("Type", "") for m in models if isinstance(m, dict)]

        cfgs = getattr(exp, 'measured_configurations', [])
        meta["first_param_values"] = {}
        if cfgs:
            hyperparams = getattr(cfgs[0], "hyperparameters", None)
            if isinstance(hyperparams, dict):
                meta["first_param_values"] = dict(hyperparams)
            elif hyperparams is not None:
                try:
                    meta["first_param_values"] = dict(hyperparams)
                except (TypeError, ValueError):
                    logger.warning(
                        "Experiment '%s' has non-mapping first hyperparameters of type '%s'",
                        meta.get("exp_name") or meta.get("exp_id") or "unknown",
                        type(hyperparams).__name__,
                    )

        dd = desc_dict.get("DomainDescription", {}) if isinstance(desc_dict.get("DomainDescription", {}), dict) else {}
        datafile = dd.get("DataFile", "") if isinstance(dd, dict) else ""
        meta["domain_datafile"] = str(datafile)
        meta["mh_type"] = ExperimentMetadata._parse_mh_type(datafile)

        predictor_models = predictor.get("models", []) if isinstance(predictor, dict) else []
        model_type_names = [m.get("Type", "").split(".")[-1] for m in predictor_models if isinstance(m, dict)]

        model_cfg = desc_dict.get("ModelConfiguration", {}) if isinstance(desc_dict.get("ModelConfiguration", {}), dict) else {}
        legacy_model_type = model_cfg.get("ModelType", "") if isinstance(model_cfg, dict) else ""

        meta["model_types_list"] = model_type_names
        meta["tuning_variant"] = ExperimentMetadata._parse_tuning_variant(
            model_type_names,
            meta.get("hyperparams_mode", ""),
            legacy_model_type=legacy_model_type,
            mh_type=meta.get("mh_type", ""),
        )

    @staticmethod
    def _parse_mh_type(datafile: str) -> str:
        """Derive a human-readable MH label from the DomainDescription DataFile path.

        Handles both old naming (``MHjMetalPyESData.json``) and new naming
        with ConfigSpace suffix (``pyESDataConfigSpace.json``).

        Examples::
            MHjMetalPyESData.json / pyESDataConfigSpace.json  → py.ES
            MHjMetalPySAData.json / pySADataConfigSpace.json  → py.SA
            MHjMetalESData.json   / jESDataConfigSpace.json   → j.ES
            HHData.json           / HHDataConfigSpace.json    → HH-PC
        """
        import os as _os
        b = _os.path.basename(datafile).lower()
        # Simulated-Annealing check must come before plain ES check
        if "pysa" in b or ("py" in b and "sa" in b):
            return "py.SA"
        if "pyes" in b or ("jmetalpy" in b and "es" in b):
            return "py.ES"
        if "jes" in b or ("jmetal" in b and "es" in b):
            return "j.ES"
        if b.startswith("hh") or "hhdata" in b:
            return "HH-PC"
        return _os.path.splitext(_os.path.basename(datafile))[0] if datafile else ""

    @staticmethod
    def _parse_tuning_variant(model_type_names: list, hyperparams_mode: str,
                               legacy_model_type: str = "", mh_type: str = "") -> str:
        """Encode the BRISE surrogate / control variant label.

        Supports two experiment formats:

        **New format** – ``Predictor.models`` list (full_benchmark):
            ModelMock + TreeParzenEstimator  →  S-TPE / H-TPE (if MAB present)
            ModelMock + BayesianRidge        →  S-BRR / H-BRR (if MAB present)
            ModelMock only + default         →  Default
            ModelMock only + tuned           →  Tuned
            ModelMock only + random          →  Random

        **Old format** – ``ModelConfiguration.ModelType`` (sparse_pc_and_hh_pc):
            BO  (= Bayesian Optimisation / TPE)  →  S-TPE or H-TPE (if HH-PC)
            brr (= Bayesian Ridge Regression)    →  S-BRR or H-BRR (if HH-PC)
        """
        hp = (hyperparams_mode or "").lower()

        if legacy_model_type and not model_type_names:
            lmt = legacy_model_type.lower()
            is_hh = "hh" in mh_type.lower()
            if "bo" in lmt:
                return "H-TPE" if is_hh else "S-TPE"
            if "brr" in lmt:
                return "H-BRR" if is_hh else "S-BRR"
            return legacy_model_type

        if hp == "default":
            return "Default"
        if hp == "tuned":
            return "Tuned"
        if hp == "random":
            return "Random"

        names = [n.lower() for n in model_type_names]
        has_mab = any("multiarmedbandit" in n for n in names)
        has_tpe = any("treeparzen" in n for n in names)
        has_brr = any("bayesianridge" in n for n in names)

        if has_mab and has_tpe:
            return "H-TPE"
        if has_mab and has_brr:
            return "H-BRR"
        if has_mab:
            return "H-TPE"
        if has_tpe:
            return "S-TPE"
        if has_brr:
            return "S-BRR"
        return "S-TPE"  # ModelMock-only with provided → control baseline

    @staticmethod
    def get(meta: Dict[str, Any], path: str) -> Any:
        """
        Retrieve a value from the metadata dict by dot-path.

        Supports both:
          - Top-level aliases:  ``"problem_instance"``, ``"model_types"`` …
          - Full dot-paths:     ``"description.TaskConfiguration.TaskName"``
          - Sub-path shortcut:  ``"TaskConfiguration.TaskName"``  (tries with
            ``"description."`` prefix automatically)
          - Hyperparameter leaf: ``"first_param_values.low level heuristic"``
        """
        if path in meta:
            return meta[path]

        # try with "description." prefix
        desc_path = "description." + path
        if desc_path in meta:
            return meta[desc_path]

        # walk first_param_values
        fpv_prefix = "first_param_values."
        if path.startswith(fpv_prefix):
            key = path[len(fpv_prefix):]
            return meta.get("first_param_values", {}).get(key)

        return None

