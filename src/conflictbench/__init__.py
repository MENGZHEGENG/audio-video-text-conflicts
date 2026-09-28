"""Controlled active-diagnosis benchmark for three-channel evidence."""

from .core import Dataset, generate_dataset, evaluate_actions, policy_actions
from .runner import run_from_config
from .temporal import (
    TEMPORAL_LEARNED_METHODS,
    TEMPORAL_MECHANISMS,
    TEMPORAL_PROTOCOL,
    TemporalDataset,
    generate_temporal_dataset,
    temporal_summary,
)
from .temporal_campaign import (
    TEMPORAL_CAMPAIGN_SCHEMA,
    TemporalCampaignJob,
    expand_campaign_spec,
    load_campaign_spec,
    resolve_campaign_job,
    validate_campaign_spec,
)

__all__ = [
    "Dataset",
    "evaluate_actions",
    "generate_dataset",
    "policy_actions",
    "run_from_config",
    "TemporalDataset",
    "generate_temporal_dataset",
    "temporal_summary",
    "TEMPORAL_LEARNED_METHODS",
    "TEMPORAL_MECHANISMS",
    "TEMPORAL_PROTOCOL",
    "TEMPORAL_CAMPAIGN_SCHEMA",
    "TemporalCampaignJob",
    "expand_campaign_spec",
    "load_campaign_spec",
    "resolve_campaign_job",
    "validate_campaign_spec",
]
