"""Prebuilt domain packs for the lakehouse knowledge graph pipeline."""

from lakehouse_kg.domain import DomainPack
from lakehouse_kg.domains.generic import GENERIC_PACK, generic_pack
from lakehouse_kg.domains.fraud import FRAUD_PACK, fraud_pack

_REGISTRY = {
    "generic": generic_pack,
    "fraud": fraud_pack,
}


def get_domain_pack(name: str) -> DomainPack:
    """Return a fresh DomainPack instance by name ('generic' or 'fraud')."""
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise ValueError(
            f"Unknown domain pack {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


__all__ = [
    "DomainPack",
    "GENERIC_PACK",
    "FRAUD_PACK",
    "generic_pack",
    "fraud_pack",
    "get_domain_pack",
]
