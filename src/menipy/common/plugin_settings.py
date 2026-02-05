"""
Infrastructure for plugin-specific configuration settings.

This module provides the registry and base classes for plugins to define their own
settings models, decoupling them from the core configuration.
"""
from __future__ import annotations

from typing import Dict, Any, Type, Optional
from pydantic import BaseModel

from menipy.common.registry import Registry

# Registry for detector settings models
# Maps method name (e.g., "log") to a Pydantic model class (e.g., LoGSettings)
DETECTOR_SETTINGS = Registry("detector_settings")


def register_detector_settings(name: str, settings_model: Type[BaseModel]) -> None:
    """Register a settings model for a specific detector method."""
    DETECTOR_SETTINGS.register(name, settings_model)


def get_detector_settings_model(name: str) -> Optional[Type[BaseModel]]:
    """Get the settings model class for a detector."""
    return DETECTOR_SETTINGS.get(name)


def resolve_plugin_settings(
    method: str,
    generic_settings: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Resolve settings for a specific plugin method.
    
    Tries to find a registered settings model for the method and validates
    the provided generic settings against it.
    
    Args:
        method: The plugin method name (e.g. "log")
        generic_settings: Dictionary of generic/plugin-specific settings (from config)
        **kwargs: Additional overrides
        
    Returns:
        Dictionary of validated settings
    """
    merged = generic_settings.copy()
    merged.update(kwargs)
    
    model_cls = get_detector_settings_model(method)
    if model_cls:
        try:
            # Validate settings using the registered model.
            # We assume the model is configured to ignore extra fields (ConfigDict(extra='ignore'))
            instance = model_cls(**merged)
            return instance.model_dump()
        except Exception:
            # If validation fails, fallback to returning the unvalidated dict
            # or we could log a warning here.
            pass
            
    return merged
