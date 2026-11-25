"""Training recipes, one per model. Add one by subclassing training.base.Recipe and listing it here."""
from .orpheus import OrpheusRecipe
from .qwen import QwenRecipe

RECIPES = {r.id: r for r in (OrpheusRecipe(), QwenRecipe())}
