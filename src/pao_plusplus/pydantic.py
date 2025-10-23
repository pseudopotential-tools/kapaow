"""Default pydantic base models to use elsewhere."""

from pydantic import BaseModel as PydanticBaseModel

class BaseModel(PydanticBaseModel):
    """Base pydantic model with common config."""

    class Config:
        """Common config for all pydantic models in `pao_plusplus`."""
        validate_assignment = True
        extra = "forbid"
        arbitrary_types_allowed = True