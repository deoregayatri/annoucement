from typing import Literal, Optional

from pydantic import BaseModel, Field


class SatOrdersDateWiseRequest(BaseModel):
    apl_type: Optional[Literal["sebi", "irda", "pfrda"]] = Field(
        default="sebi",
        description="Appeal type / bench to search (optional).",
        examples=["sebi", "irda", "pfrda"],
    )
    from_date: str = Field(
        description="Start date (inclusive). Format: DD-MM-YYYY or YYYY-MM-DD",
    )
    to_date: str = Field(
        description="End date (inclusive). Format: DD-MM-YYYY or YYYY-MM-DD",
    )
    via_proxy: bool = Field(
        default=False,
        description=(
            "Route through the ScrapeOps proxy. Default false — SAT requires a "
            "stateful cookie session (session cookie + rotating security_token), "
            "which is not preserved when requests are routed through the proxy."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "apl_type": "sebi",
                    "from_date": "01-01-2026",
                    "to_date": "30-06-2026",
                },
                {
                    "apl_type": "irda",
                    "from_date": "2026-01-01",
                    "to_date": "2026-03-31",
                },
            ]
        }
    }
