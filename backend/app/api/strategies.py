from fastapi import APIRouter, HTTPException
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.database.models import Strategy
from app.data.database.session import get_db
from app.strategies.registry import STRATEGY_CLASSES

router = APIRouter(prefix="/strategies", tags=["Strategies"])


class ParameterSchema(BaseModel):
    key: str
    label: str
    type: str
    default: int | float | str | bool | None = None
    min: float | None = None
    max: float | None = None


class StrategyResponse(BaseModel):
    id: int
    name: str
    description: str | None


@router.get("", response_model=list[StrategyResponse])
async def list_strategies(db: Session = Depends(get_db)) -> list[StrategyResponse]:
    rows = db.execute(select(Strategy).order_by(Strategy.name.asc())).scalars().all()
    return [
        StrategyResponse(id=row.id, name=row.name, description=row.description)
        for row in rows
    ]


@router.get("/names")
async def list_strategy_names(db: Session = Depends(get_db)):
    rows = db.execute(select(Strategy.name).order_by(Strategy.name.asc())).all()
    return {"available_strategies": [r[0] for r in rows]}


@router.get("/{strategy_id}/parameters", response_model=list[ParameterSchema])
async def get_strategy_parameters(
    strategy_id: int,
    db: Session = Depends(get_db),
) -> list[ParameterSchema]:
    row = db.get(Strategy, strategy_id)
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy_class = STRATEGY_CLASSES.get(row.name)
    if not strategy_class:
        raise HTTPException(
            status_code=404,
            detail=f"No implementation registered for strategy '{row.name}'",
        )

    return [ParameterSchema(**p) for p in strategy_class.get_parameters_schema()]
