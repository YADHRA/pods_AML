from pydantic import BaseModel


class DateRange(BaseModel):
    start: str
    end: str


class LabelCounts(BaseModel):
    normal: int
    laundering: int


class DashboardResponse(BaseModel):
    total_transactions: int
    total_nodes: int
    date_range: DateRange
    labels: LabelCounts
    payment_formats: dict[str, int]


class TransactionResponse(BaseModel):
    txn_id: int
    timestamp: str
    date: str
    src_node: int
    dst_node: int
    src_bank: str
    dst_bank: str
    amount_paid: float
    amount_received: float
    currency_paid: str
    currency_received: str
    payment_format: str
    is_laundering: int
    split: str
    is_self_loop: bool


class AccountResponse(BaseModel):
    node_id: int
    bank: str
    account: str
    date: str
    pagerank_log: float
    wcc_size_log: float
    unique_out_log: float
    unique_in_log: float
    has_sent_before: int
    has_received_before: int


class GraphNode(BaseModel):
    id: int
    bank: str
    account: str
    is_center: bool


class GraphEdge(BaseModel):
    source: int
    target: int


class GraphResponse(BaseModel):
    node_id: int
    date: str
    depth: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
class ModelPrediction(BaseModel):
    score: float
    alert: bool


class PredictionResponse(BaseModel):
    txn_id: int
    split: str
    is_laundering: int
    new_pair: bool
    m1_full: ModelPrediction
    m1_noflags: ModelPrediction
    m2_full: ModelPrediction
    m2_noflags: ModelPrediction

class AlertItem(BaseModel):
    txn_id: int
    split: str
    is_laundering: int
    new_pair: bool
    score: float
    alert: bool


class AlertsResponse(BaseModel):
    split: str
    model: str
    variant: str
    total: int
    limit: int
    offset: int
    items: list[AlertItem]

