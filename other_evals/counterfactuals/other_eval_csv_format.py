from pathlib import Path
from typing import Mapping
from git import Sequence
from pydantic import BaseModel
import pandas as pd
from slist import Group, Slist

from evals import response_property

class NotebookColumns(BaseModel):
    # string response_object extracted_property_object extracted_property_meta response_property_at_merge
    string: str
    response_object: str
    extracted_property_object: str
    extracted_property_meta: str
    response_property_at_merge: str
    compliance_meta: bool


class OtherEvalCSVFormat(BaseModel):
    object_history: str
    object_model: str
    object_parsed_result: str
    meta_history: str
    meta_model: str
    meta_parsed_result: str
    meta_predicted_correctly: bool
    eval_name: str



    def to_notebook_columns(self) -> NotebookColumns:
        return NotebookColumns(
            string=self.object_history,
            response_object=self.object_parsed_result,
            extracted_property_object=self.object_parsed_result,
            extracted_property_meta=self.meta_parsed_result,
            response_property_at_merge=self.eval_name,
            compliance_meta=True, # We only include compliant samples for now
        )


def load_all_other_eval_csvs(results_path: str | Path) -> Sequence[OtherEvalCSVFormat]:
    """
    Load a CSV file and plot a heatmap with the mean values and 95% confidence intervals.

    Parameters:
    - csv_path: str, the path to the CSV file.
    """
    all_files_ending_with_csv = list(Path(results_path).rglob("*.csv"))
    assert len(all_files_ending_with_csv) > 0, f"No CSV files found in {results_path}."
    print(f"Found {len(all_files_ending_with_csv)} CSV files.")
    parsed: list[OtherEvalCSVFormat] = []
    for csv_path in all_files_ending_with_csv:
        data = pd.read_csv(csv_path)
        _json = data.to_dict(orient="records")
        parsed.extend([OtherEvalCSVFormat(**row) for row in _json]) # type: ignore
    print(f"Parsed {len(parsed)} other evals samples.")
    return parsed




def to_notebook_groups(parsed: Sequence[OtherEvalCSVFormat]) -> Mapping[tuple[str, str], Sequence[NotebookColumns]]:
    # return object_model, meta_model -> list of OtherEvalCSVFormat
    parsed_ = Slist(parsed)
    # group by object_model and meta_model
    grouped =  parsed_.group_by(lambda x: (x.object_model, x.meta_model))
    mapped = grouped.map_on_group_values(lambda group_items: group_items.map(lambda x: x.to_notebook_columns()))
    return mapped.to_dict()




class FinetuneMessage(BaseModel):
    role: str
    content: str


class FinetuneConversation(BaseModel):
    # Each conversation has multiple messages between the user and the model
    messages: list[FinetuneMessage]

    @property
    def last_message_content(self) -> str:
        return self.messages[-1].content
