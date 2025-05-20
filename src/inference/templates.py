from typing import Type, Literal, Any, Dict
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
import os
import json
import outlines

from .utils import *
from .constants import *

class Rubric(BaseModel):
    metric_name: str
    metric_description: str
    options: Dict[str, str]

@outlines.prompt
def construct_judge_prompt(paper_text: str, rubric: Rubric, format: str):
    """
    ### INSTRUCTION
    Carefully evaluate the quality and characteristics of the **new datasets** introduced in the paper using the rubric provided below.

    Please follow these rules:
    - **Only assess new datasets** that are introduced by the authors. Do **not** evaluate any pre-existing datasets mentioned in the paper.
    - Base your judgments **strictly on the content of the paper**. Do **not** infer or speculate beyond what is explicitly stated.
    - Use the rubric definitions to guide your labeling. Provide clear references and reasoning wherever applicable.
    
    ### PAPER
    {{ paper_text }}

    ### RUBRIC
    Metric: {{ rubric.metric_name }}
    Description: {{ rubric.metric_description }}

    Options:
    {% for key, value in rubric.options.items() %}
    {{ key }}: {{ value }}
    {% endfor %}

    ### RESPONSE FORMAT
    Return a JSON response in the following format:
    
    {{ format }}
    
    ### RESPONSE
    """