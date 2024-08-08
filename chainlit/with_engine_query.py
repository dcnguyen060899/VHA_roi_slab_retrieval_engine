import os
import sys
import openai

openai.api_key = '' # please use your own open API keys
os.environ["OPENAI_API_KEY"] = "" # please use your own open API keys
# os.environ["ACTIVELOOP_TOKEN"] = '' # please use your own token for Deep Lake Database

# %%
# Imports
#
from typing import List


from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.legacy.multi_modal_llms import OpenAIMultiModal

from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.legacy.vector_stores import DeepLakeVectorStore
from pydantic import BaseModel

from llama_index.legacy.readers.deeplake import DeepLakeReader
import random
from llama_index.core.storage.storage_context import StorageContext

from typing import List, Tuple
import deeplake
from PIL import Image
from io import BytesIO
import re
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import pandas as pd
import ipywidgets as widgets
from llama_index.core import set_global_service_context
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
import json


from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

import chainlit as cl
import asyncio

# retrieve all the image_ids we have in the folder
class roi_slabs(BaseModel):
    """Data model for roi and slab combination from the local vector database"""

    slabs: str
    rois: str

class roi_slabs_list(BaseModel):
    """A list of roi_slab combination for the model to use"""

    roislabsList: List[roi_slabs]

import os

# Directory containing the PDF files
pdf_directory = "/content/drive/MyDrive/data_career/voronoi_project/proof_of_concept/show_vince/pdf_output"

# List all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Dictionary to hold the loaded indexes
loaded_indexes = []

# Iterate through each PDF file and attempt to load its index
for pdf_file in pdf_files:
    try:
        # Dynamically create storage context for each PDF
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(pdf_directory, pdf_file)
        )
        # Dynamically load the index from storage
        loaded_index = load_index_from_storage(storage_context)

        # Store the loaded index in a dictionary with a dynamic key
        index_loaded = True
        loaded_indexes.append(index_loaded)
    except Exception as e:
        index_loaded = False
        loaded_indexes.append(index_loaded)


# Extract a single False from the set
index_loaded = next(iter(loaded_indexes))
if not index_loaded:
  print('passed all proceed to vector store initialization')

pdf_files = [
    "FULL_SCAN_combinations.pdf",
    "T12start_to_L5end_combinations.pdf",
    "avg_L3mid1_combinations.pdf",
    "T1_combinations.pdf",
    "T2_combinations.pdf",
    "T3_combinations.pdf",
    "T4_combinations.pdf",
    "T5_combinations.pdf",
    "T6_combinations.pdf",
    "T7_combinations.pdf",
    "T8_combinations.pdf",
    "T9_combinations.pdf",
    "T10_combinations.pdf",
    "T11_combinations.pdf",
    "T12_combinations.pdf",
    "L1_combinations.pdf",
    "L2_combinations.pdf",
    "L3_combinations.pdf",
    "L4_combinations.pdf",
    "L5_combinations.pdf",
    "SACRUM_combinations.pdf",
    "t12start_combinations.pdf",
    "t12mid_combinations.pdf",
    "t12end_combinations.pdf",
    "l1start_combinations.pdf",
    "l1mid_combinations.pdf",
    "l1end_combinations.pdf",
    "l2start_combinations.pdf",
    "l2mid_combinations.pdf",
    "l2end_combinations.pdf",
    "l3start_combinations.pdf",
    "l3mid_combinations.pdf",
    "l3end_combinations.pdf",
    "l4start_combinations.pdf",
    "l4mid_combinations.pdf",
    "l4end_combinations.pdf",
    "l5start_combinations.pdf",
    "l5mid_combinations.pdf",
    "l5end_combinations.pdf",
    "sacrumstart_combinations.pdf",
    "sacrummid_combinations.pdf",
    "sacrumend_combinations.pdf"
]

for pdf_file in pdf_files:
    slab_name = pdf_file.replace('_combinations.pdf', '')
    variable_name_data = f"{slab_name}"
    variable_name_vector_store = f"vector_store_{slab_name}"
    variable_name_storage_context = f"storage_context_{slab_name}"

    # Dynamically execute code to create variables based on PDF file names
    exec(f"""
{variable_name_data} = SimpleDirectoryReader(input_files=["{os.path.join(pdf_directory, pdf_file)}"]).load_data()
{variable_name_vector_store} = VectorStoreIndex.from_documents({variable_name_data})
{variable_name_storage_context} = StorageContext.from_defaults(vector_store={variable_name_vector_store})
""")


for pdf_file in pdf_files:
    slab_name = pdf_file.replace('_combinations.pdf', '')
    variable_name_vector_store = f"vector_store_{slab_name}"
    variable_name_engine = f"{slab_name}_engine"

    # Dynamically execute code to create query engines based on vector store variables
    exec(f"""
{variable_name_engine} = {variable_name_vector_store}.as_query_engine(similarity_top_k=3, output_cls=roi_slabs_list)
""")

print()
llm = OpenAI(model="gpt-4", temperature=0.7)

query_engine_tools = [
  QueryEngineTool(
      query_engine=FULL_SCAN_engine,
      metadata=ToolMetadata(
          name="FULL_SCAN_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'FULL_SCAN',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=L1_engine,
      metadata=ToolMetadata(
          name="L1_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'L1',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
    QueryEngineTool(
      query_engine=L2_engine,
      metadata=ToolMetadata(
          name="L2_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'L2',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=L3_engine,
      metadata=ToolMetadata(
          name="L3_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'L3',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=L4_engine,
      metadata=ToolMetadata(
          name="L4_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'L4',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=L5_engine,
      metadata=ToolMetadata(
          name="L5_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'L5',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=SACRUM_engine,
      metadata=ToolMetadata(
          name="SACRUM_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'SACRUM',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T10_engine,
      metadata=ToolMetadata(
          name="T10_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'avg-L3mid[1]',"
                "    'rois': 'ALLSKM[-29,150]_NOARMS',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T11_engine,
      metadata=ToolMetadata(
          name="T11_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T11',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T12_engine,
      metadata=ToolMetadata(
          name="T12_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T12',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T12start_to_L5end_engine,
      metadata=ToolMetadata(
          name="T12start_to_L5end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T12start_to_L5end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T1_engine,
      metadata=ToolMetadata(
          name="T1_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T1',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T2_engine,
      metadata=ToolMetadata(
          name="T2_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T2',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T3_engine,
      metadata=ToolMetadata(
          name="T3_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T3',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T4_engine,
      metadata=ToolMetadata(
          name="T4_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T4',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T5_engine,
      metadata=ToolMetadata(
          name="T5_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T5',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T6_engine,
      metadata=ToolMetadata(
          name="T6_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T6',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T7_engine,
      metadata=ToolMetadata(
          name="T7_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T7',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T8_engine,
      metadata=ToolMetadata(
          name="T8_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T8',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=T9_engine,
      metadata=ToolMetadata(
          name="T9_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'T9',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=avg_L3mid1_engine,
      metadata=ToolMetadata(
          name="avg_L3mid1_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'avg-L3mid[1]',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l1end_engine,
      metadata=ToolMetadata(
          name="l1end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l1end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l1mid_engine,
      metadata=ToolMetadata(
          name="l1mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l1mid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l1start_engine,
      metadata=ToolMetadata(
          name="l1start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l1start',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l2end_engine,
      metadata=ToolMetadata(
          name="l2end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l2end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l2mid_engine,
      metadata=ToolMetadata(
          name="l2mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l2mid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l2start_engine,
      metadata=ToolMetadata(
          name="l2start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l3end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l3end_engine,
      metadata=ToolMetadata(
          name="l3end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l3end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l3mid_engine,
      metadata=ToolMetadata(
          name="l3mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l3mid',"
                "    'rois': 'ALLSKM_NOARMS',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l3start_engine,
      metadata=ToolMetadata(
          name="l3start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l3start',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l4end_engine,
      metadata=ToolMetadata(
          name="l4end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l4end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l4mid_engine,
      metadata=ToolMetadata(
          name="l4mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l4mid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l4start_engine,
      metadata=ToolMetadata(
          name="l4start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l4start',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l5end_engine,
      metadata=ToolMetadata(
          name="l5end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l5end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l5mid_engine,
      metadata=ToolMetadata(
          name="l5mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l5mid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=l5start_engine,
      metadata=ToolMetadata(
          name="l5start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'l5start',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=sacrumend_engine,
      metadata=ToolMetadata(
          name="sacrumend_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'sacrumend',"
                "    'rois': 'LIV',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=sacrummid_engine,
      metadata=ToolMetadata(
          name="sacrummid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'sacrummid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=sacrumstart_engine,
      metadata=ToolMetadata(
          name="sacrumstart_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 'sacrumstart',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=t12end_engine,
      metadata=ToolMetadata(
          name="t12end_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 't12end',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=t12mid_engine,
      metadata=ToolMetadata(
          name="t12mid_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 't12mid',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
  QueryEngineTool(
      query_engine=t12start_engine,
      metadata=ToolMetadata(
          name="t12start_engine",
          description=(
              "Provides information about the combination of rois and slabs. "
              "return the combination roi and slab in json format."
              "for example:"
                "{"
                "  'publish': {"
                "    'slabs': 't12start_engine',"
                "    'rois': 'ALLSKM',"
                "    'cross_sectional_area': 'true',"
                "    'volume': 'true',"
                "    'hu': 'true'"
                "  }"
                "}"
          ),
      ),
  ),
]

agent = OpenAIAgent.from_tools(
  system_prompt="""
Agents are equipped with access to a comprehensive database that contains all possible combinations of "slabs" (anatomical locations) and "rois" (regions of interest), including specific measurement ranges when applicable. This database is an essential tool for correctly interpreting user requests and translating them into the required JSON format for medical imaging analysis database queries or data processing tasks.

Agents have access to a comprehensive pdf local database containing all possible combinations of "slabs" and "rois", which includes specific measurement ranges for a variety of tissues and organs. These are often referred to by abbreviations:

Slabs:
- FULL_SCAN: Full Body Scan
- T1, T2, T3, ... T12: Thoracic vertebrae 1 through 12
- L1, L2, L3, ... L5: Lumbar vertebrae 1 through 5
- SACRUM: Sacrum
- avg-L3mid[1]: Average at the midpoint of the third lumbar vertebra
- t12start, t12mid, t12end: Start, mid, and end points of the twelfth thoracic vertebra
- l1start, l1mid, l1end: Start, mid, and end points of the first lumbar vertebra
- l2start, l2mid, l2end: Start, mid, and end points of the second lumbar vertebra
- l3start, l3mid, l3end: Start, mid, and end points of the third lumbar vertebra
- l4start, l4mid, l4end: Start, mid, and end points of the fourth lumbar vertebra
- l5start, l5mid, l5end: Start, mid, and end points of the fifth lumbar vertebra
- sacrumstart, sacrummid, sacrumend: Start, mid, and end points of the sacrum

ROIs:
- FULL_SCAN: Full Scan
- ALLSKM: All Skeletal Muscle
- SAT: Subcutaneous Adipose Tissue
- ALLFAT: All Fat Tissue
- ALLIMAT: All Imaging Material
- VAT: Visceral Adipose Tissue
- EpAT: Epidural Adipose Tissue
- PaAT: Paravertebral Adipose Tissue
- ThAT: Thoracic Adipose Tissue
- LIV: Liver
- SPL: Spleen
- AOC: Abdominal Oblique Composite
- CAAC: Combined Abdominal Adipose Compartment
- NOARMS: Excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)

The process for handling user requests is as follows:

1. Review the user's request to identify key terms that correspond to "slabs" and "rois" within the database. For instance, if a user asks for "Visceral adipose tissue at -150, -50" and specifies "sacrum" as the slab, you should recognize "VAT" as the roi and "[-150,-50]" as the measurement range, with "SACRUM" as the slab.
2. Upon receiving a request that mentions "points of the fifth lumbar vertebra" without specifying which point, the agent should ask for clarification by posing a question like, "Could you specify which point of the fifth lumbar vertebra you are interested in? (start, mid, end)"
However, there are exception, the user could want to see lumbar vertebra without point. The agent should ask for clarification by posing a question like, "Could you specify which number of lumbar vertebrae you are interested in? (1, 2, 3, 4, 5)".
Additionally, if the user inputs "the startpoint of the twelfth thoracic vertebra", make sure to ask the user if it goes to the end point of the fifth lumbar vertebra or just simply the the startpoint of the twelfth thoracic vertebra.
  2.1 Note that only lumbar has point of vertebra while only the twelfth thoracic vertebra has (start, mid, end). So do not ask user for point of Thoracic vertebrae if it is the twelfth thoracic vertebra.
      The agent should ask for clarification by posing a question like, "Could you specify which point of the twelfth thoracic vertebra you are interested in? (start, mid, end)"
  2.2 The agent should wait for the user to provide the specific point before generating a response.
  2.3 Once the specific point is provided, the agent can then proceed to generate the appropriate JSON response based on the user's clarified request.

3. If the user's request is vague or lacks specific details necessary for creating a structured JSON (e.g., the exact anatomical location or the region of interest (ROI) is not mentioned), you are to engage in a clarification process. Ask targeted follow-up questions to obtain the necessary details, such as the precise anatomical location and the specific ROI the user is interested in.
  3.1 If the location is specified but not the ROI, ask for further clarification (e.g., "Which region of interest (ROI) for the midpoint of the third lumbar vertebra are you interested in?").

4. Verify if the slabs and ROIs mentioned in the user's request match any of the predefined abbreviations. This step involves parsing the user input and checking against the list of valid "slabs" and "ROIs".
  4.1 If the user's input does not clearly match the predefined abbreviations or if there's a misunderstanding in the input (e.g., "l5midALLSKM[-29,150]" where slabs and ROIs are mixed up), the agent should ask for clarification by providing examples or asking the user to select from the list of valid options.
  4.2 The agent should provide a clear format or examples on how to specify slabs and ROIs correctly. For instance, "Please specify the slab and ROI separately. For example, 'I want to see l5mid for ALLSKM at -29, 150 excluding arms'."
  4.3 Wait for the user to provide a clarified request that matches the predefined abbreviations and format.

5. Please specify your request by clearly separating the anatomical location (slab) and the region of interest (ROI) using a comma or the word 'for'. For example, 'l1start for ALLSKM' or 'l1start, ALLSKM'. This helps in accurately processing your request and ensuring precise retrieval of data.

6. Accurately translate the request into JSON, using the exact terminology and format from the database. Ensure "slabs" and "rois" are correctly identified and formatted, reflecting the specific combination found in the database.

7. When a user requests imaging for specific anatomical locations and regions of interest without a specified measurement range, agents must identify the correct "slabs" and "rois" terms from the database. It is important to use the exact terminology and format for the slabs and rois as found in the database to avoid any ambiguity or error in the JSON response.

For example:
If a user requests "the start point of the twelfth thoracic vertebra and the end point of the fifth lumbar vertebra for all imaging materials," the agent must recognize the slab as "T12start-to-L5end" and the ROI as "ALLIMAT". The agent must not incorrectly separate these into "t12start" and "l5end" but should maintain the combined slab identifier as per the database.

The agent's response should be structured as follows:
>>> User: "I want to see the start point of the twelfth thoracic vertebra and the end point of the fifth lumbar vertebra for all imaging materials."
>>> Agent: Correctly recognizes the request and prepares the JSON output.
>>> Final JSON Output:
>>> {
  "publish": {
    "slabs": "T12start-to-L5end",
    "rois": "ALLIMAT",
    "cross_sectional_area": "true",
    "volume": "true",
    "hu": "true"
  }
}
So it is important to differentiate between t12start and T12start that goes to other point of other vertebra (T12start-to-L5end).

8. Lastly, make sure to differential between l3mid and avg-l3mid[1]. For example:
>>> User: "can i see mid point of the third lumbar of all skeletal muscle at -29, 150"
>>> Agent: Correctly recognizes the request and prepares the JSON output.
>>> Final JSON Output:
>>> {
  "publish": {
    "slabs": "l3mid",
    "rois": "ALLSKM[-29, 150]",
    "cross_sectional_area": "true",
    "volume": "true",
    "hu": "true"
  }
}

9. Agents must be adept at distinguishing between requests for specific sacral points (sacrum end, mid, and start) and queries that necessitate
 the broader application of the "SACRUM" engine. This distinction is crucial for accurately processing user requests and ensuring the provision of
  relevant data.

10. When processing requests involving specific measurements (such as length, volume, or area) related to anatomical points or regions of interest (ROIs), 
agents must ensure these measurements align with the ranges available in our database. This is crucial for delivering accurate and relevant data to users.
  10.1 Given the constraints mentioned, it's important to verify that the range of measurements specified for the region of interest (ROI) falls within the database's valid parameters.
   If the measurement range of "-900, 2999" for the ROI "ALLIMAT" does not match the acceptable ranges in our database, the agents ask the user to re-input.


Example Process for a Specific Request:
>>> User: "I want to see the midpoint of the third lumbar vertebra."
>>> Agent: Recognizes "avg-L3mid[1]" as the slab but missing roi. Proceed to ask: "Where would you like to see the midpoint of the third lumbar vertebra?"
>>> User: "I want to see the average at the midpoint of the third lumbar vertebra of all skeletal muscle at -29, 150 excluding Arms (used in conjunction with other tissue types to exclude measurements from the arms)"
>>> Agent: Recognizes "avg-L3mid[1]" as the slab and "ALLSKM[-29,150]_NOARMS" as the roi based on the database combinations.

>>> Final Input:
Added user message to memory: of all the skeletal muscle
=== Calling Function ===
Calling function: roi_slab_retrieval with args: {
  "input": "avg-L3mid[1] ALLSKM[-29,150]_NOARMS"
}
Got output: {"roislabsList":[{"slabs":"avg-L3mid[1]","rois":"ALLSKM[-29,150]_NOARMS"}]}

>>>Final JSON Output:
>>> {
  "publish": {
  "slabs": "avg-L3mid[1]",
  "rois": "ALLSKM[-29,150]_NOARMS",
  "cross_sectional_area": "true",
  "volume": "true",
  "hu": "true"
  }
}

Make sure your output seperate the user input into slabs and rois clear. Here is a terible example of output:
Added user message to memory: of all the skeletal muscle
=== Calling Function ===
Calling function: roi_slab_retrieval with args: {
  "input": "l1mid ALLSKM"
}
Got output: {"roislabsList":[{"slabs":"l1midALLSKM","rois":"ALLSKM"}]}
========================
""",
# Your responses should leverage the database to provide accurate and detailed structures for user requests in JSON format, ensuring a precise match between the user's description and the database's "slabs" and "rois" combinations.

                               tools=query_engine_tools,
                               llm=llm,
                               verbose=True)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the query engine from the user's session.
    query_engine = cl.user_session.get("query_engine")  # Assuming this is a RetrieverQueryEngine

    # The incoming user message is passed to the agent.
    # Assuming 'agent.chat' is the correct method to get the response from your agent.
    # You might need to adjust this part based on how your agent is set up.
    prompt = message.content  # The message from the user is the prompt for the agent.

    # Generate the response using the agent. Ensure that this is awaited if it's an async call.
    agent_response = await asyncio.get_running_loop().run_in_executor(None, agent.chat, prompt)

    # Extract the text response from the agent's response.
    response_text = agent_response.response  # Adjust based on how the response is structured.

    # Create a new Chainlit message with the agent's response and send it.
    response_message = cl.Message(content=response_text)
    await response_message.send()
