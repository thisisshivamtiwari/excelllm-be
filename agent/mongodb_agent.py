"""
MongoDB-based LangChain Agent
Orchestrates deterministic tools for user queries
"""

import os
import json
import logging
import uuid
import colorsys
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

# Import tools
from tools.mongodb_tools import (
    table_loader,
    agg_helper,
    timeseries_analyzer,
    compare_entities,
    calc_eval,
    statistical_summary,
    list_user_files,
    rank_entities
)

logger = logging.getLogger(__name__)


def create_tool_wrappers(user_id: str, file_id: Optional[str] = None) -> List[Tool]:
    """
    Create LangChain Tool wrappers for MongoDB tools.
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: Optional file ID (can be None for multi-file queries)
    
    Returns:
        List of LangChain Tool objects
    """
    
    def wrap_table_loader(query: str) -> str:
        """Load table sample and schema. Use format: file_id|table_name|file_name_pattern|filters_json|fields_json|limit
        file_id can be "*" for all files, table_name can be "*" for all sheets, file_name_pattern for filename search"""
        try:
            parts = query.split("|")
            file_id_param = parts[0].strip() if len(parts) > 0 and parts[0].strip() else (file_id or "*")
            table_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "*"
            file_name_pattern = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            filters = json.loads(parts[3]) if len(parts) > 3 and parts[3].strip() else None
            fields = json.loads(parts[4]) if len(parts) > 4 and parts[4].strip() else None
            limit = int(parts[5]) if len(parts) > 5 and parts[5].strip() else 100
            
            result = table_loader(user_id, file_id_param, table_name, file_name_pattern, filters, fields, limit)
            return json.dumps(result, default=str)
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"JSON decode error: {str(e)}"})
        except ValueError as e:
            return json.dumps({"ok": False, "error": f"Value error: {str(e)}"})
        except Exception as e:
            import traceback
            logger.error(f"Error in wrap_table_loader: {str(e)}\n{traceback.format_exc()}")
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_agg_helper(query: str) -> str:
        """Run aggregations. Use format: file_id|table_name|file_name_pattern|filters_json|metrics_json|date_filter_json|group_by_source
        file_id can be "*" for all files, table_name can be "*" for all sheets"""
        try:
            parts = query.split("|")
            file_id_param = parts[0].strip() if len(parts) > 0 and parts[0].strip() else (file_id or "*")
            table_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "*"
            file_name_pattern = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            filters = json.loads(parts[3]) if len(parts) > 3 and parts[3].strip() else None
            metrics = json.loads(parts[4]) if len(parts) > 4 and parts[4].strip() else []
            date_filter = json.loads(parts[5]) if len(parts) > 5 and parts[5].strip() else None
            group_by_source = parts[6].lower() == "true" if len(parts) > 6 and parts[6].strip() else False
            
            result = agg_helper(user_id, file_id_param, table_name, file_name_pattern, filters, metrics, date_filter, True, group_by_source)
            return json.dumps(result, default=str)
        except Exception as e:
            import traceback
            logger.error(f"Error in wrap_agg_helper: {str(e)}\n{traceback.format_exc()}")
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_timeseries_analyzer(query: str) -> str:
        """Analyze time series. Use format: file_id|table_name|time_col|metric_col|freq|agg|start|end"""
        try:
            parts = query.split("|")
            file_id_param = parts[0] if len(parts) > 0 else file_id
            table_name = parts[1] if len(parts) > 1 else "Sheet1"
            time_col = parts[2] if len(parts) > 2 else None
            metric_col = parts[3] if len(parts) > 3 else None
            freq = parts[4] if len(parts) > 4 else "month"
            agg = parts[5] if len(parts) > 5 else "sum"
            start = datetime.fromisoformat(parts[6]) if len(parts) > 6 and parts[6] else None
            end = datetime.fromisoformat(parts[7]) if len(parts) > 7 and parts[7] else None
            
            if not file_id_param or not time_col or not metric_col:
                return json.dumps({"error": "file_id, time_col, and metric_col required"})
            
            result = timeseries_analyzer(user_id, file_id_param, table_name, time_col, metric_col, freq, agg, start, end)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_compare_entities(query: str) -> str:
        """Compare two entities. Use format: file_id|table_name|key_col|metric_col|entity_a|entity_b|agg|filters_json"""
        try:
            parts = query.split("|")
            file_id_param = parts[0] if len(parts) > 0 else file_id
            table_name = parts[1] if len(parts) > 1 else "Sheet1"
            key_col = parts[2] if len(parts) > 2 else None
            metric_col = parts[3] if len(parts) > 3 else None
            entity_a = parts[4] if len(parts) > 4 else None
            entity_b = parts[5] if len(parts) > 5 else None
            agg = parts[6] if len(parts) > 6 else "sum"
            filters = json.loads(parts[7]) if len(parts) > 7 and parts[7] else None
            
            if not file_id_param or not key_col or not metric_col or not entity_a or not entity_b:
                return json.dumps({"error": "file_id, key_col, metric_col, entity_a, and entity_b required"})
            
            result = compare_entities(user_id, file_id_param, table_name, key_col, metric_col, entity_a, entity_b, agg, filters)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_calc_eval(query: str) -> str:
        """Evaluate mathematical expression. Use format: expression_string"""
        try:
            result = calc_eval(query)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_statistical_summary(query: str) -> str:
        """Get statistical summary. Use format: file_id|table_name|columns_json|filters_json"""
        try:
            parts = query.split("|")
            file_id_param = parts[0] if len(parts) > 0 else file_id
            table_name = parts[1] if len(parts) > 1 else "Sheet1"
            columns = json.loads(parts[2]) if len(parts) > 2 and parts[2] else []
            filters = json.loads(parts[3]) if len(parts) > 3 and parts[3] else None
            
            if not file_id_param or not columns:
                return json.dumps({"error": "file_id and columns required"})
            
            result = statistical_summary(user_id, file_id_param, table_name, columns, filters)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_list_user_files(query: str) -> str:
        """List all files available for the user. No parameters needed."""
        try:
            result = list_user_files(user_id)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_rank_entities(query: str) -> str:
        """Rank entities by aggregated metric. Use format: file_id|table_name|key_col|metric_col|agg|n|order|filters_json"""
        try:
            parts = query.split("|")
            file_id_param = parts[0] if len(parts) > 0 else file_id
            table_name = parts[1] if len(parts) > 1 else "Sheet1"
            key_col = parts[2] if len(parts) > 2 else None
            metric_col = parts[3] if len(parts) > 3 else None
            agg = parts[4] if len(parts) > 4 else "sum"
            n = int(parts[5]) if len(parts) > 5 and parts[5] else 5
            order = parts[6] if len(parts) > 6 else "desc"
            filters = json.loads(parts[7]) if len(parts) > 7 and parts[7] else None
            
            if not file_id_param or not key_col or not metric_col:
                return json.dumps({"error": "file_id, key_col, and metric_col required"})
            
            result = rank_entities(user_id, file_id_param, table_name, key_col, metric_col, agg, n, order, filters)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_get_date_range(query: str) -> str:
        """Get date range information for a time column. Use format: file_id|table_name|time_col"""
        try:
            parts = query.split("|")
            file_id_param = parts[0] if len(parts) > 0 else file_id
            table_name = parts[1] if len(parts) > 1 else "Sheet1"
            time_col = parts[2] if len(parts) > 2 else None
            
            if not file_id_param or not time_col:
                return json.dumps({"error": "file_id and time_col required"})
            
            from tools.mongodb_tools import get_date_range
            result = get_date_range(user_id, file_id_param, table_name, time_col)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_detect_date_columns(query: str) -> str:
        """Detect date columns across files/sheets. Use format: file_id|table_name|file_name_pattern"""
        try:
            parts = query.split("|")
            file_id_param = parts[0].strip() if len(parts) > 0 and parts[0].strip() else (file_id or "*")
            table_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "*"
            file_name_pattern = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            
            from tools.mongodb_tools import detect_date_columns
            result = detect_date_columns(user_id, file_id_param, table_name, file_name_pattern)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_extract_dates_from_filenames(query: str) -> str:
        """Extract dates from file names. Use format: file_name_pattern"""
        try:
            file_name_pattern = query.strip() if query.strip() else None
            
            from tools.mongodb_tools import extract_dates_from_filenames
            result = extract_dates_from_filenames(user_id, file_name_pattern)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_parse_relative_date(query: str) -> str:
        """Parse relative date expressions. Use format: date_expression"""
        try:
            from tools.mongodb_tools import parse_relative_date
            result = parse_relative_date(query.strip())
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    def wrap_search_across_all_files(query: str) -> str:
        """Search for a column across all files and sheets. Use format: column_name|search_value|file_name_pattern|table_name_pattern|limit"""
        try:
            parts = query.split("|")
            column_name = parts[0].strip() if len(parts) > 0 and parts[0].strip() else None
            search_value = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
            file_name_pattern = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            table_name_pattern = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
            limit = int(parts[4]) if len(parts) > 4 and parts[4].strip() else 100
            
            if not column_name:
                return json.dumps({"ok": False, "error": "column_name required"})
            
            from tools.mongodb_tools import search_across_all_files
            result = search_across_all_files(user_id, column_name, search_value, file_name_pattern, table_name_pattern, limit)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
    
    tools = [
        Tool(
            name="list_user_files",
            func=wrap_list_user_files,
            description="""List all files available for the user.
            Input: (no parameters needed, just call with empty string or "list")
            Returns: List of files with file_id, filename, file_type, table_names, and row_count.
            ALWAYS call this FIRST if you don't know which file_id to use."""
        ),
        Tool(
            name="table_loader",
            func=wrap_table_loader,
            description=f"""Load table sample and schema from MongoDB. 
            Input format: file_id|table_name|file_name_pattern|filters_json|fields_json|limit
            file_id can be "*" for all files, table_name can be "*" for all sheets
            file_name_pattern: search files by name (e.g., "July 25")
            {"Default file_id: " + file_id if file_id else "file_id can be '*' for all files"}
            Returns: schema, sample_rows, row_count, schemas_by_source. Always call this first to inspect available columns."""
        ),
        Tool(
            name="agg_helper",
            func=wrap_agg_helper,
            description="""Run deterministic aggregations (sum, avg, count, min, max, median).
            Input format: file_id|table_name|file_name_pattern|filters_json|metrics_json|date_filter_json|group_by_source
            file_id can be "*" for all files, table_name can be "*" for all sheets
            date_filter_json: {"column": "date_col", "start": "2025-01-01", "end": "2025-12-31", "auto_detect": true}
            group_by_source: "true" to aggregate separately per file/sheet
            metrics_json format: [{"op":"sum","col":"column_name","alias":"result_name"}]
            Returns: aggregated values as Decimal for accuracy.
            USE THIS FOR: Questions asking "What is the mean/average/total/sum/count/min/max/median of X?"
            Examples: "What is the mean of opening stock?", "What is the total revenue?", "What is the average downtime?"
            DO NOT use this for "which entity has highest" questions - use rank_entities instead."""
        ),
        Tool(
            name="timeseries_analyzer",
            func=wrap_timeseries_analyzer,
            description="""Analyze time series data with trend calculation.
            Input format: file_id|table_name|time_col|metric_col|freq|agg|start|end
            freq: day|week|month|year, agg: sum|avg|count|min|max
            Returns: series data and slope/trend. Use for time-based questions."""
        ),
        Tool(
            name="compare_entities",
            func=wrap_compare_entities,
            description="""Compare two entities side-by-side.
            Input format: file_id|table_name|key_col|metric_col|entity_a|entity_b|agg|filters_json
            Returns: values for both entities and percent difference. Use for comparison questions."""
        ),
        Tool(
            name="statistical_summary",
            func=wrap_statistical_summary,
            description="""Get statistical summary (min/max/mean/median/std) for numeric columns.
            Input format: file_id|table_name|columns_json|filters_json
            Returns: statistical measures (min, max, mean, median, std) for each column.
            USE THIS FOR: Questions asking for multiple statistics at once (e.g., "What are the statistics for X?")
            Examples: "Get statistics for downtime", "What are the min/max/mean for opening stock?"
            For single statistics (just mean, just total), use agg_helper instead."""
        ),
        Tool(
            name="calc_eval",
            func=wrap_calc_eval,
            description="""Safe deterministic calculator using Decimal precision.
            Input format: mathematical expression (e.g., "123.45 + 67.89")
            Returns: calculated value. Use for final arithmetic after getting values from agg_helper."""
        ),
        Tool(
            name="rank_entities",
            func=wrap_rank_entities,
            description="""Rank entities by aggregated metric (top N or bottom N).
            Input format: file_id|table_name|key_col|metric_col|agg|n|order|filters_json
            key_col: Entity identifier column (e.g., "Product", "Material_Name", "Supplier", "Operator")
            metric_col: Metric column to aggregate (e.g., "Sales", "Consumption_Kg", "Actual_Quantity")
            agg: Aggregation operation ("sum", "avg", "count", "min", "max")
            n: Number of top/bottom entities (default: 5)
            order: "desc" for top N (highest), "asc" for bottom N (lowest)
            Returns: Ranked list of entities with their aggregated values.
            USE THIS FOR: Questions asking "Which entity has the highest/lowest X?" or "Top N entities by X"
            Examples: "Which supplier provided the most?", "Which operator had the highest total?", "Top 5 products by sales"
            DO NOT use this for simple aggregation questions like "What is the mean of X?" - use agg_helper instead."""
        ),
        Tool(
            name="get_date_range",
            func=wrap_get_date_range,
            description="""Get date range information for a time column (min date, max date, row count, span in days).
            Input format: file_id|table_name|time_col
            Returns: min_date, max_date, row_count, span_days. Use to check if data is too large before analyzing."""
        ),
        Tool(
            name="detect_date_columns",
            func=wrap_detect_date_columns,
            description="""Auto-detect all columns containing date values across files/sheets.
            Input format: file_id|table_name|file_name_pattern
            file_id can be "*" for all files, table_name can be "*" for all sheets
            Returns: date_columns_by_source with confidence, min_date, max_date for each column.
            ALWAYS call this FIRST when question involves dates to find which columns contain dates."""
        ),
        Tool(
            name="extract_dates_from_filenames",
            func=wrap_extract_dates_from_filenames,
            description="""Extract date information from file names.
            Input format: file_name_pattern (e.g., "July 25", "2025-07-25", "Q1 2025")
            Returns: files_with_dates containing extracted dates from filenames.
            Use when user mentions a file name that might contain a date."""
        ),
        Tool(
            name="parse_relative_date",
            func=wrap_parse_relative_date,
            description="""Parse relative date expressions and convert to absolute dates.
            Input format: date_expression (e.g., "last month", "this week", "last 30 days", "Q1 2025")
            Returns: start_date, end_date, type. Use to convert relative dates to absolute date ranges."""
        ),
        Tool(
            name="search_across_all_files",
            func=wrap_search_across_all_files,
            description="""Search for a column across ALL files and sheets.
            Input format: column_name|search_value|file_name_pattern|table_name_pattern|limit
            Returns: results_by_source grouped by file/sheet. Use when user asks to find something across all files."""
        )
    ]
    
    return tools


def create_agent_prompt() -> PromptTemplate:
    """Create system prompt for agent"""
    
    prompt = """You are an expert data analyst assistant. Your role is to answer questions about data stored in MongoDB using ONLY the provided tools.

CRITICAL RULES FOR MULTI-FILE/MULTI-SHEET SEARCHES:

1. SEARCH SCOPE:
   - Use file_id="*" or table_name="*" to search ALL files/sheets
   - Use file_name_pattern="July 25" to search files by name
   - Default: searches single file/sheet (backward compatible)

2. WHEN TO SEARCH ALL FILES/SHEETS:
   - User mentions "all files", "all workbooks", "across files"
   - User mentions specific file name (e.g., "July 25")
   - User asks about data that might be in multiple sheets
   - Question doesn't specify a file/sheet

3. FILE NAME REFERENCES:
   - If user mentions a file name (e.g., "July 25"), use file_name_pattern="July 25"
   - First call list_user_files to see available files and their exact names
   - Match file names case-insensitively

4. HANDLING SAME SHEET/COLUMN NAMES:
   - When searching across files, results include file_id and file_name
   - When searching across sheets, results include table_name
   - Use group_by_source=True in agg_helper to get per-file/sheet breakdowns

CRITICAL RULES FOR DATE QUERIES:

1. DATE COLUMN DETECTION:
   - ALWAYS call detect_date_columns FIRST when question involves dates
   - Dates can be in ANY column (not just "date" or "time")
   - Use detect_date_columns(user_id, file_id="*", table_name="*") to find all date columns

2. DATE FROM FILE NAMES:
   - If user mentions file name with date (e.g., "July 25"), use extract_dates_from_filenames
   - File names can contain: "July 25", "2025-07-25", "Q1 2025", etc.
   - Match files by date extracted from filename

3. RELATIVE DATE EXPRESSIONS:
   - "last month" → previous month (start to end)
   - "this week" → current week (Monday to today)
   - "last 30 days" → 30 days ago to today
   - "Q1 2025" → January 1 - March 31, 2025
   - Use parse_relative_date to convert to absolute dates

4. CURRENT DATE RELATIONS:
   - Always use current date as reference for relative dates
   - "last month" means previous calendar month
   - "this week" means current week (Monday-Sunday)

GENERAL CRITICAL RULES:
1. ALWAYS call table_loader FIRST to inspect schema and available columns
2. NEVER compute numbers in your text - ALWAYS use agg_helper, timeseries_analyzer, or calc_eval
3. For time-based questions, use timeseries_analyzer
4. For comparison questions, use compare_entities
5. For final arithmetic, use calc_eval (never do math yourself)
6. If required fields are missing, respond with: insufficient_data: [list_missing_columns]
7. Always include provenance in your final answer

TOOL USAGE EXAMPLES:

Example 1: "What is total revenue across all files?"
- Step 1: agg_helper(user_id, file_id="*", table_name="*", metrics=[{"op":"sum","col":"revenue"}])

Example 2: "Show revenue from July 25 file"
- Step 1: agg_helper(user_id, file_name_pattern="July 25", table_name="*", metrics=[{"op":"sum","col":"revenue"}])

Example 3: "Compare revenue across all sheets in July 25"
- Step 1: agg_helper(user_id, file_name_pattern="July 25", table_name="*", group_by_source=True, metrics=[{"op":"sum","col":"revenue"}])

Example 4: "Find all rows with Product='Widget' in all files"
- Step 1: search_across_all_files(user_id, column_name="Product", search_value="Widget")

Example 5: "What columns exist in July 25 file?"
- Step 1: table_loader(user_id, file_name_pattern="July 25", table_name="*")

Example 6: "Revenue for last month"
- Step 1: detect_date_columns(user_id, file_id="*") to find date columns
- Step 2: parse_relative_date("last month") to get date range
- Step 3: agg_helper(user_id, file_id="*", date_filter={"start": "2025-06-01", "end": "2025-06-30", "auto_detect": true})

Example 7: "Data from July 25 file"
- Step 1: extract_dates_from_filenames(user_id, file_name_pattern="July 25")
- Step 2: Use file_id from result in queries

Example 8: "Compare this week vs last week"
- Step 1: parse_relative_date("this week") → get date range
- Step 2: parse_relative_date("last week") → get date range
- Step 3: Run two separate agg_helper calls with respective date filters
- Step 4: Compare results

Example 9: "Total sales in last 30 days across all files"
- Step 1: detect_date_columns(user_id, file_id="*")
- Step 2: parse_relative_date("last 30 days")
- Step 3: agg_helper(user_id, file_id="*", table_name="*", date_filter={"start": "...", "end": "...", "auto_detect": true})

RESPONSE FORMAT:
- When results span multiple files/sheets, clearly indicate source (file_name::table_name)
- Show breakdown by source when relevant
- Include provenance showing which files/sheets were searched
- Provide a clear, concise answer
- Include numeric values with units if applicable
- Mention which tools were used
- Include provenance (mongo_pipeline, matched_row_count) when available

Current question: {{input}}

Available tools: {{tools}}

Tool names: {{tool_names}}

{agent_scratchpad}"""
    
    return PromptTemplate.from_template(prompt)


def create_agent_executor(
    llm: BaseLanguageModel,
    user_id: str,
    file_id: Optional[str] = None,
    max_iterations: int = 25
) -> AgentExecutor:
    """
    Create LangChain agent executor with MongoDB tools.
    
    Args:
        llm: Language model instance
        user_id: User ID for multi-tenant filtering
        file_id: Optional file ID
        max_iterations: Maximum agent iterations
    
    Returns:
        AgentExecutor instance
    """
    tools = create_tool_wrappers(user_id, file_id)
    
    # Create ReAct prompt with required variables
    # Build context about file_id if available
    file_context = ""
    if file_id:
        file_context = f"\n\nIMPORTANT: The file_id to use is: {file_id}\nUse this EXACT file_id in all tool calls. Do NOT use any other file_id."
    else:
        file_context = "\n\nIMPORTANT: No file_id provided. You MUST call list_user_files FIRST to discover available files, then use a real file_id from the results."
    
    react_prompt = PromptTemplate.from_template(f"""Answer the following questions as best you can. You have access to the following tools:

{{tools}}
{file_context}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action (use EXACT format: file_id|table_name|...)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES FOR MULTI-FILE/MULTI-SHEET:
1. SEARCH SCOPE: Use file_id="*" or table_name="*" to search ALL files/sheets
2. FILE NAME SEARCH: Use file_name_pattern="July 25" when user mentions file names
3. If you don't know which file_id to use, ALWAYS call list_user_files FIRST to see available files
4. After getting files from list_user_files, pick the most relevant file_id based on the question
5. ALWAYS call table_loader FIRST to inspect schema (after you have the correct file_id or "*")
6. When searching across files/sheets, use group_by_source=True to get per-source breakdowns

EFFICIENCY RULES (IMPORTANT - Follow these to avoid hitting iteration limits):
1. MINIMIZE ITERATIONS: Answer questions in 3-5 tool calls maximum when possible
2. COMBINE OPERATIONS: Use agg_helper with date_filter instead of separate detect_date_columns + parse_relative_date + agg_helper calls
3. DIRECT TOOL SELECTION: Choose the right tool immediately based on question type (see TOOL SELECTION GUIDE below)
4. AVOID REDUNDANT CALLS: Don't call table_loader multiple times for the same file/sheet
5. USE WILDCARDS: When question says "all files" or "all sheets", use "*" immediately - don't iterate through each one
6. STOP EARLY: Once you have the answer, provide Final Answer immediately - don't make additional unnecessary tool calls
7. BATCH OPERATIONS: When possible, use single agg_helper call with multiple metrics instead of multiple separate calls

CRITICAL RULES FOR DATE QUERIES:
1. ALWAYS call detect_date_columns FIRST when question involves dates
2. Use extract_dates_from_filenames when user mentions file names with dates
3. Use parse_relative_date for expressions like "last month", "this week", "last 30 days"
4. Apply date filters using date_filter parameter in agg_helper
5. COMBINE DATE OPERATIONS: When using dates, combine detect_date_columns + parse_relative_date + agg_helper in minimal steps

GENERAL CRITICAL RULES:
1. NEVER compute numbers yourself - use agg_helper, timeseries_analyzer, statistical_summary, or calc_eval
2. Include provenance in your answer
3. If file_id is provided above, use that EXACT file_id - do NOT invent or guess file_ids like "12345" or "1"
4. Default table_name is "*" (all sheets) unless specified otherwise - use "Sheet1" only if question specifies a single sheet
5. If table_loader returns "no_rows", you're using the wrong file_id - call list_user_files again and try a different file_id
6. NEVER use placeholder file_ids - always use real file_ids from list_user_files results
7. For time series questions requesting charts, ensure the response includes chart_config with chart_type, data, and options

TOOL SELECTION GUIDE - CHOOSE THE RIGHT TOOL BASED ON QUESTION TYPE:

A. For AGGREGATION questions (mean, average, total, sum, count, min, max, median):
   → Use agg_helper
   Examples: "What is the mean of X?", "What is the total Y?", "What is the average Z?"
   Format: file_id|table_name|filters_json|metrics_json
   metrics_json: [{{{{"op":"avg","col":"column_name","alias":"result"}}}}] for mean/average
                 [{{{{"op":"sum","col":"column_name","alias":"result"}}}}] for total/sum
                 [{{{{"op":"count","col":"column_name","alias":"result"}}}}] for count
                 [{{{{"op":"min","col":"column_name","alias":"result"}}}}] for minimum
                 [{{{{"op":"max","col":"column_name","alias":"result"}}}}] for maximum
                 [{{{{"op":"median","col":"column_name","alias":"result"}}}}] for median

B. For STATISTICAL SUMMARY questions (multiple stats at once):
   → Use statistical_summary
   Examples: "What are the statistics for X?", "Get min/max/mean for Y"
   Format: file_id|table_name|columns_json|filters_json

C. For RANKING/TOP N questions (which entity has highest/lowest):
   → Use rank_entities ONLY when question asks "which", "who", "top N", "bottom N"
   Examples: "Which supplier provided the most?", "Top 5 products by sales"
   Format: file_id|table_name|key_col|metric_col|agg|n|order|filters_json
   DO NOT use rank_entities for simple aggregation questions like "What is the mean?"

D. For COMPARISON questions (compare two specific entities):
   → Use compare_entities
   Examples: "Compare Product A vs Product B", "How does X compare to Y?"
   Format: file_id|table_name|key_col|metric_col|entity_a|entity_b|agg|filters_json

E. For TIME SERIES/TREND questions:
   → Use timeseries_analyzer
   Examples: "Show sales over time", "What is the trend of X?"
   Format: file_id|table_name|time_col|metric_col|freq|agg|start|end

F. For ARITHMETIC calculations (after getting values):
   → Use calc_eval
   Examples: "What is 15% of total?", "Calculate X * Y"
   Format: mathematical expression

IMPORTANT: 
- Questions asking "What is the mean/average/total of X?" should use agg_helper, NOT rank_entities
- Questions asking "Which entity has the highest X?" should use rank_entities
- Always read the question carefully and match it to the correct tool category above

Question: {{input}}
Thought: {{agent_scratchpad}}""")
    
    agent = create_react_agent(llm, tools, react_prompt)
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors="Check your output and make sure it conforms to the format instructions. If you see an error, retry with the correct format.",
        return_intermediate_steps=True,
        max_execution_time=300  # 5 minutes max execution time
    )
    
    return executor


def get_llm_instance(provider: str = "gemini", temperature: float = 0.0):
    """
    Get LLM instance based on provider.
    
    Args:
        provider: "gemini" or "groq"
        temperature: Temperature setting (0.0 for deterministic)
    
    Returns:
        LLM instance
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        model_name = os.getenv("AGENT_MODEL_NAME", "llama-4-maverick-17b-128e-instruct")
        
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'gemini' or 'groq'")


async def execute_agent_query(
    question: str,
    user_id: str,
    file_id: Optional[str] = None,
    provider: str = "gemini",
    max_iterations: int = 25,
    conversation_id: Optional[str] = None,
    date_range: Optional[Dict[str, Optional[str]]] = None
) -> Dict[str, Any]:
    """
    Execute agent query and return structured response.
    
    Args:
        question: User's question
        user_id: User ID
        file_id: Optional file ID
        provider: LLM provider
        max_iterations: Maximum iterations
    
    Returns:
        Structured response with answer, provenance, and metadata
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Initialize conversation context variables
    requires_date_range = False
    date_range_info = None
    conv_service = None
    
    try:
        # Handle conversation context
        try:
            from services.conversation_service import ConversationService
            conv_service = ConversationService(user_id)
            
            original_question = question
            context_question = question
            
            # If conversation_id exists, get context
            if conversation_id:
                context = await conv_service.get_conversation_context(conversation_id)
                if context:
                    # Always use the CURRENT question, not the original question from context
                    # The original_question from context is just for reference
                    original_question_from_context = context.get("original_question", question)
                    
                    # If this is a date range response, incorporate it into the CURRENT question
                    if date_range and context.get("pending_date_range"):
                        start = date_range.get("start")
                        end = date_range.get("end")
                        context_question = f"""{question}

User has provided date range:
- Start date: {start or 'not specified'}
- End date: {end or 'not specified'}

Please use this date range to filter the data and answer the question."""
                        # Update context
                        await conv_service.set_date_range(conversation_id, start, end)
                        await conv_service.add_message(conversation_id, "user", f"Date range: {start} to {end}")
                    else:
                        # Use the CURRENT question, not the original from context
                        context_question = question
                    
                    # Add current user question to conversation
                    await conv_service.add_message(conversation_id, "user", question)
            else:
                # Create new conversation
                conversation_id = await conv_service.create_conversation(question, file_id)
        except ImportError:
            logger.warning("ConversationService not available - conversation context disabled")
            conv_service = None
            context_question = question
        
        # Get LLM instance
        llm = get_llm_instance(provider, temperature=0.0)
        
        # Enhance question with file_id context if provided
        enhanced_question = context_question
        if file_id:
            enhanced_question = f"""Question: {context_question}

IMPORTANT: Use this EXACT file_id in all tool calls: {file_id}
Do NOT use any other file_id or numbers from the question text.
Default table_name is "Sheet1" unless specified otherwise."""
        else:
            # If no file_id provided, instruct agent to discover files first
            enhanced_question = f"""Question: {context_question}

IMPORTANT: You don't have a specific file_id. You MUST:
1. First call list_user_files to see what files are available
2. Then use the file_id from the list_user_files result
3. Do NOT invent or guess file_ids - always use real file_ids from list_user_files
4. Default table_name is "Sheet1" unless specified otherwise"""
        
        # Create agent executor
        executor = create_agent_executor(llm, user_id, file_id, max_iterations)
        
        # Execute query with timeout and iteration limit handling
        try:
            result = executor.invoke({"input": enhanced_question})
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Agent execution error: {str(e)}")
            
            # Check if it's an iteration limit error
            if "iteration" in error_msg or "max_iterations" in error_msg or "maximum iterations" in error_msg:
                logger.warning(f"Agent stopped due to iteration limit ({max_iterations}). Question: {question[:100]}")
                return {
                    "request_id": request_id,
                    "success": False,
                    "answer_short": f"I need more iterations to complete this query. Please try breaking it into smaller questions or increase max_iterations parameter (current: {max_iterations}, max: 50).",
                    "answer_detailed": f"The query exceeded the maximum iteration limit of {max_iterations}. This usually happens with complex queries that require many tool calls. Suggestions: 1) Break the question into smaller parts, 2) Increase max_iterations parameter (up to 50), 3) Make the question more specific.",
                    "error": "iteration_limit_exceeded",
                    "tools_called": [],
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "timestamp": datetime.utcnow(),
                    "confidence": 0.0,
                    "requires_date_range": requires_date_range,
                    "date_range_info": date_range_info,
                    "conversation_id": conversation_id
                }
            # Check if it's a time limit error
            elif "time" in error_msg or "timeout" in error_msg or "execution time" in error_msg:
                logger.warning(f"Agent stopped due to time limit. Question: {question[:100]}")
                return {
                    "request_id": request_id,
                    "success": False,
                    "answer_short": f"Query took too long to execute (exceeded 5 minute limit). Please try a more specific question or break it into smaller parts.",
                    "answer_detailed": "The query exceeded the maximum execution time of 5 minutes. This usually happens with very complex queries or when processing large datasets. Suggestions: 1) Make the question more specific, 2) Filter data by date range or other criteria, 3) Break the question into smaller parts.",
                    "error": "time_limit_exceeded",
                    "tools_called": [],
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "timestamp": datetime.utcnow(),
                    "confidence": 0.0,
                    "requires_date_range": requires_date_range,
                    "date_range_info": date_range_info,
                    "conversation_id": conversation_id
                }
            # Other errors
            else:
                raise
        
        # Extract information
        answer = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Track iteration count and warn if approaching limit
        iteration_count = len(intermediate_steps)
        if iteration_count >= max_iterations * 0.8:  # 80% of limit
            logger.warning(f"Agent used {iteration_count}/{max_iterations} iterations ({iteration_count/max_iterations*100:.1f}%) for question: {question[:100]}")
        elif iteration_count >= max_iterations * 0.6:  # 60% of limit
            logger.info(f"Agent used {iteration_count}/{max_iterations} iterations ({iteration_count/max_iterations*100:.1f}%) for question: {question[:100]}")
        
        # Extract tools called
        tools_called = []
        tool_outputs = {}
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action = step[0]
                observation = step[1]
                
                tool_name = action.tool if hasattr(action, 'tool') else "unknown"
                tools_called.append(tool_name)
                
                # Parse tool output
                try:
                    if isinstance(observation, str):
                        tool_outputs[tool_name] = json.loads(observation)
                    else:
                        tool_outputs[tool_name] = observation
                except:
                    tool_outputs[tool_name] = str(observation)
        
        # Extract values and provenance from tool outputs
        values = {}
        provenance = {}
        chart_config = None
        
        for tool_name, output in tool_outputs.items():
            if isinstance(output, dict) and output.get("ok"):
                if output.get("result"):
                    if isinstance(output["result"], dict):
                        values.update(output["result"])
                if output.get("provenance"):
                    provenance[tool_name] = output["provenance"]
        
        # Generate chart_config for time series queries
        # Time series queries typically need charts, but check if question explicitly asks for just a number
        if "timeseries_analyzer" in tools_called:
            ts_output = tool_outputs.get("timeseries_analyzer", {})
            if isinstance(ts_output, dict) and ts_output.get("ok"):
                ts_result = ts_output.get("result", {})
                series = ts_result.get("series", [])
                if series:
                    question_lower = question.lower()
                    # Check if question explicitly asks for just a number (not a chart)
                    number_only_keywords = ["what is", "what's", "how much", "total", "sum", "average", "mean", "median"]
                    asks_for_number_only = any(keyword in question_lower for keyword in number_only_keywords) and \
                                         not any(keyword in question_lower for keyword in ["chart", "graph", "plot", "show", "visualize", "display", "trend", "over time"])
                    
                    # Only skip chart if question explicitly asks for just a number AND doesn't mention trends/visualization
                    if not asks_for_number_only:
                        # Determine chart type from question
                        chart_type = "line"
                        if "bar" in question_lower or "column" in question_lower:
                            chart_type = "bar"
                        elif "area" in question_lower:
                            chart_type = "line"  # Area charts can be rendered as filled line
                        
                        # Extract labels and data
                        labels = [s.get("period", "") for s in series]
                        data_values = [float(s.get("value", 0)) if s.get("value") is not None else 0 for s in series]
                        
                        # Determine metric name from question or tool output
                        metric_name = "Value"
                        if "production" in question_lower:
                            metric_name = "Production"
                        elif "sales" in question_lower:
                            metric_name = "Sales"
                        elif "quantity" in question_lower:
                            metric_name = "Quantity"
                        
                        chart_config = {
                            "success": True,
                            "chart_type": chart_type,
                            "title": question if len(question) < 60 else question[:57] + "...",
                            "data": {
                                "labels": labels,
                                "datasets": [{
                                    "label": metric_name,
                                    "data": data_values,
                                    "borderColor": "#3B82F6",
                                    "backgroundColor": "rgba(59, 130, 246, 0.1)" if chart_type == "line" else "#3B82F6",
                                    "borderWidth": 2,
                                    "fill": chart_type == "line",
                                    "tension": 0.4,
                                    "pointRadius": 3,
                                    "pointHoverRadius": 5
                                }]
                            },
                            "options": {
                                "responsive": True,
                                "maintainAspectRatio": False,
                                "plugins": {
                                    "title": {
                                        "display": True,
                                        "text": question if len(question) < 60 else question[:57] + "...",
                                        "font": {"size": 16}
                                    },
                                    "legend": {
                                        "display": True,
                                        "position": "top"
                                    }
                                },
                                "scales": {
                                    "x": {
                                        "title": {
                                            "display": True,
                                            "text": "Time Period"
                                        }
                                    },
                                    "y": {
                                        "beginAtZero": True,
                                        "title": {
                                            "display": True,
                                            "text": metric_name
                                        }
                                    }
                                }
                            }
                        }
        
        # Check if we need to ask for date range
        # Check timeseries_analyzer output for large datasets
        if "timeseries_analyzer" in tools_called:
            ts_output = tool_outputs.get("timeseries_analyzer", {})
            if isinstance(ts_output, dict) and ts_output.get("ok"):
                ts_result = ts_output.get("result", {})
                series = ts_result.get("series", [])
                matched_count = ts_output.get("provenance", {}).get("matched_row_count", 0)
                
                # If series has more than 100 points or matched_count > 1000, ask for date range
                if len(series) > 100 or matched_count > 1000:
                    # Try to get date range info
                    # Extract time_col and file_id from tool calls
                    time_col = None
                    for step in intermediate_steps:
                        if len(step) >= 2:
                            action = step[0]
                            if hasattr(action, 'tool') and action.tool == "timeseries_analyzer":
                                if hasattr(action, 'tool_input'):
                                    parts = action.tool_input.split("|")
                                    if len(parts) > 2:
                                        time_col = parts[2]
                                        break
                    
                    if time_col and file_id and conv_service:
                        # Get date range
                        from tools.mongodb_tools import get_date_range
                        date_range_result = get_date_range(user_id, file_id, "Sheet1", time_col)
                        if date_range_result.get("ok"):
                            date_info = date_range_result.get("result", {})
                            requires_date_range = True
                            date_range_info = {
                                "min_date": date_info.get("min_date"),
                                "max_date": date_info.get("max_date"),
                                "row_count": date_info.get("row_count"),
                                "span_days": date_info.get("span_days"),
                                "time_column": time_col
                            }
                            
                            # Update conversation context
                            await conv_service.mark_date_range_pending(conversation_id, date_range_info)
                            
                            # Modify answer to ask for date range
                            answer = f"""The dataset contains {date_info.get('row_count', 0)} rows spanning from {date_info.get('min_date', 'unknown')} to {date_info.get('max_date', 'unknown')} ({date_info.get('span_days', 0)} days).

To provide an accurate analysis, please specify a date range:
- You can say "last 30 days", "last month", "Q1 2025", or specific dates like "2025-11-01 to 2025-12-31"
- What date range would you like me to analyze?"""
                            
                            # Add assistant message to conversation
                            await conv_service.add_message(
                                conversation_id,
                                "assistant",
                                answer,
                                {"requires_date_range": True, "date_range_info": date_range_info}
                            )
        
        # Generate chart_config for ranking queries (bar chart)
        # Only generate chart if question explicitly asks for visualization
        elif "rank_entities" in tools_called:
            rank_output = tool_outputs.get("rank_entities", {})
            if isinstance(rank_output, dict) and rank_output.get("ok"):
                rank_result = rank_output.get("result", {})
                entities = rank_result.get("entities", [])
                if entities:
                    # Check if question asks for a chart/visualization
                    question_lower = question.lower()
                    chart_keywords = [
                        "chart", "graph", "plot", "visualize", "show", "display",
                        "bar chart", "line chart", "pie chart", "scatter", "visualization",
                        "as a chart", "as a graph", "as a bar", "as a line"
                    ]
                    # Also check for comparative questions that typically need charts
                    comparative_keywords = ["which", "who", "top", "bottom", "compare", "comparison"]
                    has_chart_keyword = any(keyword in question_lower for keyword in chart_keywords)
                    has_comparative_keyword = any(keyword in question_lower for keyword in comparative_keywords)
                    
                    # Only generate chart if:
                    # 1. Question explicitly mentions chart/graph/visualization, OR
                    # 2. Question is comparative (which/who/top/bottom) AND has multiple entities (more than 1)
                    should_generate_chart = has_chart_keyword or (has_comparative_keyword and len(entities) > 1)
                    
                    if should_generate_chart:
                        labels = [e.get("entity", "") for e in entities]
                        data_values = [float(e.get("value", 0)) if e.get("value") is not None else 0 for e in entities]
                        metric_name = rank_result.get("metric", "Value")
                        
                        # Determine chart type from question
                        chart_type = "bar"  # Default to bar chart
                        if "pie" in question_lower or "pie chart" in question_lower:
                            chart_type = "pie"
                        elif "line" in question_lower or "line chart" in question_lower:
                            chart_type = "line"
                        elif "scatter" in question_lower or "scatter chart" in question_lower:
                            chart_type = "scatter"
                        elif "bar" in question_lower or "column" in question_lower or "bar chart" in question_lower:
                            chart_type = "bar"
                        
                        # For pie charts, use different color scheme
                        if chart_type == "pie":
                            # Generate distinct colors for pie chart
                            num_colors = len(data_values)
                            background_colors = []
                            border_colors = []
                            for i in range(num_colors):
                                hue = i / num_colors
                                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                                bg_color = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)"
                                border_color = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 1)"
                                background_colors.append(bg_color)
                                border_colors.append(border_color)
                        else:
                            # For bar/line charts, use single color
                            background_colors = "#3B82F6"
                            border_colors = "#2563EB"
                        
                        chart_config = {
                            "success": True,
                            "chart_type": chart_type,
                            "title": question if len(question) < 60 else question[:57] + "...",
                            "data": {
                                "labels": labels,
                                "datasets": [{
                                    "label": metric_name,
                                    "data": data_values,
                                    "backgroundColor": background_colors,
                                    "borderColor": border_colors,
                                    "borderWidth": 1
                                }]
                            },
                            "options": {
                                "responsive": True,
                                "maintainAspectRatio": False,
                                "plugins": {
                                    "title": {
                                        "display": True,
                                        "text": question if len(question) < 60 else question[:57] + "...",
                                        "font": {"size": 16}
                                    },
                                    "legend": {
                                        "display": True,
                                        "position": "top"
                                    }
                                },
                                "scales": {
                                    "x": {
                                        "title": {
                                            "display": True,
                                            "text": "Entity"
                                        }
                                    },
                                    "y": {
                                        "beginAtZero": True,
                                        "title": {
                                            "display": True,
                                            "text": metric_name
                                        }
                                    }
                                }
                            }
                        }
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Add assistant message to conversation if not already added
        if conv_service and not requires_date_range:
            try:
                await conv_service.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    {"chart_config": chart_config is not None}
                )
            except Exception as e:
                logger.warning(f"Failed to add message to conversation: {str(e)}")
        
        return {
            "request_id": request_id,
            "success": True,
            "answer_short": answer,
            "answer_detailed": answer,
            "values": values if values else None,
            "chart_config": chart_config,
            "provenance": provenance if provenance else None,
            "verification": None,  # Will be populated with verification results
            "confidence": 0.95 if tools_called else 0.5,
            "tools_called": tools_called,
            "tool_outputs": tool_outputs,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(),
            "error": None,
            "requires_date_range": requires_date_range,
            "date_range_info": date_range_info,
            "conversation_id": conversation_id
        }
    
    except Exception as e:
        logger.error(f"Error executing agent query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "request_id": request_id,
            "success": False,
            "answer_short": f"Error: {str(e)}",
            "answer_detailed": None,
            "values": None,
            "chart_config": None,
            "provenance": None,
            "verification": None,
            "confidence": 0.0,
            "tools_called": [],
            "tool_outputs": None,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(),
            "error": str(e)
        }

