"""
MongoDB-based Deterministic Tools for Agent System
All tools return canonical envelope format with provenance
"""

import time
import json
import ast
import operator as op
from decimal import Decimal, getcontext
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os
import re
import calendar
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

# High precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)

# Allowed operations for calc_eval
ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv
}


def now_ms() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def get_sync_mongo_client():
    """Get synchronous MongoDB client for tools (pymongo)"""
    from pymongo import MongoClient
    import os
    
    uri = os.getenv("MONGODB_URI")
    if uri:
        client = MongoClient(uri)
        # Extract database name from URI
        uri_parts = uri.split("?")[0]
        if "/" in uri_parts:
            db_name_from_uri = uri_parts.split("/")[-1]
            if db_name_from_uri and db_name_from_uri not in ["", "mongodb+srv:", "mongodb:"]:
                db_name = db_name_from_uri
            else:
                db_name = os.getenv("MONGODB_DB_NAME", "excelllm")
        else:
            db_name = os.getenv("MONGODB_DB_NAME", "excelllm")
        return client[db_name]
    
    host = os.getenv("MONGODB_HOST", "localhost")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    db_name = os.getenv("MONGODB_DB_NAME", "excelllm")
    
    return MongoClient(f"mongodb://{host}:{port}")[db_name]


def build_multi_search_query(
    user_id: str,
    file_id: Optional[str] = None,
    table_name: Optional[str] = None,
    file_name_pattern: Optional[str] = None,
    filters: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Build MongoDB query that supports:
    - Single file/sheet (current behavior)
    - All files (*)
    - All sheets (*)
    - File name pattern matching
    """
    from bson import ObjectId
    try:
        user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
    except:
        user_id_obj = user_id
    
    query = {"user_id": user_id_obj}
    
    # Handle file_id
    if file_id and file_id != "*" and file_id != "all":
        query["file_id"] = file_id
    elif file_name_pattern:
        # Need to lookup file_ids by filename pattern
        db = get_sync_mongo_client()
        files_coll = db['files']
        matching_files = list(files_coll.find(
            {
                "user_id": user_id_obj,
                "original_filename": {"$regex": file_name_pattern, "$options": "i"}
            },
            {"file_id": 1}
        ))
        if matching_files:
            query["file_id"] = {"$in": [f["file_id"] for f in matching_files]}
        else:
            return None  # No matching files
    
    # Handle table_name
    if table_name and table_name != "*" and table_name != "all":
        query["table_name"] = table_name
    
    if filters:
        query.update(filters)
    
    return query


def _merge_schemas(schemas_by_source: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Merge schemas from multiple sources into unified schema"""
    unified = {}
    for source, schema in schemas_by_source.items():
        for col_name, col_type in schema.items():
            if col_name not in unified:
                unified[col_name] = col_type
            # If types differ, use most common or "mixed"
            elif unified[col_name] != col_type:
                unified[col_name] = "mixed"
    return unified


def _build_metric_expr(metric: Dict[str, str]) -> Dict[str, Any]:
    """Build MongoDB aggregation expression for a metric"""
    op_type = metric.get("op", "").lower()
    col = metric.get("col", "")
    alias = metric.get("alias", col)
    
    col_path = f"$row.{col}"
    
    if op_type == "sum":
        return {"$sum": col_path}
    elif op_type == "avg" or op_type == "average" or op_type == "mean":
        return {"$avg": col_path}
    elif op_type == "count":
        return {"$sum": 1}
    elif op_type == "min":
        return {"$min": col_path}
    elif op_type == "max":
        return {"$max": col_path}
    elif op_type == "median":
        # Median requires $percentile aggregation (MongoDB 7.0+)
        # Fallback to approximate median
        return {"$avg": col_path}
    else:
        return {"$sum": col_path}  # Default to sum


def get_date_range(
    user_id: str,
    file_id: str,
    table_name: str,
    time_col: str
) -> Dict[str, Any]:
    """
    Get date range information for a time column.
    Used to detect if data is too large and needs date filtering.
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID
        table_name: Table/sheet name
        time_col: Time column name
    
    Returns:
        Canonical tool envelope with min/max dates and row count
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        
        # Convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Build match stage
        match_stage = {
            "$match": {
                "user_id": user_id_obj,
                "file_id": file_id,
                "table_name": table_name,
                f"row.{time_col}": {"$exists": True, "$ne": None}
            }
        }
        
        # Convert string dates to date objects
        date_conversion = {
            "$cond": {
                "if": {"$eq": [{"$type": f"$row.{time_col}"}, "string"]},
                "then": {
                    "$dateFromString": {
                        "dateString": f"$row.{time_col}",
                        "onError": None,
                        "onNull": None
                    }
                },
                "else": f"$row.{time_col}"
            }
        }
        
        # Get min/max dates and count
        pipeline = [
            match_stage,
            {"$addFields": {"_date_converted": date_conversion}},
            {"$match": {"_date_converted": {"$ne": None}}},
            {
                "$group": {
                    "_id": None,
                    "min_date": {"$min": "$_date_converted"},
                    "max_date": {"$max": "$_date_converted"},
                    "count": {"$sum": 1}
                }
            }
        ]
        
        res = list(coll.aggregate(pipeline))
        
        if res and res[0]:
            result_doc = res[0]
            min_date = result_doc.get("min_date")
            max_date = result_doc.get("max_date")
            count = result_doc.get("count", 0)
            
            # Calculate date range span in days
            span_days = None
            if min_date and max_date:
                if isinstance(min_date, datetime) and isinstance(max_date, datetime):
                    span_days = (max_date - min_date).days
                elif isinstance(min_date, str) and isinstance(max_date, str):
                    try:
                        from datetime import datetime as dt
                        min_dt = dt.fromisoformat(min_date.replace('Z', '+00:00'))
                        max_dt = dt.fromisoformat(max_date.replace('Z', '+00:00'))
                        span_days = (max_dt - min_dt).days
                    except:
                        pass
            
            return {
                "ok": True,
                "tool": "get_date_range",
                "result": {
                    "min_date": min_date.isoformat() if isinstance(min_date, datetime) else str(min_date) if min_date else None,
                    "max_date": max_date.isoformat() if isinstance(max_date, datetime) else str(max_date) if max_date else None,
                    "row_count": count,
                    "span_days": span_days
                },
                "unit": None,
                "provenance": {
                    "mongo_pipeline": pipeline,
                    "time_column": time_col
                },
                "meta": {"time_ms": now_ms() - start_ms}
            }
        else:
            return {
                "ok": False,
                "tool": "get_date_range",
                "error": "no_data_found",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
    except Exception as e:
        logger.error(f"Error in get_date_range: {str(e)}")
        return {
            "ok": False,
            "tool": "get_date_range",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def table_loader(
    user_id: str,
    file_id: Optional[str] = None,  # Can be None, "*", "all", or specific file_id
    table_name: Optional[str] = None,  # Can be None, "*", "all", or specific sheet
    file_name_pattern: Optional[str] = None,  # Search by filename
    filters: Optional[Dict] = None,
    fields: Optional[List[str]] = None,
    limit: int = 100,
    include_source: bool = True  # Include file_id, file_name, table_name in results
) -> Dict[str, Any]:
    """
    Enhanced table_loader that can search:
    - All files: file_id="*" or None
    - All sheets: table_name="*" or None
    - By filename: file_name_pattern="July 25"
    - Single file/sheet: file_id="xxx", table_name="Sheet1"
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID (can be "*" for all files)
        table_name: Table/sheet name (can be "*" for all sheets)
        file_name_pattern: Search files by name pattern
        filters: Optional MongoDB filter dict
        fields: Optional list of fields to project
        limit: Maximum rows to return
        include_source: Include source information in results
    
    Returns:
        Canonical tool envelope with schema, sample rows, and provenance
    """
    start_ms = now_ms()
    
    try:
        db = get_sync_mongo_client()
        coll = db['tables']
        files_coll = db['files']
        
        # Build query with multi-search support
        query = build_multi_search_query(user_id, file_id, table_name, file_name_pattern, filters)
        if query is None:
            return {
                "ok": False,
                "tool": "table_loader",
                "error": "no_matching_files",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Projection
        projection = None
        if fields:
            projection = {f: 1 for f in fields}
            projection["_id"] = 0
        
        # Fetch rows with source information
        cursor = coll.find(query, projection=projection).limit(limit)
        rows = list(cursor)
        
        if not rows:
            return {
                "ok": False,
                "tool": "table_loader",
                "error": "no_rows",
                "result": None,
                "unit": None,
                "provenance": {"mongo_query": query, "matched_row_count": 0},
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Get file metadata for source tracking
        # First try to get file_name from rows (if normalized rows include it)
        file_ids_in_results = set(r.get("file_id") for r in rows)
        file_metadata = {}
        
        # Check if rows already have file_name (from normalized rows)
        rows_with_file_name = [r for r in rows if r.get("file_name")]
        if rows_with_file_name:
            # Use file_name directly from rows (more efficient)
            for row in rows:
                fid = row.get("file_id")
                if fid and fid not in file_metadata:
                    file_metadata[fid] = row.get("file_name", "Unknown")
        else:
            # Fallback: lookup from files collection
            for fid in file_ids_in_results:
                file_doc = files_coll.find_one({"file_id": fid}, {"original_filename": 1, "file_id": 1})
                if file_doc:
                    file_metadata[fid] = file_doc.get("original_filename", "Unknown")
        
        # Build schema with source context
        schemas_by_source = {}
        sample_rows_by_source = {}
        
        for row in rows:
            # Use file_name from row if available, otherwise lookup
            file_name = row.get("file_name") or file_metadata.get(row.get("file_id"), "Unknown")
            source_key = f"{file_name}::{row.get('table_name', 'Unknown')}"
            
            if source_key not in schemas_by_source:
                schemas_by_source[source_key] = {}
                sample_rows_by_source[source_key] = []
            
            # Extract schema from row
            row_data = row.get("row", {})
            for key, value in row_data.items():
                if key not in schemas_by_source[source_key]:
                    schemas_by_source[source_key][key] = type(value).__name__
            
            if len(sample_rows_by_source[source_key]) < 5:
                sample_rows_by_source[source_key].append({
                    "file_id": row.get("file_id"),
                    "file_name": file_name,
                    "table_name": row.get("table_name"),
                    "row_id": row.get("row_id"),
                    "data": row_data
                })
        
        matched_count = coll.count_documents(query)
        
        # For backward compatibility, also include single schema if only one source
        unified_schema = _merge_schemas(schemas_by_source)
        single_schema = list(schemas_by_source.values())[0] if len(schemas_by_source) == 1 else unified_schema
        
        result = {
            "ok": True,
            "tool": "table_loader",
            "result": {
                "schema": single_schema,  # Backward compatible
                "schemas_by_source": schemas_by_source,  # Grouped by file::sheet
                "unified_schema": unified_schema,  # All columns across all sources
                "sample_rows": [r["data"] for r in list(sample_rows_by_source.values())[0][:5]] if sample_rows_by_source else [],  # Backward compatible
                "sample_rows_by_source": sample_rows_by_source,
                "sources": [
                    {
                        "file_id": fid,
                        "file_name": file_metadata.get(fid, "Unknown"),
                        "table_names": list(set(r.get("table_name") for r in rows if r.get("file_id") == fid))
                    }
                    for fid in file_ids_in_results
                ],
                "row_count": matched_count,
                "total_sources": len(file_ids_in_results)
            },
            "unit": None,
            "provenance": {
                "mongo_query": query,
                "matched_row_count": matched_count,
                "sources_searched": len(file_ids_in_results)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in table_loader: {str(e)}")
        return {
            "ok": False,
            "tool": "table_loader",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def agg_helper(
    user_id: str,
    file_id: Optional[str] = None,  # "*" for all files
    table_name: Optional[str] = None,  # "*" for all sheets
    file_name_pattern: Optional[str] = None,
    filters: Optional[Dict] = None,
    metrics: List[Dict[str, str]] = None,
    date_filter: Optional[Dict[str, Any]] = None,  # {"column": "date_col", "start": "2025-01-01", "end": "2025-12-31", "auto_detect": True}
    auto_detect_date_column: bool = True,
    group_by_source: bool = False  # If True, aggregate separately per file/sheet
) -> Dict[str, Any]:
    """
    Enhanced aggregation that can aggregate:
    - Across all files and sheets (unified result)
    - Per file/sheet (group_by_source=True)
    - With automatic date filtering
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID (can be "*" for all files)
        table_name: Table/sheet name (can be "*" for all sheets)
        file_name_pattern: Search files by name pattern
        filters: Optional MongoDB filter dict
        metrics: List of metric operations
        date_filter: Optional date filter dict
        auto_detect_date_column: Auto-detect date columns if date_filter provided
        group_by_source: If True, aggregate separately per file/sheet
    
    Returns:
        Canonical tool envelope with aggregated results and provenance
    """
    start_ms = now_ms()
    
    try:
        db = get_sync_mongo_client()
        coll = db['tables']
        files_coll = db['files']
        
        if not metrics:
            return {
                "ok": False,
                "tool": "agg_helper",
                "error": "no_metrics_provided",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Build base match stage with multi-search support
        query = build_multi_search_query(user_id, file_id, table_name, file_name_pattern, filters)
        if query is None:
            return {
                "ok": False,
                "tool": "agg_helper",
                "error": "no_matching_files",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        match_stage = {"$match": query}
        
        # Handle date filter
        date_column = None
        if date_filter:
            date_column = date_filter.get("column")
            
            # Auto-detect date column if needed
            if not date_column and date_filter.get("auto_detect", auto_detect_date_column):
                # Import here to avoid circular dependency
                detect_result = detect_date_columns(user_id, file_id, table_name, file_name_pattern)
                if detect_result.get("ok"):
                    date_cols_by_source = detect_result.get("result", {}).get("date_columns_by_source", {})
                    for source, cols in date_cols_by_source.items():
                        if cols:
                            date_column = list(cols.keys())[0]
                            break
            
            if date_column:
                date_start = date_filter.get("start")
                date_end = date_filter.get("end")
                
                # Parse relative dates if needed
                if isinstance(date_start, str) and not re.match(r'^\d{4}-\d{2}-\d{2}', date_start):
                    parsed = parse_relative_date(date_start)
                    if parsed.get("ok"):
                        date_start = parsed["result"]["start_date"]
                
                if isinstance(date_end, str) and not re.match(r'^\d{4}-\d{2}-\d{2}', date_end):
                    parsed = parse_relative_date(date_end)
                    if parsed.get("ok"):
                        date_end = parsed["result"]["end_date"]
                
                # Add date filter to match stage
                if date_start or date_end:
                    date_query = {}
                    if date_start:
                        try:
                            date_query["$gte"] = datetime.fromisoformat(date_start.replace('Z', '+00:00'))
                        except:
                            date_query["$gte"] = date_start
                    if date_end:
                        try:
                            date_query["$lte"] = datetime.fromisoformat(date_end.replace('Z', '+00:00'))
                        except:
                            date_query["$lte"] = date_end
                    
                    match_stage["$match"][f"row.{date_column}"] = date_query
        
        # Handle filters (list format)
        if filters and isinstance(filters, list):
            for filter_obj in filters:
                if isinstance(filter_obj, dict):
                    col = filter_obj.get("col", "")
                    op = filter_obj.get("op", "").lower()
                    value = filter_obj.get("value")
                    
                    if col and value is not None:
                        col_path = f"row.{col}"
                        if op == "starts_with" or op == "startswith":
                            match_stage["$match"][col_path] = {"$regex": f"^{value}", "$options": "i"}
                        elif op == "ends_with" or op == "endswith":
                            match_stage["$match"][col_path] = {"$regex": f"{value}$", "$options": "i"}
                        elif op == "contains":
                            match_stage["$match"][col_path] = {"$regex": value, "$options": "i"}
                        elif op == "eq" or op == "equals" or op == "==":
                            match_stage["$match"][col_path] = value
                        elif op == "ne" or op == "not_equals" or op == "!=":
                            match_stage["$match"][col_path] = {"$ne": value}
                        elif op == "gt" or op == "greater_than" or op == ">":
                            match_stage["$match"][col_path] = {"$gt": value}
                        elif op == "gte" or op == "greater_than_equal" or op == ">=":
                            match_stage["$match"][col_path] = {"$gte": value}
                        elif op == "lt" or op == "less_than" or op == "<":
                            match_stage["$match"][col_path] = {"$lt": value}
                        elif op == "lte" or op == "less_than_equal" or op == "<=":
                            match_stage["$match"][col_path] = {"$lte": value}
                        elif op == "in":
                            match_stage["$match"][col_path] = {"$in": value if isinstance(value, list) else [value]}
                        else:
                            match_stage["$match"][col_path] = value
        
        # Build aggregation pipeline
        pipeline = [match_stage]
        
        if group_by_source:
            # Group by file_id and table_name to get per-source results
            group_stage = {
                "$group": {
                    "_id": {
                        "file_id": "$file_id",
                        "table_name": "$table_name"
                    },
                    **{metric["alias"]: _build_metric_expr(metric) for metric in metrics}
                }
            }
            pipeline.append(group_stage)
            
            # Lookup file names
            lookup_stage = {
                "$lookup": {
                    "from": "files",
                    "let": {"file_id": "$_id.file_id"},
                    "pipeline": [
                        {"$match": {"$expr": {"$eq": ["$file_id", "$$file_id"]}}},
                        {"$project": {"original_filename": 1}}
                    ],
                    "as": "file_info"
                }
            }
            pipeline.append(lookup_stage)
            
            # Format results
            project_stage = {
                "$project": {
                    "file_id": "$_id.file_id",
                    "file_name": {"$arrayElemAt": ["$file_info.original_filename", 0]},
                    "table_name": "$_id.table_name",
                    **{metric["alias"]: 1 for metric in metrics}
                }
            }
            pipeline.append(project_stage)
        else:
            # Unified aggregation across all sources
            group_stage = {
                "$group": {
                    "_id": None,
                    **{metric["alias"]: _build_metric_expr(metric) for metric in metrics}
                }
            }
            pipeline.append(group_stage)
            
            # Remove _id from result
            project_stage = {
                "$project": {
                    "_id": 0,
                    **{metric["alias"]: 1 for metric in metrics}
                }
            }
            pipeline.append(project_stage)
        
        # Execute pipeline
        results = list(coll.aggregate(pipeline))
        
        # Convert numeric fields to Decimal for accuracy
        for result_item in results:
            for key in list(result_item.keys()):
                if key == "_id":
                    continue
                value = result_item[key]
                if isinstance(value, (int, float, np.number)):
                    result_item[key] = Decimal(str(value))
                elif isinstance(value, list) and len(value) == 1:
                    result_item[key] = Decimal(str(value[0]))
        
        matched_count = coll.count_documents(match_stage["$match"])
        
        return {
            "ok": True,
            "tool": "agg_helper",
            "result": {
                "aggregated": results[0] if not group_by_source and results else results,
                "grouped_by_source": group_by_source
            },
            "unit": None,
            "provenance": {
                "mongo_pipeline": pipeline,
                "matched_row_count": matched_count,
                "sources_searched": len(set(query.get("file_id", []))) if isinstance(query.get("file_id"), list) else 1,
                "date_column_used": date_column
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in agg_helper: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "ok": False,
            "tool": "agg_helper",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def timeseries_analyzer(
    user_id: str,
    file_id: str,
    table_name: str,
    time_col: str,
    metric_col: str,
    freq: str = "month",
    agg: str = "sum",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Compute time-bucketed series using MongoDB dateTrunc and return slope/trend.
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID
        table_name: Table/sheet name
        time_col: Time column name
        metric_col: Metric column to aggregate
        freq: Frequency ("day", "week", "month", "year")
        agg: Aggregation operation ("sum", "avg", "count", "min", "max")
        start: Optional start date
        end: Optional end date
    
    Returns:
        Canonical tool envelope with time series and trend slope
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        
        # Convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Build match stage
        match_stage = {
            "$match": {
                "user_id": user_id_obj,
                "file_id": file_id,
                "table_name": table_name,
                f"row.{time_col}": {"$exists": True, "$ne": None},
                f"row.{metric_col}": {"$exists": True, "$ne": None}
            }
        }
        
        if start or end:
            time_filter = {}
            if start:
                time_filter["$gte"] = start
            if end:
                time_filter["$lte"] = end
            match_stage["$match"][f"row.{time_col}"] = time_filter
        
        # Convert string dates to date objects first, then truncate
        # Handle both string dates (like "2025-11-02") and date objects
        # Use $dateFromString to convert ISO date strings to date objects
        date_conversion = {
            "$cond": {
                "if": {"$eq": [{"$type": f"$row.{time_col}"}, "string"]},
                "then": {
                    "$dateFromString": {
                        "dateString": f"$row.{time_col}",
                        "onError": None,
                        "onNull": None
                    }
                },
                "else": f"$row.{time_col}"
            }
        }
        
        # Build date truncation - use $dateTrunc if date conversion succeeds, else use $dateToString
        date_bucket_trunc = {
            "$dateTrunc": {
                "date": "$_date_converted",
                "unit": freq
            }
        }
        
        # Fallback format for string-based grouping if date conversion fails
        date_format_map = {
            "day": "%Y-%m-%d",
            "week": "%Y-W%V",
            "month": "%Y-%m",
            "year": "%Y"
        }
        date_format = date_format_map.get(freq, "%Y-%m-%d")
        
        date_bucket = {
            "$cond": {
                "if": {"$ne": ["$_date_converted", None]},
                "then": date_bucket_trunc,
                "else": {
                    "$dateToString": {
                        "format": date_format,
                        "date": {
                            "$dateFromString": {
                                "dateString": f"$row.{time_col}",
                                "onError": None
                            }
                        }
                    }
                }
            }
        }
        
        # Build aggregation pipeline
        pipeline = [
            match_stage,
            {"$addFields": {"_date_converted": date_conversion}},
            {"$match": {"_date_converted": {"$ne": None}}},
            {"$addFields": {"_period": date_bucket}},
            {
                "$group": {
                    "_id": "$_period",
                    "value": {f"${agg}": f"$row.{metric_col}"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        res = list(coll.aggregate(pipeline))
        
        # Convert to series format
        series = []
        for r in res:
            period = r["_id"]
            value = r.get("value")
            series.append({
                "period": period.isoformat() if isinstance(period, datetime) else str(period),
                "value": Decimal(str(value)) if value is not None else None
            })
        
        # Compute slope deterministically using numpy
        slope = Decimal("0")
        if len(series) >= 2:
            xs = np.arange(len(series))
            ys = np.array([
                float(s["value"]) if s["value"] is not None else 0.0
                for s in series
            ])
            if len(ys) > 0 and np.any(ys != 0):
                slope_coeffs = np.polyfit(xs, ys, 1)
                slope = Decimal(str(slope_coeffs[0]))
        
        matched_count = coll.count_documents(match_stage["$match"])
        
        return {
            "ok": True,
            "tool": "timeseries_analyzer",
            "result": {
                "series": series,
                "slope": slope,
                "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            },
            "unit": None,
            "provenance": {
                "mongo_pipeline": pipeline,
                "matched_row_count": matched_count
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in timeseries_analyzer: {str(e)}")
        return {
            "ok": False,
            "tool": "timeseries_analyzer",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def compare_entities(
    user_id: str,
    file_id: str,
    table_name: str,
    key_col: str,
    metric_col: str,
    entity_a: str,
    entity_b: str,
    agg: str = "sum",
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compare two entities side-by-side with percent difference.
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID
        table_name: Table/sheet name
        key_col: Entity identifier column
        metric_col: Metric column to aggregate
        entity_a: First entity value
        entity_b: Second entity value
        agg: Aggregation operation ("sum", "avg", "count", "min", "max")
        filters: Optional additional filters
    
    Returns:
        Canonical tool envelope with comparison results
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        
        # Build match stages for each entity - convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        base_match = {
            "user_id": user_id_obj,
            "file_id": file_id,
            "table_name": table_name,
            f"row.{key_col}": {"$exists": True},
            f"row.{metric_col}": {"$exists": True, "$ne": None}
        }
        
        if filters:
            if isinstance(filters, dict):
                base_match.update(filters)
            elif isinstance(filters, list):
                # Convert filter format to MongoDB query format
                for filter_obj in filters:
                    if isinstance(filter_obj, dict):
                        col = filter_obj.get("col", "")
                        op = filter_obj.get("op", "").lower()
                        value = filter_obj.get("value")
                        
                        if col and value is not None:
                            col_path = f"row.{col}"
                            if op == "starts_with" or op == "startswith":
                                base_match[col_path] = {"$regex": f"^{value}", "$options": "i"}
                            elif op == "eq" or op == "equals" or op == "==":
                                base_match[col_path] = value
                            else:
                                base_match[col_path] = value
        
        # Support partial matching for entity values (e.g., "Line-1" matches "Line-1/Machine-M1")
        # Use regex for partial matching
        match_a = {**base_match, f"row.{key_col}": {"$regex": f"^{entity_a}", "$options": "i"}}
        match_b = {**base_match, f"row.{key_col}": {"$regex": f"^{entity_b}", "$options": "i"}}
        
        # Aggregate for entity A
        pipeline_a = [
            {"$match": match_a},
            {
                "$group": {
                    "_id": None,
                    "value": {f"${agg}": f"$row.{metric_col}"}
                }
            }
        ]
        res_a = list(coll.aggregate(pipeline_a))
        value_a = Decimal(str(res_a[0]["value"])) if res_a and res_a[0].get("value") is not None else Decimal("0")
        
        # Aggregate for entity B
        pipeline_b = [
            {"$match": match_b},
            {
                "$group": {
                    "_id": None,
                    "value": {f"${agg}": f"$row.{metric_col}"}
                }
            }
        ]
        res_b = list(coll.aggregate(pipeline_b))
        value_b = Decimal(str(res_b[0]["value"])) if res_b and res_b[0].get("value") is not None else Decimal("0")
        
        # Calculate percent difference
        if value_b != 0:
            percent_diff = ((value_a - value_b) / value_b) * 100
        else:
            percent_diff = Decimal("0") if value_a == 0 else Decimal("inf")
        
        return {
            "ok": True,
            "tool": "compare_entities",
            "result": {
                entity_a: float(value_a),
                entity_b: float(value_b),
                "difference": float(value_a - value_b),
                "percent_difference": float(percent_diff)
            },
            "unit": None,
            "provenance": {
                "entity_a_pipeline": pipeline_a,
                "entity_b_pipeline": pipeline_b,
                "matched_a_count": coll.count_documents(match_a),
                "matched_b_count": coll.count_documents(match_b)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in compare_entities: {str(e)}")
        return {
            "ok": False,
            "tool": "compare_entities",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def calc_eval(expr: str) -> Dict[str, Any]:
    """
    Safe deterministic calculator using Decimal and AST parsing.
    Only allows numbers and basic arithmetic operations.
    
    Args:
        expr: Mathematical expression string
    
    Returns:
        Canonical tool envelope with calculated value
    """
    start_ms = now_ms()
    
    try:
        def _eval(node):
            if isinstance(node, ast.Constant):
                return Decimal(str(node.value))
            if isinstance(node, ast.Num):  # Python < 3.8
                return Decimal(str(node.n))
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type not in ALLOWED_OPS:
                    raise ValueError(f"Unsupported operation: {op_type}")
                return ALLOWED_OPS[op_type](left, right)
            if isinstance(node, ast.UnaryOp):
                return ALLOWED_OPS[type(node.op)](_eval(node.operand))
            raise ValueError(f"Unsupported node type: {type(node)}")
        
        # Parse and evaluate
        tree = ast.parse(expr, mode='eval')
        value = _eval(tree.body)
        
        return {
            "ok": True,
            "tool": "calc_eval",
            "result": {"value": float(value), "decimal_value": str(value)},
            "unit": None,
            "provenance": {"expression": expr},
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in calc_eval: {str(e)}")
        return {
            "ok": False,
            "tool": "calc_eval",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def statistical_summary(
    user_id: str,
    file_id: str,
    table_name: str,
    columns: List[str],
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compute statistical summary (min/max/mean/median/std) for numeric columns.
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID
        table_name: Table/sheet name
        columns: List of column names to summarize
        filters: Optional MongoDB filter dict
    
    Returns:
        Canonical tool envelope with statistical summaries
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        
        # Convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        match_stage = {
            "$match": {
                "user_id": user_id_obj,
                "file_id": file_id,
                "table_name": table_name
            }
        }
        
        if filters:
            match_stage["$match"].update(filters)
        
        # Build group stage with all statistics
        group_stage = {"$group": {"_id": None}}
        
        for col in columns:
            col_path = f"$row.{col}"
            group_stage["$group"][f"{col}_min"] = {"$min": col_path}
            group_stage["$group"][f"{col}_max"] = {"$max": col_path}
            group_stage["$group"][f"{col}_avg"] = {"$avg": col_path}
            group_stage["$group"][f"{col}_std"] = {"$stdDevPop": col_path}
            group_stage["$group"][f"{col}_median"] = {"$percentile": {
                "input": col_path,
                "p": [0.5],
                "method": "approximate"
            }}
        
        pipeline = [match_stage, group_stage]
        res = list(coll.aggregate(pipeline))
        
        result = {}
        if res:
            stats = res[0]
            for col in columns:
                result[col] = {
                    "min": float(stats.get(f"{col}_min", 0)) if stats.get(f"{col}_min") is not None else None,
                    "max": float(stats.get(f"{col}_max", 0)) if stats.get(f"{col}_max") is not None else None,
                    "mean": float(stats.get(f"{col}_avg", 0)) if stats.get(f"{col}_avg") is not None else None,
                    "std": float(stats.get(f"{col}_std", 0)) if stats.get(f"{col}_std") is not None else None,
                    "median": float(stats.get(f"{col}_median", [0])[0]) if isinstance(stats.get(f"{col}_median"), list) else None
                }
        
        matched_count = coll.count_documents(match_stage["$match"])
        
        return {
            "ok": True,
            "tool": "statistical_summary",
            "result": result,
            "unit": None,
            "provenance": {
                "mongo_pipeline": pipeline,
                "matched_row_count": matched_count
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in statistical_summary: {str(e)}")
        return {
            "ok": False,
            "tool": "statistical_summary",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def rank_entities(
    user_id: str,
    file_id: str,
    table_name: str,
    key_col: str,
    metric_col: str,
    agg: str = "sum",
    n: int = 5,
    order: str = "desc",
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Rank entities by aggregated metric (top N or bottom N).
    
    Args:
        user_id: User ID for multi-tenant filtering
        file_id: File ID
        table_name: Table/sheet name
        key_col: Entity identifier column (e.g., "Product", "Material_Name")
        metric_col: Metric column to aggregate (e.g., "Sales", "Consumption_Kg")
        agg: Aggregation operation ("sum", "avg", "count", "min", "max")
        n: Number of top/bottom entities to return
        order: "desc" for top N (highest), "asc" for bottom N (lowest)
        filters: Optional additional filters
    
    Returns:
        Canonical tool envelope with ranked entities and their values
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        
        # Convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Build match stage
        match_stage = {
            "$match": {
                "user_id": user_id_obj,
                "file_id": file_id,
                "table_name": table_name,
                f"row.{key_col}": {"$exists": True, "$ne": None},
                f"row.{metric_col}": {"$exists": True, "$ne": None}
            }
        }
        
        if filters:
            if isinstance(filters, dict):
                match_stage["$match"].update(filters)
            elif isinstance(filters, list):
                # Convert filter format to MongoDB query format
                for filter_obj in filters:
                    if isinstance(filter_obj, dict):
                        col = filter_obj.get("col", "")
                        op = filter_obj.get("op", "").lower()
                        value = filter_obj.get("value")
                        
                        if col and value is not None:
                            col_path = f"row.{col}"
                            if op == "starts_with" or op == "startswith":
                                match_stage["$match"][col_path] = {"$regex": f"^{value}", "$options": "i"}
                            elif op == "eq" or op == "equals" or op == "==":
                                match_stage["$match"][col_path] = value
                            else:
                                match_stage["$match"][col_path] = value
        
        # Build aggregation pipeline
        sort_order = -1 if order.lower() == "desc" else 1
        
        # Handle "count" aggregation specially - count documents, not sum a metric
        if agg.lower() == "count":
            # For count, we don't need metric_col - just count documents per entity
            # Remove metric_col requirement from match stage
            if f"row.{metric_col}" in match_stage["$match"]:
                del match_stage["$match"][f"row.{metric_col}"]
            
            group_stage = {
                "$group": {
                    "_id": f"$row.{key_col}",
                    "value": {"$sum": 1}  # Count documents in each group
                }
            }
        else:
            # For other aggregations (sum, avg, min, max), use the metric column
            group_stage = {
                "$group": {
                    "_id": f"$row.{key_col}",
                    "value": {f"${agg}": f"$row.{metric_col}"}
                }
            }
        
        pipeline = [
            match_stage,
            group_stage,
            {"$sort": {"value": sort_order}},
            {"$limit": n}
        ]
        
        res = list(coll.aggregate(pipeline))
        
        # Format results
        ranked_entities = []
        for r in res:
            entity_name = r.get("_id", "")
            value = r.get("value")
            ranked_entities.append({
                "entity": str(entity_name),
                "value": Decimal(str(value)) if value is not None else Decimal("0")
            })
        
        matched_count = coll.count_documents(match_stage["$match"])
        
        return {
            "ok": True,
            "tool": "rank_entities",
            "result": {
                "entities": ranked_entities,
                "count": len(ranked_entities),
                "metric": metric_col,
                "aggregation": agg,
                "order": order
            },
            "unit": None,
            "provenance": {
                "mongo_pipeline": pipeline,
                "matched_row_count": matched_count
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in rank_entities: {str(e)}")
        return {
            "ok": False,
            "tool": "rank_entities",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def parse_relative_date(
    date_expression: str,
    reference_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Parse relative date expressions and convert to absolute dates.
    
    Examples:
    - "last month" → start/end of previous month
    - "this week" → start/end of current week
    - "last 30 days" → 30 days ago to today
    - "Q1 2025" → Jan 1 - Mar 31, 2025
    - "July 25" → July 25 of current/next year
    """
    start_ms = now_ms()
    
    try:
        if reference_date is None:
            reference_date = datetime.now()
        
        date_expr_lower = date_expression.lower().strip()
        
        # Relative expressions
        if "last month" in date_expr_lower or "previous month" in date_expr_lower:
            # First day of last month
            if reference_date.month == 1:
                start_date = datetime(reference_date.year - 1, 12, 1)
            else:
                start_date = datetime(reference_date.year, reference_date.month - 1, 1)
            # Last day of last month
            last_day = calendar.monthrange(start_date.year, start_date.month)[1]
            end_date = datetime(start_date.year, start_date.month, last_day)
            
            return {
                "ok": True,
                "tool": "parse_relative_date",
                "result": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": "last_month",
                    "expression": date_expression
                },
                "unit": None,
                "provenance": {
                    "reference_date": reference_date.isoformat(),
                    "parsed_as": "last_month"
                },
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        elif "this month" in date_expr_lower or "current month" in date_expr_lower:
            start_date = datetime(reference_date.year, reference_date.month, 1)
            last_day = calendar.monthrange(reference_date.year, reference_date.month)[1]
            end_date = datetime(reference_date.year, reference_date.month, last_day)
            
            return {
                "ok": True,
                "tool": "parse_relative_date",
                "result": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": "this_month",
                    "expression": date_expression
                },
                "unit": None,
                "provenance": {"reference_date": reference_date.isoformat()},
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        elif "last week" in date_expr_lower:
            days_since_monday = reference_date.weekday()
            last_monday = reference_date - timedelta(days=days_since_monday + 7)
            start_date = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            return {
                "ok": True,
                "tool": "parse_relative_date",
                "result": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": "last_week",
                    "expression": date_expression
                },
                "unit": None,
                "provenance": {"reference_date": reference_date.isoformat()},
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        elif "this week" in date_expr_lower:
            days_since_monday = reference_date.weekday()
            this_monday = reference_date - timedelta(days=days_since_monday)
            start_date = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = reference_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            return {
                "ok": True,
                "tool": "parse_relative_date",
                "result": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": "this_week",
                    "expression": date_expression
                },
                "unit": None,
                "provenance": {"reference_date": reference_date.isoformat()},
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # "last N days" pattern
        days_match = re.search(r'last\s+(\d+)\s+days?', date_expr_lower)
        if days_match:
            days = int(days_match.group(1))
            end_date = reference_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            start_date = (reference_date - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            return {
                "ok": True,
                "tool": "parse_relative_date",
                "result": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": f"last_{days}_days",
                    "expression": date_expression
                },
                "unit": None,
                "provenance": {"reference_date": reference_date.isoformat()},
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Try parsing as absolute date
        if date_parser:
            try:
                parsed_date = date_parser.parse(date_expression, fuzzy=True)
                return {
                    "ok": True,
                    "tool": "parse_relative_date",
                    "result": {
                        "start_date": parsed_date.isoformat(),
                        "end_date": parsed_date.isoformat(),
                        "type": "absolute_date",
                        "expression": date_expression
                    },
                    "unit": None,
                    "provenance": {"reference_date": reference_date.isoformat()},
                    "meta": {"time_ms": now_ms() - start_ms}
                }
            except:
                pass
        
        return {
            "ok": False,
            "tool": "parse_relative_date",
            "error": "could_not_parse",
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in parse_relative_date: {str(e)}")
        return {
            "ok": False,
            "tool": "parse_relative_date",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def extract_dates_from_filenames(
    user_id: str,
    file_name_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract date information from file names.
    Handles patterns like:
    - "July 25", "July 2025", "2025-07-25"
    - "Q1 2025", "Q2-2025"
    - "2025-07", "07-2025"
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        files_coll = db['files']
        
        # Convert user_id
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Get all files
        query = {"user_id": user_id_obj}
        if file_name_pattern:
            query["original_filename"] = {"$regex": file_name_pattern, "$options": "i"}
        
        files = list(files_coll.find(query, {"file_id": 1, "original_filename": 1}))
        
        current_date = datetime.now()
        extracted_dates = []
        
        # Month name mapping
        month_names = {month.lower(): i for i, month in enumerate(calendar.month_name[1:], 1)}
        month_abbr = {abbr.lower(): i for i, abbr in enumerate(calendar.month_abbr[1:], 1)}
        
        for file_doc in files:
            filename = file_doc.get("original_filename", "")
            file_id = file_doc.get("file_id")
            
            date_info = {
                "file_id": file_id,
                "filename": filename,
                "extracted_dates": []
            }
            
            # Pattern 1: YYYY-MM-DD, YYYY/MM/DD, etc.
            date_patterns = [
                (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
                (r'(\d{4})/(\d{2})/(\d{2})', '%Y/%m/%d'),
                (r'(\d{2})-(\d{2})-(\d{4})', '%m-%d-%Y'),
                (r'(\d{2})/(\d{2})/(\d{4})', '%m/%d/%Y'),
            ]
            
            for pattern, fmt in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    try:
                        date_str = match.group(0)
                        parsed_date = datetime.strptime(date_str, fmt)
                        date_info["extracted_dates"].append({
                            "date": parsed_date.isoformat(),
                            "type": "specific_date",
                            "pattern": date_str
                        })
                    except:
                        pass
            
            # Pattern 2: Month name + day/year (e.g., "July 25", "July 2025")
            month_day_pattern = r'(\w+)\s+(\d{1,2})(?:\s+(\d{4}))?'
            match = re.search(month_day_pattern, filename, re.IGNORECASE)
            if match:
                month_str = match.group(1).lower()
                day = int(match.group(2))
                year = int(match.group(3)) if match.group(3) else current_date.year
                
                month_num = month_names.get(month_str) or month_abbr.get(month_str)
                if month_num:
                    try:
                        parsed_date = datetime(year, month_num, day)
                        date_info["extracted_dates"].append({
                            "date": parsed_date.isoformat(),
                            "type": "month_day",
                            "pattern": match.group(0)
                        })
                    except:
                        pass
            
            # Pattern 3: YYYY-MM or MM-YYYY (month only)
            month_only_patterns = [
                (r'(\d{4})-(\d{2})(?!-\d{2})', '%Y-%m'),  # YYYY-MM
                (r'(\d{2})-(\d{4})(?!-\d{2})', '%m-%Y'),  # MM-YYYY
            ]
            for pattern, fmt in month_only_patterns:
                match = re.search(pattern, filename)
                if match:
                    try:
                        date_str = match.group(0)
                        parsed_date = datetime.strptime(date_str, fmt)
                        date_info["extracted_dates"].append({
                            "date": parsed_date.isoformat(),
                            "type": "month_only",
                            "pattern": date_str
                        })
                    except:
                        pass
            
            # Pattern 4: Quarter (Q1 2025, Q2-2025)
            quarter_pattern = r'Q([1-4])(?:\s+|-)?(\d{4})?'
            match = re.search(quarter_pattern, filename, re.IGNORECASE)
            if match:
                quarter = int(match.group(1))
                year = int(match.group(2)) if match.group(2) else current_date.year
                month = (quarter - 1) * 3 + 1
                start_date = datetime(year, month, 1)
                end_date = datetime(year, month + 2, calendar.monthrange(year, month + 2)[1])
                date_info["extracted_dates"].append({
                    "date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "type": "quarter",
                    "pattern": match.group(0)
                })
            
            if date_info["extracted_dates"]:
                extracted_dates.append(date_info)
        
        return {
            "ok": True,
            "tool": "extract_dates_from_filenames",
            "result": {
                "files_with_dates": extracted_dates,
                "total_files": len(extracted_dates)
            },
            "unit": None,
            "provenance": {
                "current_date": current_date.isoformat(),
                "files_searched": len(files)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in extract_dates_from_filenames: {str(e)}")
        return {
            "ok": False,
            "tool": "extract_dates_from_filenames",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def detect_date_columns(
    user_id: str,
    file_id: Optional[str] = None,  # "*" for all files
    table_name: Optional[str] = None,  # "*" for all sheets
    file_name_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Auto-detect all columns that contain date values across files/sheets.
    Returns columns with date patterns, date ranges, and formats.
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        files_coll = db['files']
        
        # Build query
        query = build_multi_search_query(user_id, file_id, table_name, file_name_pattern)
        if query is None:
            return {
                "ok": False,
                "tool": "detect_date_columns",
                "error": "no_matching_files",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Sample rows to detect date columns
        cursor = coll.find(query).limit(1000)
        rows = list(cursor)
        
        if not rows:
            return {
                "ok": False,
                "tool": "detect_date_columns",
                "error": "no_rows",
                "result": None,
                "unit": None,
                "provenance": None,
                "meta": {"time_ms": now_ms() - start_ms}
            }
        
        # Analyze each column for date patterns
        date_columns_by_source = {}
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
        ]
        
        for row in rows:
            source_key = f"{row.get('file_id')}::{row.get('table_name')}"
            row_data = row.get("row", {})
            
            if source_key not in date_columns_by_source:
                date_columns_by_source[source_key] = {}
            
            for col_name, col_value in row_data.items():
                if col_name not in date_columns_by_source[source_key]:
                    date_columns_by_source[source_key][col_name] = {
                        "date_count": 0,
                        "total_count": 0,
                        "min_date": None,
                        "max_date": None
                    }
                
                date_info = date_columns_by_source[source_key][col_name]
                date_info["total_count"] += 1
                
                if col_value is not None:
                    # Try to parse as date
                    is_date = False
                    parsed_date = None
                    
                    # Check if it's already a datetime object
                    if isinstance(col_value, datetime):
                        is_date = True
                        parsed_date = col_value
                    # Check string patterns
                    elif isinstance(col_value, str):
                        # Try dateutil parser (handles many formats)
                        if date_parser:
                            try:
                                parsed_date = date_parser.parse(col_value, fuzzy=False)
                                is_date = True
                            except:
                                # Check regex patterns
                                for pattern in date_patterns:
                                    if re.match(pattern, col_value):
                                        try:
                                            parsed_date = date_parser.parse(col_value)
                                            is_date = True
                                            break
                                        except:
                                            pass
                    
                    if is_date and parsed_date:
                        date_info["date_count"] += 1
                        if date_info["min_date"] is None or parsed_date < date_info["min_date"]:
                            date_info["min_date"] = parsed_date
                        if date_info["max_date"] is None or parsed_date > date_info["max_date"]:
                            date_info["max_date"] = parsed_date
        
        # Filter columns that are likely dates (>= 50% date values)
        confirmed_date_columns = {}
        for source_key, columns in date_columns_by_source.items():
            confirmed_date_columns[source_key] = {}
            for col_name, info in columns.items():
                if info["total_count"] > 0:
                    date_ratio = info["date_count"] / info["total_count"]
                    if date_ratio >= 0.5:  # At least 50% are dates
                        confirmed_date_columns[source_key][col_name] = {
                            "confidence": date_ratio,
                            "min_date": info["min_date"].isoformat() if info["min_date"] else None,
                            "max_date": info["max_date"].isoformat() if info["max_date"] else None,
                            "sample_count": info["total_count"]
                        }
        
        # Get file names
        file_ids = set(r.get("file_id") for r in rows)
        file_metadata = {}
        for fid in file_ids:
            file_doc = files_coll.find_one({"file_id": fid}, {"original_filename": 1})
            if file_doc:
                file_metadata[fid] = file_doc.get("original_filename", "Unknown")
        
        return {
            "ok": True,
            "tool": "detect_date_columns",
            "result": {
                "date_columns_by_source": confirmed_date_columns,
                "file_metadata": file_metadata,
                "total_sources": len(file_ids)
            },
            "unit": None,
            "provenance": {
                "mongo_query": query,
                "sources_analyzed": len(file_ids)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in detect_date_columns: {str(e)}")
        return {
            "ok": False,
            "tool": "detect_date_columns",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def search_across_all_files(
    user_id: str,
    column_name: str,
    search_value: Optional[str] = None,
    file_name_pattern: Optional[str] = None,
    table_name_pattern: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Search for a column across ALL files and sheets.
    Useful when user asks "find X in all my files"
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        coll = db['tables']
        files_coll = db['files']
        
        # Convert user_id
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Build query
        query = {"user_id": user_id_obj}
        
        if file_name_pattern:
            # Find matching file_ids
            matching_files = list(files_coll.find(
                {"user_id": user_id_obj, "original_filename": {"$regex": file_name_pattern, "$options": "i"}},
                {"file_id": 1}
            ))
            if matching_files:
                query["file_id"] = {"$in": [f["file_id"] for f in matching_files]}
        
        if table_name_pattern:
            query["table_name"] = {"$regex": table_name_pattern, "$options": "i"}
        
        # Check if column exists
        query[f"row.{column_name}"] = {"$exists": True}
        
        if search_value:
            query[f"row.{column_name}"] = {"$regex": search_value, "$options": "i"}
        
        # Fetch results
        cursor = coll.find(query).limit(limit)
        rows = list(cursor)
        
        # Group by source
        results_by_source = {}
        for row in rows:
            file_id = row.get("file_id")
            table_name = row.get("table_name")
            source_key = f"{file_id}::{table_name}"
            
            if source_key not in results_by_source:
                file_doc = files_coll.find_one({"file_id": file_id}, {"original_filename": 1})
                results_by_source[source_key] = {
                    "file_id": file_id,
                    "file_name": file_doc.get("original_filename", "Unknown") if file_doc else "Unknown",
                    "table_name": table_name,
                    "matches": []
                }
            
            results_by_source[source_key]["matches"].append({
                "row_id": row.get("row_id"),
                "value": row.get("row", {}).get(column_name),
                "row_data": row.get("row", {})
            })
        
        return {
            "ok": True,
            "tool": "search_across_all_files",
            "result": {
                "column_name": column_name,
                "results_by_source": list(results_by_source.values()),
                "total_matches": len(rows),
                "sources_found": len(results_by_source)
            },
            "unit": None,
            "provenance": {
                "mongo_query": query,
                "matched_row_count": len(rows)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in search_across_all_files: {str(e)}")
        return {
            "ok": False,
            "tool": "search_across_all_files",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }


def list_user_files(user_id: str) -> Dict[str, Any]:
    """
    List all files available for the user.
    
    Args:
        user_id: User ID for multi-tenant filtering
    
    Returns:
        Canonical tool envelope with list of files and their metadata
    """
    start_ms = now_ms()
    
    try:
        from bson import ObjectId
        db = get_sync_mongo_client()
        files_collection = db['files']
        
        # Convert user_id to ObjectId if needed
        try:
            user_id_obj = ObjectId(user_id) if isinstance(user_id, str) and len(user_id) == 24 else user_id
        except:
            user_id_obj = user_id
        
        # Get all files for user
        files = list(files_collection.find(
            {"user_id": user_id_obj},
            {"file_id": 1, "original_filename": 1, "file_type": 1, "metadata": 1, "_id": 0}
        ))
        
        # Format file list
        file_list = []
        for file_doc in files:
            metadata = file_doc.get("metadata", {})
            sheets = metadata.get("sheets", {})
            file_list.append({
                "file_id": file_doc.get("file_id"),
                "filename": file_doc.get("original_filename"),
                "file_type": file_doc.get("file_type"),
                "table_names": list(sheets.keys()) if sheets else ["Sheet1"],
                "row_count": sum(sheet.get("row_count", 0) for sheet in sheets.values()) if sheets else 0
            })
        
        return {
            "ok": True,
            "tool": "list_user_files",
            "result": {
                "files": file_list,
                "count": len(file_list)
            },
            "unit": None,
            "provenance": {
                "user_id": str(user_id_obj),
                "matched_files": len(file_list)
            },
            "meta": {"time_ms": now_ms() - start_ms}
        }
    except Exception as e:
        logger.error(f"Error in list_user_files: {str(e)}")
        return {
            "ok": False,
            "tool": "list_user_files",
            "error": str(e),
            "result": None,
            "unit": None,
            "provenance": None,
            "meta": {"time_ms": now_ms() - start_ms}
        }

