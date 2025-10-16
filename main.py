from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging
from pydantic import BaseModel, Field
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentinel PantherEdge",
    description="Automated vendor overcharge detection for SMBs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class PriceChangeRecord(BaseModel):
    vendor: str
    item: str
    previous_price: float
    current_price: float
    price_change_pct: float
    previous_date: str
    current_date: str
    absolute_change: float
    
class AnalysisResult(BaseModel):
    total_items_analyzed: int
    flagged_items: int
    total_overage_amount: float
    analysis_date: str
    threshold_used: float
    records: List[PriceChangeRecord]
    warnings: List[str] = []

# ============================================================================
# INTELLIGENT FILE READER - HANDLES MESSY REAL-WORLD DATA
# ============================================================================

class SmartFileReader:
    """
    Bulletproof file reader that handles:
    - Headers in wrong rows (common in ERP exports)
    - Multiple header rows
    - Empty rows at top
    - Metadata rows before data
    - Various encodings
    """
    
    @staticmethod
    def read_file(file: UploadFile) -> Tuple[pd.DataFrame, List[str]]:
        """Read file with automatic header detection"""
        warnings = []
        
        try:
            content = file.file.read()
            
            # Validate file size (50MB limit)
            if len(content) > 50 * 1024 * 1024:
                raise ValueError("File too large. Maximum size is 50MB.")
            
            # Read based on file type
            if file.filename.endswith('.csv'):
                df, csv_warnings = SmartFileReader._read_csv_smart(content, file.filename)
                warnings.extend(csv_warnings)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df, excel_warnings = SmartFileReader._read_excel_smart(content, file.filename)
                warnings.extend(excel_warnings)
            else:
                raise ValueError(
                    "Unsupported file format. Please upload CSV (.csv) or Excel (.xlsx, .xls) files only."
                )
            
            # Validate we got data
            if df is None or df.empty:
                raise ValueError("File appears to be empty or contains no valid data.")
            
            # Validate row count
            if len(df) > 100000:
                raise ValueError(
                    f"File contains {len(df):,} rows. Maximum supported is 100,000 rows. "
                    "Please filter your export to a specific date range."
                )
            
            if len(df) < 2:
                raise ValueError(
                    "File contains too few rows. Need at least 2 transactions to compare prices."
                )
            
            logger.info(f"Successfully loaded {len(df)} rows from {file.filename}")
            return df, warnings
            
        except UnicodeDecodeError:
            raise ValueError(
                "File encoding error. Please save your file as UTF-8 or standard CSV format."
            )
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
            if "Excel" in str(e) or "xlrd" in str(e):
                raise ValueError(
                    "Excel file error. Please try: 1) Re-saving as .xlsx, or 2) Exporting as CSV instead."
                )
            raise ValueError(f"Unable to read file: {str(e)}")
    
    @staticmethod
    def _read_csv_smart(content: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
        """Smart CSV reader with automatic header detection"""
        warnings = []
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        decoded_content = None
        
        for encoding in encodings:
            try:
                decoded_content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if decoded_content is None:
            raise ValueError("Could not decode file. Please save as UTF-8 CSV.")
        
        # Find the header row (where actual column names are)
        lines = decoded_content.split('\n')
        header_row_idx = SmartFileReader._find_header_row(lines)
        
        if header_row_idx > 0:
            warnings.append(
                f"Detected {header_row_idx} metadata row(s) at top of file. Skipped automatically."
            )
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        best_df = None
        max_columns = 0
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    io.StringIO(decoded_content),
                    skiprows=header_row_idx,
                    delimiter=delimiter,
                    encoding='utf-8',
                    skip_blank_lines=True,
                    on_bad_lines='skip'
                )
                
                # Choose delimiter that gives most columns (most likely correct)
                if len(df.columns) > max_columns:
                    max_columns = len(df.columns)
                    best_df = df
                    
            except Exception:
                continue
        
        if best_df is None or best_df.empty:
            raise ValueError("Could not parse CSV. Please check file format.")
        
        return best_df, warnings
    
    @staticmethod
    def _read_excel_smart(content: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
        """Smart Excel reader with automatic header detection"""
        warnings = []
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(io.BytesIO(content))
            
            # If multiple sheets, use first sheet with data
            sheet_name = excel_file.sheet_names[0]
            if len(excel_file.sheet_names) > 1:
                warnings.append(
                    f"File contains {len(excel_file.sheet_names)} sheets. "
                    f"Using first sheet: '{sheet_name}'"
                )
            
            # Read with no header first to detect structure
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            # Find header row
            header_row_idx = SmartFileReader._find_header_row_excel(df_raw)
            
            if header_row_idx > 0:
                warnings.append(
                    f"Detected {header_row_idx} metadata row(s) at top of sheet. Skipped automatically."
                )
            
            # Re-read with correct header
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row_idx)
            
            return df, warnings
            
        except Exception as e:
            raise ValueError(f"Excel read error: {str(e)}")
    
    @staticmethod
    def _find_header_row(lines: List[str]) -> int:
        """
        Intelligently detect which row contains actual column headers
        Common patterns in ERP exports:
        - Row 0: Company name or report title
        - Row 1: Date range or filter info
        - Row 2: Actual headers (Vendor, Item, Price, etc.)
        """
        keywords = [
            'vendor', 'supplier', 'item', 'sku', 'product', 'price', 'cost',
            'date', 'invoice', 'amount', 'quantity', 'description', 'total'
        ]
        
        for idx, line in enumerate(lines[:10]):  # Check first 10 rows
            line_lower = line.lower()
            
            # Count how many expected column names appear
            keyword_count = sum(1 for kw in keywords if kw in line_lower)
            
            # If we find 3+ expected columns, this is likely the header
            if keyword_count >= 3:
                return idx
        
        # Default to row 0 if no clear header found
        return 0
    
    @staticmethod
    def _find_header_row_excel(df: pd.DataFrame) -> int:
        """Find header row in Excel DataFrame"""
        keywords = [
            'vendor', 'supplier', 'item', 'sku', 'product', 'price', 'cost',
            'date', 'invoice', 'amount', 'quantity', 'description', 'total'
        ]
        
        for idx in range(min(10, len(df))):
            row_str = ' '.join(str(val).lower() for val in df.iloc[idx] if pd.notna(val))
            keyword_count = sum(1 for kw in keywords if kw in row_str)
            
            if keyword_count >= 3:
                return idx
        
        return 0

# ============================================================================
# BULLETPROOF DATA ANALYZER
# ============================================================================

class PriceAnalyzer:
    """
    Production-grade analyzer with extensive validation and error handling
    """
    
    def __init__(self, threshold_percent: float = 5.0):
        self.threshold = threshold_percent
        self.warnings = []
        
    def analyze_dataframe(self, df: pd.DataFrame) -> AnalysisResult:
        """Analyze with bulletproof error handling"""
        try:
            original_rows = len(df)
            
            # Step 1: Normalize column names
            df = self._normalize_columns(df)
            
            # Step 2: Validate and map columns
            column_mapping = self._detect_columns(df)
            if not column_mapping:
                raise ValueError(self._build_helpful_error_message(df))
            
            # Step 3: Rename to standard names
            df = df.rename(columns=column_mapping)
            
            # Step 4: Clean and validate data
            df = self._clean_data(df)
            
            # Step 5: Calculate unit prices
            df = self._calculate_unit_prices(df)
            
            # Report data quality
            rows_after_cleaning = len(df)
            if rows_after_cleaning < original_rows * 0.5:
                self.warnings.append(
                    f"Warning: {original_rows - rows_after_cleaning} rows removed due to missing/invalid data. "
                    "This is normal for exports with subtotal or summary rows."
                )
            
            if rows_after_cleaning == 0:
                raise ValueError(
                    "No valid data rows found after cleaning. Please check that your file contains:\n"
                    "- Vendor/supplier names\n"
                    "- Item/product descriptions\n"
                    "- Prices (numeric values)\n"
                    "- Dates (in a recognizable format)"
                )
            
            # Step 6: Perform price analysis
            results = self._analyze_price_changes(df)
            
            return results
            
        except ValueError as e:
            # User-friendly errors
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Unexpected errors
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during analysis. Please contact support if this persists."
            )
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names"""
        # Remove leading/trailing spaces and convert to lowercase
        df.columns = df.columns.str.strip().str.lower()
        
        # Remove special characters
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        
        # Replace multiple spaces with single space
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
        
        return df
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Intelligently detect which columns map to our required fields
        Returns mapping of {actual_column_name: standard_name}
        """
        column_patterns = {
            'vendor': [
                r'vendor', r'supplier', r'vendor name', r'supplier name',
                r'company', r'manufacturer', r'seller'
            ],
            'item': [
                r'item', r'sku', r'product', r'description', r'item description',
                r'product name', r'item name', r'part', r'part number', r'material'
            ],
            'unit_price': [
                r'unit price', r'price', r'cost', r'unit cost', r'price per unit',
                r'rate', r'amount', r'unit amount', r'price each', r'each'
            ],
            'date': [
                r'date', r'invoice date', r'purchase date', r'order date',
                r'transaction date', r'posting date', r'doc date'
            ],
            'quantity': [
                r'quantity', r'qty', r'amount', r'units', r'count', r'number'
            ],
            'total': [
                r'total', r'total price', r'total cost', r'line total',
                r'extended price', r'extended cost'
            ]
        }
        
        mapping = {}
        
        for standard_name, patterns in column_patterns.items():
            for col in df.columns:
                col_clean = str(col).lower().strip()
                
                # Check if any pattern matches
                for pattern in patterns:
                    if re.search(pattern, col_clean):
                        mapping[col] = standard_name
                        break
                
                if col in mapping:
                    break
        
        return mapping
    
    def _build_helpful_error_message(self, df: pd.DataFrame) -> str:
        """Build helpful error message showing what columns were found"""
        msg = "Could not detect required columns in your file.\n\n"
        msg += "📋 Columns found in your file:\n"
        for col in df.columns[:20]:  # Show first 20 columns
            msg += f"  • {col}\n"
        
        if len(df.columns) > 20:
            msg += f"  ... and {len(df.columns) - 20} more\n"
        
        msg += "\n✅ Required columns (with common variations):\n"
        msg += "  • Vendor: 'Vendor', 'Supplier', 'Company'\n"
        msg += "  • Item: 'Item', 'SKU', 'Product', 'Description'\n"
        msg += "  • Price: 'Price', 'Unit Price', 'Cost'\n"
        msg += "  • Date: 'Date', 'Invoice Date', 'Purchase Date'\n"
        msg += "\n💡 Tip: Make sure your file has clearly labeled columns matching the patterns above."
        
        return msg
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data with detailed logging"""
        initial_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Check for required columns
        required = ['vendor', 'item', 'date']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                "Please ensure your file has Vendor, Item/SKU, and Date columns."
            )
        
        # Must have price data
        if 'unit_price' not in df.columns and 'total' not in df.columns:
            raise ValueError(
                "No price column found. Your file must have either:\n"
                "  • 'Unit Price' or 'Price' column, OR\n"
                "  • 'Total' column with 'Quantity' column (to calculate unit price)"
            )
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['vendor', 'item'])
        
        # Convert and validate dates with multiple format support
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m/%d/%y', '%d/%m/%y'
        ]
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
        
        # Try manual parsing for failed dates
        if df['date'].isna().any():
            def try_parse_date(date_val):
                if pd.isna(date_val):
                    return pd.NaT
                for fmt in date_formats:
                    try:
                        return pd.to_datetime(date_val, format=fmt)
                    except:
                        continue
                return pd.NaT
            
            df.loc[df['date'].isna(), 'date'] = df.loc[df['date'].isna(), 'date'].apply(try_parse_date)
        
        date_nulls = df['date'].isna().sum()
        if date_nulls > 0:
            self.warnings.append(f"{date_nulls} rows have invalid dates and were excluded.")
            df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            raise ValueError(
                "No rows with valid dates found. Please check your date column format. "
                "Supported formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc."
            )
        
        # Clean vendor and item names
        df['vendor'] = df['vendor'].astype(str).str.strip()
        df['item'] = df['item'].astype(str).str.strip()
        
        # Remove rows where vendor or item is just "nan" or empty
        df = df[df['vendor'].str.len() > 0]
        df = df[df['item'].str.len() > 0]
        df = df[df['vendor'] != 'nan']
        df = df[df['item'] != 'nan']
        
        final_rows = len(df)
        if final_rows < initial_rows * 0.3:
            self.warnings.append(
                f"Only {final_rows}/{initial_rows} rows contain valid data. "
                "This might indicate formatting issues in your export."
            )
        
        return df
    
    def _calculate_unit_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate unit prices with robust error handling"""
        
        if 'unit_price' in df.columns:
            # Already have unit price
            df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        elif 'total' in df.columns and 'quantity' in df.columns:
            # Calculate from total and quantity
            df['total'] = pd.to_numeric(df['total'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            
            # Avoid division by zero
            df['unit_price'] = df.apply(
                lambda row: row['total'] / row['quantity'] if row['quantity'] > 0 else None,
                axis=1
            )
        else:
            raise ValueError(
                "Cannot determine unit prices. Need either:\n"
                "  • 'Unit Price' column, OR\n"
                "  • Both 'Total' and 'Quantity' columns"
            )
        
        # Remove invalid prices
        initial_count = len(df)
        df = df.dropna(subset=['unit_price'])
        df = df[df['unit_price'] > 0]  # Remove negative or zero prices
        df = df[df['unit_price'] < 1000000]  # Remove unrealistic prices (likely data errors)
        
        removed = initial_count - len(df)
        if removed > 0:
            self.warnings.append(f"{removed} rows had invalid prices (negative, zero, or missing) and were excluded.")
        
        if len(df) == 0:
            raise ValueError(
                "No valid prices found. Please check that your price column contains positive numeric values."
            )
        
        return df
    
    def _analyze_price_changes(self, df: pd.DataFrame) -> AnalysisResult:
        """Perform the actual price change analysis"""
        results = []
        grouped = df.groupby(['vendor', 'item'])
        
        for (vendor, item), group in grouped:
            if len(group) < 2:
                continue
            
            # Sort by date and get last 2 records
            sorted_group = group.sort_values('date')
            last_two = sorted_group.tail(2)
            
            previous = last_two.iloc[0]
            current = last_two.iloc[1]
            
            # Calculate change
            price_change_pct = ((current['unit_price'] - previous['unit_price']) / previous['unit_price']) * 100
            
            # Flag if exceeds threshold
            if price_change_pct >= self.threshold:
                results.append(PriceChangeRecord(
                    vendor=str(vendor),
                    item=str(item),
                    previous_price=round(float(previous['unit_price']), 2),
                    current_price=round(float(current['unit_price']), 2),
                    price_change_pct=round(price_change_pct, 2),
                    previous_date=previous['date'].strftime('%Y-%m-%d'),
                    current_date=current['date'].strftime('%Y-%m-%d'),
                    absolute_change=round(float(current['unit_price'] - previous['unit_price']), 2)
                ))
        
        total_overage = sum(r.absolute_change for r in results)
        
        return AnalysisResult(
            total_items_analyzed=len(grouped),
            flagged_items=len(results),
            total_overage_amount=round(total_overage, 2),
            analysis_date=datetime.now().isoformat(),
            threshold_used=self.threshold,
            records=sorted(results, key=lambda x: x.price_change_pct, reverse=True),
            warnings=self.warnings
        )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentinel PantherEdge - Vendor Overcharge Detection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
        <style>
            :root {
                --primary: #2563eb;
                --danger: #dc2626;
                --success: #16a34a;
                --warning: #f59e0b;
            }
            body { background: #f8fafc; font-family: 'Inter', -apple-system, sans-serif; }
            .hero { background: linear-gradient(135deg, var(--primary) 0%, #1e40af 100%); color: white; padding: 3rem 0; }
            .upload-zone { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem; text-align: center; transition: all 0.3s; background: white; }
            .upload-zone:hover { border-color: var(--primary); background: #f1f5f9; }
            .upload-zone.dragover { border-color: var(--primary); background: #e0e7ff; }
            .metric-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
            .table-container { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .price-increase { color: var(--danger); font-weight: 600; }
            .badge-danger { background: #fef2f2; color: var(--danger); padding: 0.35rem 0.75rem; border-radius: 6px; }
            .btn-primary { background: var(--primary); border: none; }
            .btn-primary:hover { background: #1e40af; }
            .loading { display: none; }
            .loading.active { display: flex; align-items: center; justify-content: center; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9999; flex-direction: column; }
            .spinner { border: 4px solid #f3f4f6; border-top: 4px solid var(--primary); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .warning-box { background: #fffbeb; border-left: 4px solid var(--warning); padding: 1rem; margin: 1rem 0; border-radius: 6px; }
            .error-detail { background: #fef2f2; border: 1px solid #fecaca; padding: 1.5rem; border-radius: 8px; margin: 2rem 0; }
            .error-detail h4 { color: var(--danger); margin-bottom: 1rem; }
            .error-detail pre { background: white; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.875rem; }
        </style>
    </head>
    <body>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p class="text-white mt-3">Analyzing your data...</p>
        </div>
        
        <div class="hero">
            <div class="container">
                <h1 class="display-4 fw-bold"><i class="bi bi-shield-check"></i> Sentinel PantherEdge</h1>
                <p class="lead">Automated vendor overcharge detection for your business</p>
                <p class="small">Instantly identify price increases in your invoices and purchase orders</p>
            </div>
        </div>
        
        <div class="container my-5">
            <div class="row mb-4">
                <div class="col-md-8">
                    <div class="upload-zone" id="uploadZone">
                        <i class="bi bi-cloud-upload" style="font-size: 3rem; color: var(--primary);"></i>
                        <h3 class="mt-3">Upload Your Invoice Data</h3>
                        <p class="text-muted">CSV or Excel file from QuickBooks, NetSuite, SAP, or any ERP system</p>
                        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
                        <button class="btn btn-primary btn-lg mt-2" onclick="document.getElementById('fileInput').click()">
                            <i class="bi bi-file-earmark-arrow-up"></i> Choose File
                        </button>
                        <p class="text-muted mt-3 small"><strong>Required columns:</strong> Vendor, Item/SKU, Price, Date</p>
                        <p class="text-muted small">Maximum file size: 50MB | Maximum rows: 100,000</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <h5 class="text-muted">Price Increase Threshold</h5>
                        <div class="input-group mt-3">
                            <input type="number" class="form-control" id="threshold" value="5" min="0" max="100" step="0.5">
                            <span class="input-group-text">%</span>
                        </div>
                        <small class="text-muted">Flag items with price increases above this percentage</small>
                    </div>
                    
                    <div class="metric-card mt-3" style="background: #f0f9ff;">
                        <h6 class="text-primary"><i class="bi bi-info-circle"></i> How it works</h6>
                        <ol class="small mb-0 ps-3">
                            <li>Automatically detects headers in your file</li>
                            <li>Groups purchases by vendor + item</li>
                            <li>Compares last 2 purchase prices</li>
                            <li>Flags increases above threshold</li>
                        </ol>
                    </div>
                </div>
            </div>
            
            <div id="errorBox" class="error-detail" style="display: none;">
                <h4><i class="bi bi-exclamation-triangle"></i> Upload Error</h4>
                <p id="errorMessage"></p>
            </div>
            
            <div id="warnings" style="display: none;"></div>
            
            <div id="results" style="display: none;">
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <i class="bi bi-box-seam" style="font-size: 2rem; color: var(--primary);"></i>
                            <div class="metric-value" id="totalItems">0</div>
                            <div class="text-muted">Items Analyzed</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <i class="bi bi-exclamation-triangle" style="font-size: 2rem; color: var(--danger);"></i>
                            <div class="metric-value text-danger" id="totalOverage">$0</div>
                            <div class="text-muted">Potential Overage</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <button class="btn btn-success btn-lg w-100" onclick="exportCSV()">
                                <i class="bi bi-download"></i> Export Report
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table class="table table-hover mb-0" id="resultsTable">
                        <thead class="table-light">
                            <tr>
                                <th onclick="sortTable(0)" style="cursor: pointer;">Vendor <i class="bi bi-arrow-down-up"></i></th>
                                <th onclick="sortTable(1)" style="cursor: pointer;">Item <i class="bi bi-arrow-down-up"></i></th>
                                <th onclick="sortTable(2)" style="cursor: pointer;">Previous Price <i class="bi bi-arrow-down-up"></i></th>
                                <th onclick="sortTable(3)" style="cursor: pointer;">Current Price <i class="bi bi-arrow-down-up"></i></th>
                                <th onclick="sortTable(4)" style="cursor: pointer;">% Change <i class="bi bi-arrow-down-up"></i></th>
                                <th>Previous Date</th>
                                <th>Current Date</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="noResults" style="display: none;" class="alert alert-success mt-4">
                <i class="bi bi-check-circle"></i> <strong>Great news!</strong> No significant price increases detected above the threshold.
            </div>
        </div>
        
        <script>
            let analysisData = null;
            
            const fileInput = document.getElementById('fileInput');
            const uploadZone = document.getElementById('uploadZone');
            const loading = document.getElementById('loading');
            const errorBox = document.getElementById('errorBox');
            
            fileInput.addEventListener('change', handleFile);
            
            // Drag and drop
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });
            
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });
            
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    handleFile();
                }
            });
            
            async function handleFile() {
                const file = fileInput.files[0];
                if (!file) return;
                
                // Hide previous results/errors
                document.getElementById('results').style.display = 'none';
                document.getElementById('noResults').style.display = 'none';
                document.getElementById('errorBox').style.display = 'none';
                document.getElementById('warnings').style.display = 'none';
                
                // Validate file type
                const validTypes = ['.csv', '.xlsx', '.xls'];
                const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
                if (!validTypes.includes(fileExt)) {
                    showError('Invalid file type. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.');
                    return;
                }
                
                // Validate file size (50MB)
                if (file.size > 50 * 1024 * 1024) {
                    showError('File too large. Maximum file size is 50MB. Please filter your export to a shorter date range.');
                    return;
                }
                
                loading.classList.add('active');
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('threshold', document.getElementById('threshold').value);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Analysis failed');
                    }
                    
                    analysisData = await response.json();
                    displayResults(analysisData);
                } catch (error) {
                    showError(error.message);
                } finally {
                    loading.classList.remove('active');
                }
            }
            
            function showError(message) {
                errorBox.style.display = 'block';
                document.getElementById('errorMessage').innerHTML = message.replace(/\\n/g, '<br>');
                
                // Scroll to error
                errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            function displayResults(data) {
                // Display warnings if any
                if (data.warnings && data.warnings.length > 0) {
                    const warningsDiv = document.getElementById('warnings');
                    warningsDiv.innerHTML = data.warnings.map(w => 
                        `<div class="warning-box"><i class="bi bi-info-circle text-warning"></i> ${w}</div>`
                    ).join('');
                    warningsDiv.style.display = 'block';
                }
                
                document.getElementById('totalItems').textContent = data.total_items_analyzed.toLocaleString();
                document.getElementById('flaggedItems').textContent = data.flagged_items.toLocaleString();
                document.getElementById('totalOverage').textContent = 'danger" id="flaggedItems">0</div>
                            <div class="text-muted">Price Increases</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <i class="bi bi-cash-stack" style="font-size: 2rem; color: var(--danger);"></i>
                            <div class="metric-value text- + data.total_overage_amount.toLocaleString();
                
                if (data.flagged_items === 0) {
                    document.getElementById('results').style.display = 'none';
                    document.getElementById('noResults').style.display = 'block';
                    return;
                }
                
                document.getElementById('noResults').style.display = 'none';
                document.getElementById('results').style.display = 'block';
                
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';
                
                data.records.forEach(record => {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td><strong>${escapeHtml(record.vendor)}</strong></td>
                        <td>${escapeHtml(record.item)}</td>
                        <td>${record.previous_price.toFixed(2)}</td>
                        <td>${record.current_price.toFixed(2)}</td>
                        <td><span class="badge-danger">+${record.price_change_pct.toFixed(1)}%</span></td>
                        <td>${new Date(record.previous_date).toLocaleDateString()}</td>
                        <td>${new Date(record.current_date).toLocaleDateString()}</td>
                    `;
                });
                
                // Scroll to results
                document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            function sortTable(columnIndex) {
                const table = document.getElementById('resultsTable');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                const isNumeric = columnIndex >= 2 && columnIndex <= 4;
                
                rows.sort((a, b) => {
                    let aVal = a.cells[columnIndex].textContent.trim();
                    let bVal = b.cells[columnIndex].textContent.trim();
                    
                    if (isNumeric) {
                        aVal = parseFloat(aVal.replace(/[$,%+]/g, ''));
                        bVal = parseFloat(bVal.replace(/[$,%+]/g, ''));
                        return bVal - aVal;
                    }
                    
                    return aVal.localeCompare(bVal);
                });
                
                rows.forEach(row => tbody.appendChild(row));
            }
            
            function exportCSV() {
                if (!analysisData) return;
                
                let csv = 'Vendor,Item,Previous Price,Current Price,Change %,Change Amount,Previous Date,Current Date\\n';
                
                analysisData.records.forEach(record => {
                    csv += `"${record.vendor.replace(/"/g, '""')}","${record.item.replace(/"/g, '""')}",${record.previous_price},${record.current_price},${record.price_change_pct},${record.absolute_change},"${record.previous_date}","${record.current_date}"\\n`;
                });
                
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `sentinel-price-increases-${new Date().toISOString().split('T')[0]}.csv`;
                a.click();
            }
        </script>
    </body>
    </html>
    """

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_file(
    file: UploadFile = File(...),
    threshold: float = Form(5.0)
):
    """
    Main analysis endpoint with bulletproof error handling
    """
    logger.info(f"Processing file: {file.filename} (threshold: {threshold}%)")
    
    # Validate threshold
    if threshold < 0 or threshold > 100:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0 and 100 percent."
        )
    
    try:
        # Read file with smart detection
        df, file_warnings = SmartFileReader.read_file(file)
        
        # Run analysis
        analyzer = PriceAnalyzer(threshold_percent=threshold)
        results = analyzer.analyze_dataframe(df)
        
        # Combine warnings
        results.warnings.extend(file_warnings)
        
        logger.info(
            f"Analysis complete: {results.total_items_analyzed} items analyzed, "
            f"{results.flagged_items} flagged"
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sentinel PantherEdge",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)danger" id="flaggedItems">0</div>
                            <div class="text-muted">Price Increases</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <i class="bi bi-cash-stack" style="font-size: 2rem; color: var(--danger);"></i>
                            <div class="metric-value text-