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

class SmartFileReader:
    @staticmethod
    def read_file(file: UploadFile) -> Tuple[pd.DataFrame, List[str]]:
        warnings = []
        try:
            content = file.file.read()
            if len(content) > 50 * 1024 * 1024:
                raise ValueError("File too large. Maximum size is 50MB.")
            
            if file.filename.endswith('.csv'):
                df, csv_warnings = SmartFileReader._read_csv_smart(content, file.filename)
                warnings.extend(csv_warnings)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df, excel_warnings = SmartFileReader._read_excel_smart(content, file.filename)
                warnings.extend(excel_warnings)
            else:
                raise ValueError("Unsupported file format. Please upload CSV (.csv) or Excel (.xlsx, .xls) files only.")
            
            if df is None or df.empty:
                raise ValueError("File appears to be empty or contains no valid data.")
            
            if len(df) > 100000:
                raise ValueError(f"File contains {len(df):,} rows. Maximum supported is 100,000 rows.")
            
            if len(df) < 2:
                raise ValueError("File contains too few rows. Need at least 2 transactions to compare prices.")
            
            logger.info(f"Successfully loaded {len(df)} rows from {file.filename}")
            return df, warnings
        except Exception as e:
            logger.error(f"File read error: {str(e)}")
            raise ValueError(f"Unable to read file: {str(e)}")
    
    @staticmethod
    def _read_csv_smart(content: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
        warnings = []
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
        
        lines = decoded_content.split('\n')
        header_row_idx = SmartFileReader._find_header_row(lines)
        
        if header_row_idx > 0:
            warnings.append(f"Detected {header_row_idx} metadata row(s) at top of file. Skipped automatically.")
        
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
        warnings = []
        try:
            excel_file = pd.ExcelFile(io.BytesIO(content))
            sheet_name = excel_file.sheet_names[0]
            
            if len(excel_file.sheet_names) > 1:
                warnings.append(f"File contains {len(excel_file.sheet_names)} sheets. Using first sheet: '{sheet_name}'")
            
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            header_row_idx = SmartFileReader._find_header_row_excel(df_raw)
            
            if header_row_idx > 0:
                warnings.append(f"Detected {header_row_idx} metadata row(s) at top of sheet. Skipped automatically.")
            
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row_idx)
            return df, warnings
        except Exception as e:
            raise ValueError(f"Excel read error: {str(e)}")
    
    @staticmethod
    def _find_header_row(lines: List[str]) -> int:
        keywords = ['vendor', 'supplier', 'item', 'sku', 'product', 'price', 'cost', 'date', 'invoice', 'amount', 'quantity', 'description', 'total', 'material', 'rate']
        for idx, line in enumerate(lines[:10]):
            line_lower = line.lower()
            keyword_count = sum(1 for kw in keywords if kw in line_lower)
            if keyword_count >= 3:
                return idx
        return 0
    
    @staticmethod
    def _find_header_row_excel(df: pd.DataFrame) -> int:
        keywords = ['vendor', 'supplier', 'item', 'sku', 'product', 'price', 'cost', 'date', 'invoice', 'amount', 'quantity', 'description', 'total', 'material', 'rate']
        for idx in range(min(10, len(df))):
            row_str = ' '.join(str(val).lower() for val in df.iloc[idx] if pd.notna(val))
            keyword_count = sum(1 for kw in keywords if kw in row_str)
            if keyword_count >= 3:
                return idx
        return 0

class PriceAnalyzer:
    def __init__(self, threshold_percent: float = 5.0):
        self.threshold = threshold_percent
        self.warnings = []
        
    def analyze_dataframe(self, df: pd.DataFrame) -> AnalysisResult:
        try:
            original_rows = len(df)
            df = self._normalize_columns(df)
            column_mapping = self._detect_columns(df)
            
            if not column_mapping:
                raise ValueError(self._build_helpful_error_message(df))
            
            df = df.rename(columns=column_mapping)
            df = self._clean_data(df)
            df = self._calculate_unit_prices(df)
            
            rows_after_cleaning = len(df)
            if rows_after_cleaning < original_rows * 0.5:
                self.warnings.append(f"Warning: {original_rows - rows_after_cleaning} rows removed due to missing/invalid data.")
            
            if rows_after_cleaning == 0:
                raise ValueError("No valid data rows found after cleaning.")
            
            results = self._analyze_price_changes(df)
            return results
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected error occurred during analysis.")
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
        return df
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Improved column detection with flexible matching"""
        mapping = {}
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Vendor detection - must come first
            if 'vendor' not in mapping.values():
                if any(x in col_lower for x in ['vendor', 'supplier']):
                    mapping[col] = 'vendor'
                    continue
            
            # Item detection - flexible matching
            if 'item' not in mapping.values():
                if any(x in col_lower for x in ['item', 'sku', 'product', 'material', 'description', 'part']):
                    # Avoid matching "item description" to both item and description
                    if 'product' in col_lower and 'service' in col_lower:
                        mapping[col] = 'item'
                        continue
                    elif 'material' in col_lower and ('desc' in col_lower or 'description' in col_lower):
                        mapping[col] = 'item'
                        continue
                    elif col_lower in ['item', 'sku', 'product', 'material', 'part']:
                        mapping[col] = 'item'
                        continue
                    elif 'description' not in col_lower:
                        mapping[col] = 'item'
                        continue
            
            # Price detection - exclude totals
            if 'unit_price' not in mapping.values():
                if ('price' in col_lower or 'cost' in col_lower or col_lower == 'rate') and 'total' not in col_lower:
                    mapping[col] = 'unit_price'
                    continue
            
            # Date detection
            if 'date' not in mapping.values():
                if 'date' in col_lower:
                    mapping[col] = 'date'
                    continue
            
            # Quantity detection
            if 'quantity' not in mapping.values():
                if any(x in col_lower for x in ['quantity', 'qty', 'ordered']):
                    if 'order' not in col_lower or 'quantity' in col_lower or 'qty' in col_lower:
                        mapping[col] = 'quantity'
                        continue
            
            # Total detection
            if 'total' not in mapping.values():
                if 'total' in col_lower or (col_lower == 'amount' and 'unit' not in col_lower):
                    mapping[col] = 'total'
                    continue
        
        return mapping
    
    def _build_helpful_error_message(self, df: pd.DataFrame) -> str:
        msg = "Could not detect required columns in your file.\n\n"
        msg += "📋 Columns found in your file:\n"
        for col in df.columns[:20]:
            msg += f"  • {col}\n"
        if len(df.columns) > 20:
            msg += f"  ... and {len(df.columns) - 20} more\n"
        msg += "\n✅ Required columns (with common variations):\n"
        msg += "  • Vendor: 'Vendor', 'Supplier', 'Vendor Name', 'Supplier Name'\n"
        msg += "  • Item: 'Item', 'SKU', 'Product', 'Material', 'Description'\n"
        msg += "  • Price: 'Price', 'Unit Price', 'Cost', 'Rate', 'Unit Cost'\n"
        msg += "  • Date: 'Date', 'Invoice Date', 'Purchase Date', 'PO Date', 'Document Date'\n"
        return msg
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df = df.dropna(how='all')
        
        required = ['vendor', 'item', 'date']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}.")
        
        if 'unit_price' not in df.columns and 'total' not in df.columns:
            raise ValueError("No price column found. Need either 'Unit Price' or 'Total' column.")
        
        df = df.dropna(subset=['vendor', 'item'])
        
        # Enhanced date parsing with multiple formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
        
        # Try European format (DD.MM.YYYY) if many failed
        if df['date'].isna().sum() > len(df) * 0.3:
            df.loc[df['date'].isna(), 'date'] = pd.to_datetime(
                df.loc[df['date'].isna(), 'date'], 
                format='%d.%m.%Y', 
                errors='coerce'
            )
        
        date_nulls = df['date'].isna().sum()
        if date_nulls > 0:
            self.warnings.append(f"{date_nulls} rows have invalid dates and were excluded.")
            df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            raise ValueError("No rows with valid dates found.")
        
        df['vendor'] = df['vendor'].astype(str).str.strip()
        df['item'] = df['item'].astype(str).str.strip()
        df = df[df['vendor'].str.len() > 0]
        df = df[df['item'].str.len() > 0]
        df = df[df['vendor'] != 'nan']
        df = df[df['item'] != 'nan']
        
        return df
    
    def _calculate_unit_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'unit_price' in df.columns:
            df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        elif 'total' in df.columns and 'quantity' in df.columns:
            df['total'] = pd.to_numeric(df['total'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['unit_price'] = df.apply(lambda row: row['total'] / row['quantity'] if row['quantity'] > 0 else None, axis=1)
        else:
            raise ValueError("Cannot determine unit prices.")
        
        initial_count = len(df)
        df = df.dropna(subset=['unit_price'])
        df = df[df['unit_price'] > 0]
        df = df[df['unit_price'] < 1000000]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.warnings.append(f"{removed} rows had invalid prices and were excluded.")
        
        if len(df) == 0:
            raise ValueError("No valid prices found.")
        
        return df
    
    def _analyze_price_changes(self, df: pd.DataFrame) -> AnalysisResult:
        results = []
        grouped = df.groupby(['vendor', 'item'])
        
        for (vendor, item), group in grouped:
            if len(group) < 2:
                continue
            
            sorted_group = group.sort_values('date')
            last_two = sorted_group.tail(2)
            
            previous = last_two.iloc[0]
            current = last_two.iloc[1]
            
            price_change_pct = ((current['unit_price'] - previous['unit_price']) / previous['unit_price']) * 100
            
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

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel PantherEdge</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8fafc; }
        .hero { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); color: white; padding: 3rem 0; }
        .upload-zone { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem; text-align: center; background: white; }
        .upload-zone:hover { border-color: #2563eb; background: #f1f5f9; }
        .metric-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2rem; font-weight: 700; }
        .loading { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9999; }
        .loading.active { display: flex; align-items: center; justify-content: center; }
        .spinner { border: 4px solid #f3f4f6; border-top: 4px solid #2563eb; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="loading" id="loading"><div class="spinner"></div></div>
    <div class="hero"><div class="container"><h1 class="display-4 fw-bold">🛡️ Sentinel PantherEdge</h1><p class="lead">Automated vendor overcharge detection</p></div></div>
    <div class="container my-5">
        <div class="row mb-4">
            <div class="col-md-8">
                <div class="upload-zone" id="uploadZone">
                    <h3>Upload Your Invoice Data</h3>
                    <p class="text-muted">CSV or Excel file</p>
                    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
                    <button class="btn btn-primary btn-lg" onclick="document.getElementById('fileInput').click()">Choose File</button>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h5>Threshold</h5>
                    <div class="input-group"><input type="number" class="form-control" id="threshold" value="5" min="0" max="100"><span class="input-group-text">%</span></div>
                </div>
            </div>
        </div>
        <div id="results" style="display: none;">
            <div class="row mb-4">
                <div class="col-md-4"><div class="metric-card text-center"><div class="metric-value" id="totalItems">0</div><div>Items Analyzed</div></div></div>
                <div class="col-md-4"><div class="metric-card text-center"><div class="metric-value text-danger" id="flaggedItems">0</div><div>Price Increases</div></div></div>
                <div class="col-md-4"><div class="metric-card text-center"><div class="metric-value text-danger" id="totalOverage">$0</div><div>Potential Overage</div></div></div>
            </div>
            <div class="table-responsive"><table class="table table-hover" id="resultsTable"><thead><tr><th>Vendor</th><th>Item</th><th>Previous</th><th>Current</th><th>Change</th><th>Dates</th></tr></thead><tbody id="tableBody"></tbody></table></div>
            <button class="btn btn-success" onclick="exportCSV()">Export CSV</button>
        </div>
        <div id="noResults" style="display: none;" class="alert alert-success">No price increases detected!</div>
    </div>
    <script>
        let analysisData = null;
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        
        fileInput.addEventListener('change', handleFile);
        
        async function handleFile() {
            const file = fileInput.files[0];
            if (!file) return;
            
            document.getElementById('results').style.display = 'none';
            document.getElementById('noResults').style.display = 'none';
            loading.classList.add('active');
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('threshold', document.getElementById('threshold').value);
            
            try {
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }
                analysisData = await response.json();
                displayResults(analysisData);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.classList.remove('active');
            }
        }
        
        function displayResults(data) {
            document.getElementById('totalItems').textContent = data.total_items_analyzed;
            document.getElementById('flaggedItems').textContent = data.flagged_items;
            document.getElementById('totalOverage').textContent = '$' + data.total_overage_amount.toLocaleString();
            
            if (data.flagged_items === 0) {
                document.getElementById('noResults').style.display = 'block';
                return;
            }
            
            document.getElementById('results').style.display = 'block';
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';
            
            data.records.forEach(r => {
                const row = tbody.insertRow();
                row.innerHTML = `<td>${r.vendor}</td><td>${r.item}</td><td>$${r.previous_price}</td><td>$${r.current_price}</td><td class="text-danger">+${r.price_change_pct}%</td><td>${r.previous_date} → ${r.current_date}</td>`;
            });
        }
        
        function exportCSV() {
            if (!analysisData) return;
            let csv = 'Vendor,Item,Previous,Current,Change%,Amount,Date\\n';
            analysisData.records.forEach(r => {
                csv += `"${r.vendor}","${r.item}",${r.previous_price},${r.current_price},${r.price_change_pct},${r.absolute_change},"${r.previous_date} to ${r.current_date}"\\n`;
            });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sentinel-report-${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
        }
    </script>
</body>
</html>"""
    return html_content

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_file(file: UploadFile = File(...), threshold: float = Form(5.0)):
    logger.info(f"Processing file: {file.filename}")
    if threshold < 0 or threshold > 100:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 100.")
    
    try:
        df, file_warnings = SmartFileReader.read_file(file)
        analyzer = PriceAnalyzer(threshold_percent=threshold)
        results = analyzer.analyze_dataframe(df)
        results.warnings.extend(file_warnings)
        logger.info(f"Analysis complete: {results.flagged_items} items flagged")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Sentinel PantherEdge"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)