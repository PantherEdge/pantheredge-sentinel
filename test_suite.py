"""
Comprehensive test suite for Sentinel PantherEdge
Run with: pytest test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient
from main import app, SmartFileReader, PriceAnalyzer
import pandas as pd
from io import BytesIO, StringIO
from datetime import datetime

client = TestClient(app)

# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test that root serves HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Sentinel PantherEdge" in response.text

def test_analyze_with_valid_csv():
    """Test successful analysis with clean CSV"""
    csv_data = """Vendor,Item,Unit Price,Date,Quantity
Acme Corp,Widget A,10.00,2024-01-15,100
Acme Corp,Widget A,11.50,2024-03-20,100
Global Supply,Gadget B,25.00,2024-02-01,50
Global Supply,Gadget B,28.00,2024-04-10,50"""
    
    files = {"file": ("test.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["total_items_analyzed"] == 2
    assert result["flagged_items"] == 2
    assert result["threshold_used"] == 5.0

def test_analyze_with_invalid_file_type():
    """Test that invalid file types are rejected"""
    files = {"file": ("test.txt", BytesIO(b"Invalid content"), "text/plain")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_analyze_with_empty_file():
    """Test that empty files are rejected"""
    csv_data = ""
    files = {"file": ("empty.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400

def test_analyze_with_invalid_threshold():
    """Test that invalid thresholds are rejected"""
    csv_data = "Vendor,Item,Price,Date\nTest,Test,10,2024-01-01"
    files = {"file": ("test.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "150"}  # Invalid: >100
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400

# ============================================================================
# FILE READER TESTS
# ============================================================================

def test_find_header_row_with_metadata():
    """Test automatic header detection with metadata rows"""
    lines = [
        "Company: ABC Manufacturing",
        "Report Date: 2024-01-01",
        "",
        "Vendor,Item,Price,Date"
    ]
    
    header_idx = SmartFileReader._find_header_row(lines)
    assert header_idx == 3

def test_find_header_row_clean():
    """Test header detection with clean data"""
    lines = ["Vendor,Item,Price,Date", "Acme,Widget,10,2024-01-01"]
    header_idx = SmartFileReader._find_header_row(lines)
    assert header_idx == 0

def test_csv_with_different_delimiters():
    """Test CSV parsing with different delimiters"""
    # Semicolon delimiter
    csv_data = "Vendor;Item;Price;Date\nAcme;Widget;10.00;2024-01-01"
    
    from fastapi import UploadFile
    from io import BytesIO
    
    class MockUploadFile:
        def __init__(self, content, filename):
            self.file = BytesIO(content)
            self.filename = filename
    
    upload_file = MockUploadFile(csv_data.encode(), "test.csv")
    df, warnings = SmartFileReader.read_file(upload_file)
    
    assert not df.empty
    assert len(df.columns) >= 4

def test_csv_encoding_detection():
    """Test handling of different encodings"""
    # UTF-8 with BOM
    csv_data = "\ufeffVendor,Item,Price,Date\nAcme,Widget,10,2024-01-01"
    
    class MockUploadFile:
        def __init__(self, content, filename):
            self.file = BytesIO(content)
            self.filename = filename
    
    upload_file = MockUploadFile(csv_data.encode('utf-8-sig'), "test.csv")
    df, warnings = SmartFileReader.read_file(upload_file)
    
    assert not df.empty

# ============================================================================
# ANALYZER TESTS
# ============================================================================

def test_column_detection():
    """Test intelligent column name detection"""
    df = pd.DataFrame({
        'Supplier Name': ['Acme', 'Acme'],
        'Product Description': ['Widget', 'Widget'],
        'Unit Cost': [10.0, 11.0],
        'Purchase Date': ['2024-01-01', '2024-02-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    df = analyzer._normalize_columns(df)
    mapping = analyzer._detect_columns(df)
    
    assert 'vendor' in mapping.values()
    assert 'item' in mapping.values()
    assert 'unit_price' in mapping.values() or 'total' in mapping.values()
    assert 'date' in mapping.values()

def test_date_format_parsing():
    """Test parsing various date formats"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme', 'Acme'],
        'item': ['Widget', 'Widget', 'Widget'],
        'unit_price': [10.0, 11.0, 12.0],
        'date': ['2024-01-15', '01/15/2024', '15-01-2024']
    })
    
    analyzer = PriceAnalyzer()
    df_clean = analyzer._clean_data(df)
    
    assert df_clean['date'].notna().all()
    assert all(isinstance(d, pd.Timestamp) for d in df_clean['date'])

def test_unit_price_calculation():
    """Test unit price calculation from total and quantity"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme'],
        'item': ['Widget', 'Widget'],
        'total': [100.0, 110.0],
        'quantity': [10, 10],
        'date': ['2024-01-01', '2024-02-01']
    })
    
    analyzer = PriceAnalyzer()
    df_with_price = analyzer._calculate_unit_prices(df)
    
    assert 'unit_price' in df_with_price.columns
    assert df_with_price['unit_price'].iloc[0] == 10.0
    assert df_with_price['unit_price'].iloc[1] == 11.0

def test_price_increase_detection():
    """Test that price increases are correctly flagged"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme', 'Global', 'Global'],
        'item': ['Widget', 'Widget', 'Gadget', 'Gadget'],
        'unit_price': [10.0, 11.0, 20.0, 21.0],  # 10% and 5% increases
        'date': ['2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    results = analyzer._analyze_price_changes(df)
    
    assert results.total_items_analyzed == 2
    assert results.flagged_items == 2  # Both exceed 5%
    assert all(r.price_change_pct >= 5.0 for r in results.records)

def test_below_threshold_not_flagged():
    """Test that increases below threshold are not flagged"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme'],
        'item': ['Widget', 'Widget'],
        'unit_price': [10.0, 10.3],  # Only 3% increase
        'date': ['2024-01-01', '2024-02-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    results = analyzer._analyze_price_changes(df)
    
    assert results.total_items_analyzed == 1
    assert results.flagged_items == 0

def test_single_purchase_ignored():
    """Test that items with only one purchase are ignored"""
    df = pd.DataFrame({
        'vendor': ['Acme'],
        'item': ['Widget'],
        'unit_price': [10.0],
        'date': ['2024-01-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    results = analyzer._analyze_price_changes(df)
    
    assert results.flagged_items == 0

def test_missing_vendor_removed():
    """Test that rows with missing vendors are removed"""
    df = pd.DataFrame({
        'vendor': ['Acme', None, ''],
        'item': ['Widget', 'Gadget', 'Tool'],
        'unit_price': [10.0, 20.0, 30.0],
        'date': ['2024-01-01', '2024-01-01', '2024-01-01']
    })
    
    analyzer = PriceAnalyzer()
    df = analyzer._normalize_columns(df)
    df_clean = analyzer._clean_data(df)
    
    assert len(df_clean) == 1
    assert df_clean.iloc[0]['vendor'] == 'acme'

def test_negative_prices_removed():
    """Test that negative or zero prices are removed"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme', 'Acme'],
        'item': ['Widget', 'Widget', 'Widget'],
        'unit_price': [10.0, -5.0, 0.0],
        'date': ['2024-01-01', '2024-02-01', '2024-03-01']
    })
    
    analyzer = PriceAnalyzer()
    df_clean = analyzer._calculate_unit_prices(df)
    
    assert len(df_clean) == 1
    assert df_clean['unit_price'].iloc[0] == 10.0

def test_multiple_purchases_uses_last_two():
    """Test that analysis compares last two purchases when multiple exist"""
    df = pd.DataFrame({
        'vendor': ['Acme'] * 5,
        'item': ['Widget'] * 5,
        'unit_price': [10.0, 11.0, 12.0, 13.0, 14.0],
        'date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    results = analyzer._analyze_price_changes(df)
    
    assert results.flagged_items == 1
    record = results.records[0]
    assert record.previous_price == 13.0
    assert record.current_price == 14.0

def test_overage_calculation():
    """Test that total overage is calculated correctly"""
    df = pd.DataFrame({
        'vendor': ['Acme', 'Acme', 'Global', 'Global'],
        'item': ['Widget', 'Widget', 'Gadget', 'Gadget'],
        'unit_price': [10.0, 15.0, 20.0, 25.0],  # +$5 and +$5
        'date': ['2024-01-01', '2024-02-01', '2024-01-01', '2024-02-01']
    })
    
    analyzer = PriceAnalyzer(threshold_percent=5.0)
    results = analyzer._analyze_price_changes(df)
    
    assert results.total_overage_amount == 10.0  # $5 + $5

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_quickbooks_format():
    """Test complete workflow with QuickBooks-style export"""
    csv_data = """Company: ABC Manufacturing Inc
Report: Purchase History
Date Range: Q1 2024

Vendor Name,Product/Service,Rate,Transaction Date,Qty
Acme Corporation,Part #12345,50.00,01/15/2024,10
Acme Corporation,Part #12345,55.00,03/20/2024,10
Global Supplies,Component XYZ,100.00,02/01/2024,5
Global Supplies,Component XYZ,110.00,04/10/2024,5"""
    
    files = {"file": ("quickbooks_export.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["flagged_items"] == 2
    assert len(result["warnings"]) > 0  # Should warn about skipped metadata rows

def test_end_to_end_excel_with_totals():
    """Test complete workflow with Excel file using Total column"""
    df = pd.DataFrame({
        'Vendor': ['Acme', 'Acme'],
        'Item': ['Widget', 'Widget'],
        'Quantity': [10, 10],
        'Total': [100.0, 120.0],  # $10 -> $12 per unit
        'Invoice Date': ['2024-01-01', '2024-02-01']
    })
    
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    
    files = {"file": ("invoice_export.xlsx", excel_buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["flagged_items"] == 1
    assert result["records"][0]["price_change_pct"] == 20.0

def test_no_price_increases():
    """Test handling when no price increases are found"""
    csv_data = """Vendor,Item,Unit Price,Date
Acme,Widget,10.00,2024-01-01
Acme,Widget,10.00,2024-02-01
Global,Gadget,20.00,2024-01-01
Global,Gadget,19.00,2024-02-01"""
    
    files = {"file": ("no_increases.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["flagged_items"] == 0
    assert result["total_overage_amount"] == 0

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_missing_required_columns():
    """Test error when required columns are missing"""
    csv_data = """Column1,Column2,Column3
Value1,Value2,Value3"""
    
    files = {"file": ("bad.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400
    assert "Could not detect required columns" in response.json()["detail"]

def test_all_invalid_data():
    """Test error when all data is invalid"""
    csv_data = """Vendor,Item,Unit Price,Date
,,,
invalid,invalid,invalid,invalid"""
    
    files = {"file": ("invalid.csv", BytesIO(csv_data.encode()), "text/csv")}
    data = {"threshold": "5.0"}
    
    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
