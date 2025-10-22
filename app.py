import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== DATA CLEANING FUNCTIONS ====================

def format_row_list(indices, max_display=10):
    """Format row indices for display, showing ranges for large lists."""
    if len(indices) == 0:
        return "None"
    
    # Indices are already 1-based from collection
    indices = sorted(indices)
    
    if len(indices) > max_display:
        # Show ranges for large lists
        ranges = []
        start = indices[0]
        end = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")
                start = indices[i]
                end = indices[i]
        
        # Add final range
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
        
        # If too many ranges, summarize
        if len(ranges) > 5:
            return f"rows {ranges[0]}, {ranges[1]}, ..., {ranges[-1]} ({len(indices)} total)"
        else:
            return f"rows {', '.join(ranges)}"
    else:
        return f"rows {', '.join(map(str, indices))}"

@st.cache_data
def load_and_clean_data(file_path):
    """Load CSV and perform comprehensive data cleaning with detailed row tracking."""
    cleaning_log = []
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        initial_rows = len(df)
        cleaning_log.append(f"✓ Loaded data: {initial_rows} rows, {len(df.columns)} columns")
        
        # 1. Convert Date to datetime
        try:
            date_errors = [i + 1 for i in df[pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce').isna()].index.tolist()]
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            
            if len(date_errors) > 0:
                cleaning_log.append(f"✓ Date conversion: Successful, with errors in {format_row_list(date_errors)}")
            else:
                cleaning_log.append(f"✓ Date conversion: Successful (all {initial_rows} rows)")
        except Exception as e:
            cleaning_log.append(f"⚠ Date conversion error: {e}")
        
        # 2. Handle duplicates based on Date
        duplicate_indices = [i + 1 for i in df[df.duplicated(subset=['Date'], keep='first')].index.tolist()]
        duplicates = len(duplicate_indices)
        if duplicates > 0:
            df = df.drop_duplicates(subset=['Date'], keep='first')
            cleaning_log.append(f"✓ Removed {duplicates} duplicate dates: {format_row_list(duplicate_indices)}")
        else:
            cleaning_log.append("✓ No duplicates found")
        
        # 3. Clean numerical columns - Track conversions
        numeric_cols = ['Price', 'Open', 'High', 'Low']
        for col in numeric_cols:
            if col in df.columns:
                original_values = df[col].copy()
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Track which rows had commas/special chars
                affected = [i + 1 for i in df[original_values.astype(str).str.contains(',|%', na=False)].index.tolist()]
                if len(affected) > 0:
                    cleaning_log.append(f"✓ {col}: Cleaned {len(affected)} rows - {format_row_list(affected)}")
        
        if 'Change %' in df.columns:
            original_values = df['Change %'].copy()
            df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
            df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
            affected = [i + 1 for i in df[original_values.astype(str).str.contains('%', na=False)].index.tolist()]
            if len(affected) > 0:
                cleaning_log.append(f"✓ Change %: Cleaned {len(affected)} rows - {format_row_list(affected)}")
        
        if 'Vol.' in df.columns:
            original_values = df['Vol.'].copy()
            df['Vol.'] = df['Vol.'].astype(str).str.strip()
            
            # Track volume conversions
            k_indices = [i + 1 for i in df[original_values.astype(str).str.contains('k|K', na=False)].index.tolist()]
            m_indices = [i + 1 for i in df[original_values.astype(str).str.contains('m|M', na=False)].index.tolist()]
            
            # Convert volume strings like '107.50k' to numeric values
            def convert_volume(vol_str):
                vol_str = str(vol_str).strip().upper()
                if 'K' in vol_str:
                    return float(vol_str.replace('K', '')) * 1000
                elif 'M' in vol_str:
                    return float(vol_str.replace('M', '')) * 1000000
                else:
                    return float(vol_str)
            
            df['Vol.'] = df['Vol.'].apply(lambda x: convert_volume(x) if pd.notna(x) else np.nan)
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce')
            
            if len(k_indices) > 0:
                cleaning_log.append(f"✓ Vol.: Converted 'K' suffix ({len(k_indices)} rows) - {format_row_list(k_indices)}")
            if len(m_indices) > 0:
                cleaning_log.append(f"✓ Vol.: Converted 'M' suffix ({len(m_indices)} rows) - {format_row_list(m_indices)}")
        
        # 4. Handle missing values
        missing_by_col = {}
        for col in df.columns:
            missing_idx = [i + 1 for i in df[df[col].isna()].index.tolist()]
            if len(missing_idx) > 0:
                missing_by_col[col] = missing_idx
        
        if missing_by_col:
            total_missing = sum(len(v) for v in missing_by_col.values())
            cleaning_log.append(f"⚠ Missing values detected: {total_missing} total")
            
            for col, indices in missing_by_col.items():
                cleaning_log.append(f"  • {col}: {format_row_list(indices)}")
            
            # Forward fill for time-series continuity
            df = df.fillna(method='ffill').fillna(method='bfill')
            missing_after = df.isnull().sum().sum()
            cleaning_log.append(f"✓ Forward-filled missing values: {missing_after} remaining")
        else:
            cleaning_log.append("✓ No missing values found")
        
        # 5. Outlier detection (IQR method) - Flag but don't remove
        outlier_info = {}
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = [i + 1 for i in df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()]
                outliers = len(outlier_indices)
                outlier_info[col] = outliers
                
                if outliers > 0:
                    cleaning_log.append(f"✓ {col} outliers: {outliers} flagged - {format_row_list(outlier_indices)} (retained)")
        
        total_outliers = sum(outlier_info.values())
        if total_outliers == 0:
            cleaning_log.append(f"✓ Outlier detection: No outliers detected using IQR method")
        
        # 6. Remove impossible values (negative prices)
        negative_indices = [i + 1 for i in df[df['Price'] < 0].index.tolist()]
        negative_prices = len(negative_indices)
        if negative_prices > 0:
            df = df[df['Price'] >= 0]
            cleaning_log.append(f"✓ Removed {negative_prices} negative price records: {format_row_list(negative_indices)}")
        else:
            cleaning_log.append("✓ No negative prices found")
        
        # Reset index to maintain original row order
        df = df.reset_index(drop=True)
        
        final_rows = len(df)
        cleaning_log.append(f"✓ Final dataset: {final_rows} rows (removed {initial_rows - final_rows} rows)")
        
        return df, cleaning_log, outlier_info
    
    except Exception as e:
        cleaning_log.append(f"✗ Error during cleaning: {str(e)}")
        logger.error(f"Data cleaning error: {e}")
        return None, cleaning_log, {}

# ==================== EDA FUNCTIONS ====================

def create_trend_chart(df):
    """Monthly trend line chart."""
    monthly_data = df.set_index('Date').resample('ME')['Price'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data['Date'],
        y=monthly_data['Price'],
        mode='lines',
        name='Monthly Avg Price',
        line=dict(color='#FFD700', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Avg Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Gold Price Trend (Monthly Averages, 2013-2023)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    return fig

def create_yearly_boxplot(df):
    """Yearly price distribution boxplot."""
    df['Year'] = df['Date'].dt.year
    
    fig = go.Figure()
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        fig.add_trace(go.Box(
            y=year_data['Price'],
            name=str(year),
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Price Distribution by Year",
        yaxis_title="Price ($)",
        xaxis_title="Year",
        template='plotly_dark',
        height=400,
        hovermode='closest'
    )
    return fig

def create_seasonality_heatmap(df):
    """Monthly seasonality heatmap."""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    pivot_data = df.pivot_table(values='Price', index='Month', columns='Year', aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        colorscale='YlOrRd',
        hovertemplate='<b>Year:</b> %{x}<br><b>Month:</b> %{y}<br><b>Avg Price:</b> $%{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monthly Price Seasonality (Average Price by Month & Year)",
        xaxis_title="Year",
        yaxis_title="Month",
        template='plotly_dark',
        height=500
    )
    return fig

def create_volatility_chart(df):
    """Monthly volatility (High - Low range) chart."""
    df_sorted = df.sort_values('Date')
    monthly_vol = df_sorted.set_index('Date').resample('ME').apply({
        'High': 'mean',
        'Low': 'mean'
    }).reset_index()
    monthly_vol['Range'] = monthly_vol['High'] - monthly_vol['Low']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_vol['Date'],
        y=monthly_vol['Range'],
        name='Price Range',
        marker=dict(color='#FF6B6B'),
        hovertemplate='<b>Month:</b> %{x|%Y-%m}<br><b>Range:</b> $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monthly Price Volatility (High - Low Range)",
        xaxis_title="Date",
        yaxis_title="Price Range ($)",
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    return fig

def create_volume_price_chart(df):
    """Dual-axis chart: Price line + Volume bars."""
    monthly_data = df.set_index('Date').resample('ME').agg({
        'Price': 'mean',
        'Vol.': 'sum'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Date'],
            y=monthly_data['Price'],
            name='Avg Price',
            line=dict(color='#FFD700', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Avg Price:</b> $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=monthly_data['Date'],
            y=monthly_data['Vol.'],
            name='Total Volume',
            marker=dict(color='rgba(100, 149, 237, 0.5)'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    fig.update_layout(
        title="Gold Price vs Trading Volume",
        template='plotly_dark',
        height=400,
        hovermode='x unified'
    )
    return fig

def create_correlation_heatmap(df):
    """Correlation matrix heatmap."""
    numeric_cols = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Matrix of Numerical Features",
        template='plotly_dark',
        height=500
    )
    return fig

# ==================== PREDICTION FUNCTIONS ====================

def prepare_monthly_data(df):
    """Resample to monthly averages and format for Prophet."""
    monthly_df = df.set_index('Date').resample('ME')['Price'].mean().reset_index()
    # Prophet requires columns 'ds' (datestamp) and 'y' (value)
    monthly_df.columns = ['ds', 'y']
    return monthly_df

def train_prophet_model(monthly_df):
    """Train Prophet model and get in-sample performance."""
    
    # Initialize Prophet. Multiplicative seasonality is often better for financial data.
    model = Prophet(interval_width=0.95, seasonality_mode='multiplicative')
    
    # Fit the model
    model.fit(monthly_df)
    
    # Make predictions on training data to get performance metrics
    in_sample_df = model.predict(monthly_df)
    
    y_true = monthly_df['y']
    y_pred = in_sample_df['yhat']
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def forecast_prophet_model(model, months_ahead=60):
    """Generate future forecast using Prophet."""
    
    # Create a dataframe for future dates
    future_df = model.make_future_dataframe(periods=months_ahead, freq='ME')
    
    # Generate forecast
    # This df will contain historical fit (yhat) and future forecast
    forecast_df = model.predict(future_df)
    
    return forecast_df

def create_forecast_chart(historical_df, forecast_df):
    """Create interactive forecast visualization using Prophet output."""
    fig = go.Figure()
    
    # Historical data (from the original 'y' values)
    fig.add_trace(go.Scatter(
        x=historical_df['ds'],
        y=historical_df['y'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#FFD700', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Forecast (Prophet's 'yhat')
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast (yhat)',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Forecast:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Confidence bands (Prophet's 'yhat_upper' and 'yhat_lower')
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(255, 107, 107, 0)',
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255, 107, 107, 0)',
        name='95% Confidence Interval',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hovertemplate='<b>Date:</b> %{x|%Y-%m}<br><b>Lower:</b> $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Gold Price Forecast: 2023-2027 (Prophet Model with 95% Confidence Interval)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    return fig

def create_prophet_components_chart(model, forecast_df):
    """Create Prophet's built-in components plot."""
    # Use Prophet's built-in plotly function
    fig = plot_components_plotly(model, forecast_df)
    # Update layout to match the dark theme
    fig.update_layout(template='plotly_dark')
    return fig


# ==================== STREAMLIT APP ====================

st.set_page_config(page_title="Gold Price Analysis & Prediction", layout="wide")

st.title("🏆 Gold Price Analysis & Prediction Dashboard")
st.markdown("10-Year Analysis (2013-2023) with Forecasts through 2027")

# Load and clean data from default file
file_path = 'gold_price_2013-2023.csv'
df, cleaning_log, outlier_info = load_and_clean_data(file_path)

if df is not None:
    
    # ==================== SECTION 1: DATA OVERVIEW ====================
    st.header("1. Data Overview & Cleaning Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    col3.metric("Years Covered", f"{df['Date'].dt.year.max() - df['Date'].dt.year.min() + 1}")
    col4.metric("Price Range", f"${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
    
    st.subheader("Cleaning Log")
    for log_entry in cleaning_log:
        st.write(log_entry)
    
    st.subheader("Before & After Summary Statistics")
    st.write(df[['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']].describe())
    
    st.subheader("📥 Download Cleaned Dataset")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data (CSV)",
        data=csv_data,
        file_name="gold_price_2013-2023.csv",
        mime="text/csv"
    )
    st.success("✓ Click the button above to download the cleaned dataset for future use or external analysis")
    
    # ==================== SECTION 2: EDA ====================
    st.header("2. Exploratory Data Analysis")
    
    st.subheader("📈 Price Trend Analysis")
    st.plotly_chart(create_trend_chart(df), use_container_width=True)
    
    st.subheader("📊 Yearly Price Distribution")
    st.plotly_chart(create_yearly_boxplot(df), use_container_width=True)
    
    st.subheader("🗓️ Monthly Seasonality Heatmap")
    st.plotly_chart(create_seasonality_heatmap(df), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Price Volatility (Monthly Range)")
        st.plotly_chart(create_volatility_chart(df), use_container_width=True)
    
    with col2:
        st.subheader("🔄 Volume vs Price")
        st.plotly_chart(create_volume_price_chart(df), use_container_width=True)
    
    st.subheader("🔗 Correlation Analysis")
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    
    # ==================== SECTION 3: PREDICTION (PROPHET) ====================
    st.header("3. Prophet Prediction Model")
    
    # Prepare data (renames to 'ds' and 'y')
    monthly_df = prepare_monthly_data(df)
    
    # Train model and get in-sample metrics
    model, metrics = train_prophet_model(monthly_df)
    
    st.subheader("Model Performance (In-Sample)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:.2f}")
    col3.metric("R² Score", f"{metrics['R2']:.4f}")
    
    # Generate forecast (includes historical fit + future)
    forecast_df = forecast_prophet_model(model, months_ahead=60)
    
    st.subheader("📊 Forecast Visualization (2023-2027)")
    st.plotly_chart(create_forecast_chart(monthly_df, forecast_df), use_container_width=True)
    
    st.subheader("📊 Model Components (Trend & Seasonality)")
    st.plotly_chart(create_prophet_components_chart(model, forecast_df), use_container_width=True)

    st.subheader("📈 Forecast Data (Sample)")
    # Show only future predictions, not the in-sample fit
    future_only_df = forecast_df[forecast_df['ds'] > monthly_df['ds'].max()]
    
    # Rename columns for clarity in the table
    display_df = future_only_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    display_df.columns = ['Date', 'Forecast (yhat)', 'Lower_Band_95%', 'Upper_Band_95%']
    
    st.dataframe(
        display_df.head(12),
        use_container_width=True
    )
    
    st.info("ℹ️ The model uses Facebook's Prophet time-series algorithm on monthly average prices. Uncertainty bands represent the 95% confidence interval.")
else:
    st.error("❌ Error: Could not load or process 'gold_price_2013-2023.csv'. Please ensure the file exists in the app directory.")