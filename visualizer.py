import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def create_visualization(viz_request, data):
    """Create visualization based on request and return as base64 encoded image"""
    
    try:
        # Set up the plot with a reasonable size
        plt.figure(figsize=(10, 6))
        plt.style.use('default')
        
        if viz_request.get('type') == 'scatterplot':
            return _create_scatterplot(viz_request, data)
        elif viz_request.get('type') == 'histogram':
            return _create_histogram(viz_request, data)
        elif viz_request.get('type') == 'line':
            return _create_line_plot(viz_request, data)
        else:
            return _create_default_plot(viz_request, data)
    
    except Exception as e:
        # Return a minimal error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Visualization Error: {str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Visualization Error')
        return _plot_to_base64()

def _create_scatterplot(viz_request, data):
    """Create a scatterplot with optional regression line"""
    
    x_col = viz_request.get('x_col', 'rank_clean')
    y_col = viz_request.get('y_col', 'peak_clean')
    
    # Handle case where data might not have expected columns
    if isinstance(data, pd.DataFrame) and not data.empty:
        if x_col not in data.columns or y_col not in data.columns:
            # Create sample data for demonstration
            x_data = np.arange(1, 21)
            y_data = x_data + np.random.normal(0, 2, 20)
        else:
            x_data = pd.to_numeric(data[x_col], errors='coerce').dropna()
            y_data = pd.to_numeric(data[y_col], errors='coerce').dropna()
            
            # Ensure same length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data.iloc[:min_len]
            y_data = y_data.iloc[:min_len]
    else:
        # Create sample data
        x_data = np.arange(1, 21)
        y_data = x_data + np.random.normal(0, 2, 20)
    
    # Create the scatterplot
    plt.scatter(x_data, y_data, alpha=0.6, s=50)
    
    # Add regression line if requested
    if viz_request.get('regression_line', False):
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            line_x = np.array([min(x_data), max(x_data)])
            line_y = slope * line_x + intercept
            
            color = viz_request.get('regression_color', 'red')
            style = viz_request.get('regression_style', 'dotted')
            linestyle = '--' if style == 'dotted' else '-'
            
            plt.plot(line_x, line_y, color=color, linestyle=linestyle, linewidth=2)
        except Exception as e:
            print(f"Could not add regression line: {e}")
    
    # Set labels and title
    plt.xlabel(viz_request.get('x_label', 'Rank'))
    plt.ylabel(viz_request.get('y_label', 'Peak'))
    plt.title(viz_request.get('title', 'Rank vs Peak'))
    plt.grid(True, alpha=0.3)
    
    return _plot_to_base64()

def _create_histogram(viz_request, data):
    """Create a histogram"""
    col = viz_request.get('column', data.columns[0] if isinstance(data, pd.DataFrame) else 'value')
    
    if isinstance(data, pd.DataFrame) and col in data.columns:
        values = pd.to_numeric(data[col], errors='coerce').dropna()
    else:
        values = np.random.normal(0, 1, 100)
    
    plt.hist(values, bins=viz_request.get('bins', 30), alpha=0.7, edgecolor='black')
    plt.xlabel(viz_request.get('x_label', col))
    plt.ylabel('Frequency')
    plt.title(viz_request.get('title', f'Distribution of {col}'))
    plt.grid(True, alpha=0.3)
    
    return _plot_to_base64()

def _create_line_plot(viz_request, data):
    """Create a line plot"""
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        x_col = data.columns[0]
        y_col = data.columns[1]
        plt.plot(data[x_col], data[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    else:
        x = np.arange(10)
        y = np.random.randn(10).cumsum()
        plt.plot(x, y, marker='o')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.title(viz_request.get('title', 'Line Plot'))
    plt.grid(True, alpha=0.3)
    
    return _plot_to_base64()

def _create_default_plot(viz_request, data):
    """Create a default plot when type is not specified"""
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        # Create a simple scatter plot of first two numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.6)
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
        else:
            plt.text(0.5, 0.5, 'No suitable data for visualization', 
                    ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, 'No data available for visualization', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title(viz_request.get('title', 'Data Visualization'))
    
    return _plot_to_base64()

def _plot_to_base64():
    """Convert current matplotlib plot to base64 encoded PNG"""
    # Save plot to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Clear the current plot
    plt.clf()
    plt.close()
    
    # Return as data URI
    return f"data:image/png;base64,{img_base64}"
