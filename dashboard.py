import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torchvision.models as models
import os
import time
from gpu_predictor import GPUPredictor
from model_analyser import ModelAnalyzer

# Initialize components
@st.cache_resource
def get_predictor():
    return GPUPredictor()

@st.cache_resource
def get_analyzer():
    return ModelAnalyzer()

# Load data
@st.cache_data
def load_model_data():
    data_files = []
    for file in os.listdir('data/raw'):
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(f'data/raw/{file}')
                data_files.append(df)
            except:
                pass
    
    if data_files:
        return pd.concat(data_files, ignore_index=True)
    return pd.DataFrame()

# Create sample model
def create_sample_model(num_layers=3, channels=16):
    """Create a sample CNN model for testing"""
    class SampleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super(SampleCNN, self).__init__()
            layers = []
            in_channels = 3
            
            for i in range(num_layers):
                out_channels = channels * (2 ** i)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
                in_channels = out_channels
            
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(out_channels * (224 // (2**num_layers)) * (224 // (2**num_layers)), 10)
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return SampleCNN(num_layers, channels)

# Main app
def main():
    st.set_page_config(
        page_title="GPU Usage Prediction Dashboard", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    predictor = get_predictor()
    analyzer = get_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.title("GPU Usage Prediction")
        
        # Navigation
        page = st.radio("Navigation", [
            "Predict", 
            "Batch Size Optimizer", 
            "Model Comparison", 
            "Performance Monitor",
            "About"
        ])
        
        # System stats
        st.subheader("System Stats")
        cache_stats = predictor.get_cache_stats()
        st.metric("Prediction Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")
        
        # Settings
        st.subheader("Settings")
        memory_limit = st.slider("GPU Memory Limit (MB)", 1000, 32000, 8000)
    
    # Main content
    if page == "Predict":
        show_prediction_page(predictor, analyzer)
    elif page == "Batch Size Optimizer":
        show_batch_optimizer(predictor, analyzer, memory_limit)
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Performance Monitor":
        show_performance_monitor()
    else:
        show_about_page()

def show_prediction_page(predictor, analyzer):
    st.title("GPU Usage Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Custom CNN", "Pre-trained Models"]
        )
        
        if model_type == "Custom CNN":
            num_layers = st.slider("Number of Layers", 1, 10, 3)
            base_channels = st.slider("Base Channels", 8, 128, 16)
            
            # Create model description
            st.markdown(f"""
            **Model Architecture:**
            - Type: Custom CNN
            - Layers: {num_layers}
            - Base Channels: {base_channels}
            - Input Shape: (3, 224, 224)
            """)
            
            model = create_sample_model(num_layers, base_channels)
            
        else:
            model_name = st.selectbox(
                "Select Pre-trained Model",
                ["ResNet18", "ResNet50", "VGG16", "MobileNetV2", "DenseNet121"]
            )
            
            # Load selected model
            if model_name == "ResNet18":
                model = models.resnet18(weights=None)
            elif model_name == "ResNet50":
                model = models.resnet50(weights=None)
            elif model_name == "VGG16":
                model = models.vgg16(weights=None)
            elif model_name == "MobileNetV2":
                model = models.mobilenet_v2(weights=None)
            else:
                model = models.densenet121(weights=None)
        
        batch_sizes = st.multiselect(
            "Select Batch Sizes",
            [1, 2, 4, 8, 16, 32],
            default=[1, 2, 4]
        )
        
        if st.button("Predict GPU Usage"):
            with st.spinner("Analyzing model architecture..."):
                # Extract features
                start_time = time.time()
                features = analyzer.extract_features(model, (3, 224, 224))
                analysis_time = time.time() - start_time
                
                # Make predictions
                results = []
                start_time = time.time()
                
                # Prepare batch of features for prediction
                features_batch = []
                for bs in batch_sizes:
                    features_copy = features.copy()
                    features_copy['batch_size'] = bs
                    features_batch.append(features_copy)
                
                # Batch prediction
                predictions = predictor.predict(features_batch)
                prediction_time = time.time() - start_time
                
                for i, bs in enumerate(batch_sizes):
                    results.append({
                        "Batch Size": bs,
                        "Execution Time (ms)": predictions[i]
                    })
                
                # Create DataFrame for display
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Prediction Results")
                st.table(results_df)
                
                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results_df["Batch Size"],
                    y=results_df["Execution Time (ms)"],
                    text=results_df["Execution Time (ms)"].round(2),
                    textposition="auto"
                ))
                fig.update_layout(
                    title="Predicted Execution Time by Batch Size",
                    xaxis_title="Batch Size",
                    yaxis_title="Execution Time (ms)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display model details
                st.subheader("Model Details")
                st.write(f"Total Parameters: {features['total_parameters']:,}")
                st.write(f"Model Size: {features['model_size_mb']:.2f} MB")
                
                # Display performance metrics
                st.subheader("Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Analysis Time", f"{analysis_time*1000:.2f} ms")
                with col2:
                    st.metric("Prediction Time", f"{prediction_time*1000:.2f} ms")
    
    with col2:
        st.subheader("How It Works")
        st.info("""
        This tool predicts GPU execution time for neural network models without actually running them on GPU hardware.
        
        **Steps:**
        1. Select a model type
        2. Configure model parameters
        3. Select batch sizes
        4. Click "Predict GPU Usage"
        
        The system analyzes the model architecture and predicts execution times based on historical data from similar models.
        """)
        
        # Show architecture patterns
        if 'features' in locals():
            st.subheader("Architecture Patterns")
            patterns = features['architecture_patterns']
            
            pattern_data = {
                "Pattern": ["Skip Connections", "Attention Mechanism", "Normalization Layers", "Model Depth"],
                "Present": [
                    "✓" if patterns['has_skip_connections'] else "✗",
                    "✓" if patterns['has_attention'] else "✗",
                    "✓" if patterns['has_normalization'] else "✗",
                    str(patterns['max_depth'])
                ]
            }
            st.table(pd.DataFrame(pattern_data))

def show_batch_optimizer(predictor, analyzer, memory_limit):
    st.title("Batch Size Optimizer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Selection")
        
        model_type = st.selectbox(
            "Select Model Type",
            ["Custom CNN", "Pre-trained Models"],
            key="opt_model_type"
        )
        
        if model_type == "Custom CNN":
            num_layers = st.slider("Number of Layers", 1, 10, 3, key="opt_num_layers")
            base_channels = st.slider("Base Channels", 8, 128, 16, key="opt_base_channels")
            model = create_sample_model(num_layers, base_channels)
            model_name = f"Custom CNN ({num_layers} layers, {base_channels} channels)"
        else:
            model_name = st.selectbox(
                "Select Pre-trained Model",
                ["ResNet18", "ResNet50", "VGG16", "MobileNetV2", "DenseNet121"],
                key="opt_model_name"
            )
            
            # Load selected model
            if model_name == "ResNet18":
                model = models.resnet18(weights=None)
            elif model_name == "ResNet50":
                model = models.resnet50(weights=None)
            elif model_name == "VGG16":
                model = models.vgg16(weights=None)
            elif model_name == "MobileNetV2":
                model = models.mobilenet_v2(weights=None)
            else:
                model = models.densenet121(weights=None)
        
        min_batch = st.number_input("Minimum Batch Size", 1, 16, 1)
        max_batch = st.number_input("Maximum Batch Size", min_batch, 64, 32)
        
        if st.button("Find Optimal Batch Size"):
            with st.spinner("Analyzing model and finding optimal batch size..."):
                # Extract features
                features = analyzer.extract_features(model, (3, 224, 224))
                
                # Find optimal batch size
                optimization_result = predictor.optimize_batch_size(
                    features, 
                    min_batch=min_batch, 
                    max_batch=max_batch,
                    memory_limit_mb=memory_limit
                )
                
                # Display results
                st.subheader("Optimization Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Batch Size", optimization_result['optimal_batch_size'])
                with col2:
                    st.metric("Execution Time", f"{optimization_result['predicted_execution_time']:.2f} ms")
                with col3:
                    st.metric("Memory Usage", f"{optimization_result['estimated_memory_usage']:.2f} MB")
                
                # Create detailed results dataframe
                batch_results = pd.DataFrame(optimization_result['batch_results'])
                
                # Plot results
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Execution Time vs Batch Size", "Throughput vs Batch Size"),
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                # Add execution time trace
                fig.add_trace(
                    go.Scatter(
                        x=batch_results['batch_size'],
                        y=batch_results['exec_time_ms'],
                        mode='lines+markers',
                        name='Execution Time (ms)'
                    ),
                    row=1, col=1
                )
                
                # Add throughput trace
                fig.add_trace(
                    go.Scatter(
                        x=batch_results['batch_size'],
                        y=batch_results['throughput'],
                        mode='lines+markers',
                        name='Throughput (samples/s)'
                    ),
                    row=2, col=1
                )
                
                # Add optimal batch size marker
                optimal_batch = optimization_result['optimal_batch_size']
                fig.add_vline(
                    x=optimal_batch, 
                    line_dash="dash", 
                    line_color="green",
                    annotation_text=f"Optimal: {optimal_batch}",
                    annotation_position="top right",
                    row=1, col=1
                )
                
                fig.add_vline(
                    x=optimal_batch, 
                    line_dash="dash", 
                    line_color="green",
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    height=600,
                    title=f"Batch Size Optimization for {model_name}",
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Batch Size", row=2, col=1)
                fig.update_yaxes(title_text="Execution Time (ms)", row=1, col=1)
                fig.update_yaxes(title_text="Throughput (samples/s)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed results
                st.subheader("Detailed Results")
                st.dataframe(batch_results)
    
    with col2:
        st.subheader("Optimization Strategy")
        st.info("""
        The batch size optimizer finds the optimal batch size that maximizes throughput while staying within memory constraints.
        
        **How it works:**
        1. For each batch size in the specified range:
           - Estimate memory usage
           - Predict execution time
           - Calculate throughput (samples/second)
        
        2. Select the batch size with highest throughput that fits in memory
        
        **Memory Estimation:**
        - Base memory (model parameters)
        - Activation memory (scales with batch size)
        - Optimizer state memory
        
        The memory limit can be adjusted in the sidebar.
        """)

def show_model_comparison():
    st.title("Model Architecture Comparison")
    
    # Load historical data
    df = load_model_data()
    
    if df.empty:
        st.warning("No historical data found. Please check your data directory.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Execution Time", "Scaling Efficiency", "Parameter Efficiency"])
    
    with tab1:
        # Select models to compare
        available_models = sorted(df['model_name'].unique())
        selected_models = st.multiselect(
            "Select models to compare",
            available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models
        )
        
        if not selected_models:
            st.warning("Please select at least one model to display.")
            return
        
        # Filter data for selected models
        filtered_df = df[df['model_name'].isin(selected_models)]
        
        # Create interactive plot
        fig = go.Figure()
        
        for model in selected_models:
            model_data = filtered_df[filtered_df['model_name'] == model]
            fig.add_trace(go.Scatter(
                x=model_data['batch_size'],
                y=model_data['execution_time_ms'],
                mode='lines+markers',
                name=model
            ))
        
        fig.update_layout(
            title="Execution Time vs Batch Size",
            xaxis_title="Batch Size",
            yaxis_title="Execution Time (ms)",
            legend_title="Models",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Scaling efficiency analysis
        st.subheader("Scaling Efficiency")
        
        if not available_models:
            st.warning("No models available for analysis.")
            return
            
        # Calculate scaling efficiency
        pivot = df.pivot_table(
            index='model_name', 
            columns='batch_size', 
            values='execution_time_ms'
        )
        
        # Calculate relative scaling (normalized by batch size 1)
        scaling_data = []
        
        for model in pivot.index:
            base_time = pivot.loc[model, 1] if 1 in pivot.columns else None
            if base_time is None:
                continue
                
            for batch_size in [b for b in pivot.columns if b > 1]:
                if pd.isna(pivot.loc[model, batch_size]):
                    continue
                    
                exec_time = pivot.loc[model, batch_size]
                ideal_time = base_time * batch_size
                efficiency = base_time / (exec_time / batch_size)
                
                scaling_data.append({
                    'model_name': model,
                    'batch_size': batch_size,
                    'efficiency': efficiency
                })
        
        if not scaling_data:
            st.warning("Insufficient data for scaling efficiency analysis.")
            return
            
        scaling_df = pd.DataFrame(scaling_data)
        
        # Create plot
        fig = px.bar(
            scaling_df,
            x='model_name',
            y='efficiency',
            color='batch_size',
            title="Scaling Efficiency by Model (higher is better)",
            labels={
                'model_name': 'Model',
                'efficiency': 'Scaling Efficiency',
                'batch_size': 'Batch Size'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Scaling Efficiency** measures how well a model utilizes increased batch sizes.
        
        A value of 1.0 means perfect scaling (execution time increases linearly with batch size).
        Higher values indicate better parallelization and GPU utilization.
        Lower values indicate memory or compute bottlenecks.
        """)
    
    with tab3:
        # Parameter efficiency analysis
        st.subheader("Parameter Efficiency")
        
        # Get model parameters and execution time
        model_params = df.groupby('model_name')[['total_parameters', 'model_size_mb']].first()
        model_perf = df[df['batch_size'] == 1].groupby('model_name')['execution_time_ms'].first()
        
        if model_params.empty or model_perf.empty:
            st.warning("Insufficient data for parameter efficiency analysis.")
            return
            
        # Combine data
        efficiency_df = pd.DataFrame({
            'total_parameters': model_params['total_parameters'],
            'model_size_mb': model_params['model_size_mb'],
            'execution_time_ms': model_perf
        })
        
        efficiency_df['ms_per_million_params'] = efficiency_df['execution_time_ms'] / (efficiency_df['total_parameters'] / 1_000_000)
        
        # Create plot
        fig = px.bar(
            efficiency_df.reset_index(),
            x='model_name',
            y='ms_per_million_params',
            title="Execution Time per Million Parameters (lower is better)",
            labels={
                'model_name': 'Model',
                'ms_per_million_params': 'ms per Million Parameters'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(efficiency_df.reset_index())
        
        st.info("""
        **Parameter Efficiency** measures how efficiently a model uses its parameters.
        
        Lower values indicate better efficiency - less execution time per parameter.
        This metric helps identify models with good architecture design that maximizes parameter utilization.
        """)

def show_performance_monitor():
    st.title("Performance Monitor")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction Accuracy", "Cache Performance"])
    
    with tab1:
        st.subheader("Prediction Accuracy Analysis")
        
        # Load feedback data if available
        feedback_path = 'data/feedback_log.csv'
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
            
            if not feedback_df.empty:
                # Create accuracy plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=feedback_df['timestamp'],
                    y=feedback_df['error_percent'],
                    mode='lines+markers',
                    name='Prediction Error (%)'
                ))
                
                fig.update_layout(
                    title="Prediction Error Over Time",
                    xaxis_title="Timestamp",
                    yaxis_title="Error (%)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error distribution
                st.subheader("Error Distribution")
                
                fig = px.histogram(
                    feedback_df,
                    x='error_percent',
                    nbins=20,
                    title="Distribution of Prediction Errors"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.subheader("Recent Feedback Data")
                st.dataframe(feedback_df.tail(10))
            else:
                st.info("No feedback data available yet. As you use the system, prediction accuracy data will be collected here.")
        else:
            st.info("No feedback data available yet. As you use the system, prediction accuracy data will be collected here.")
    
    with tab2:
        st.subheader("Cache Performance")
        
        # Get cache stats
        predictor = get_predictor()
        cache_stats = predictor.get_cache_stats()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")
        with col2:
            st.metric("Cache Size", f"{cache_stats['cache_size']} / {cache_stats['max_cache_size']}")
        with col3:
            st.metric("Total Predictions", cache_stats['cache_hits'] + cache_stats['cache_misses'])
        
        # Create hit/miss chart
        hit_miss_data = pd.DataFrame([
            {'Category': 'Hits', 'Count': cache_stats['cache_hits']},
            {'Category': 'Misses', 'Count': cache_stats['cache_misses']}
        ])
        
        fig = px.pie(
            hit_miss_data,
            values='Count',
            names='Category',
            title="Cache Hits vs Misses",
            color='Category',
            color_discrete_map={'Hits': 'green', 'Misses': 'red'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        The prediction cache stores results of previous predictions to speed up repeated queries.
        
        A high hit rate indicates efficient use of the cache and faster predictions.
        The cache automatically manages its size to balance memory usage and performance.
        """)

def show_about_page():
    st.title("About GPU Usage Prediction System")
    
    st.markdown("""
    ## Overview
    
    This scalable GPU usage prediction system provides accurate estimates of execution time for deep learning models without requiring actual execution on GPU hardware. It uses machine learning to predict performance based on model architecture characteristics.
    
    ## Key Features
    
    - **Efficient Prediction**: Makes predictions in milliseconds using caching and batch processing
    - **Batch Size Optimization**: Finds optimal batch sizes to maximize throughput within memory constraints
    - **Model Comparison**: Analyzes scaling efficiency and parameter utilization across different architectures
    - **Performance Monitoring**: Tracks prediction accuracy and system performance over time
    
    ## How It Works
    
    1. **Model Analysis**: Extracts features from neural network architectures
    2. **Prediction**: Uses gradient boosting to predict execution times based on model features
    3. **Optimization**: Recommends optimal configurations for maximum performance
    4. **Monitoring**: Continuously improves through feedback and performance tracking
    
    ## Dataset
    
    The system was trained on data from diverse model architectures:
    
    - Simple CNNs (525K parameters)
    - Complex models like VGG16 (138M parameters)
    - Transformer models like RoBERTa-base (124M parameters)
    
    ## Performance
    
    - Prediction time: < 50ms
    - Accuracy: < 6% error for most models
    - Scalability: Handles batch prediction and parallel processing
    """)

if __name__ == "__main__":
    import plotly.express as px  # Import here to avoid circular import
    main()
