class BatchSizeOptimizer:
    def __init__(self, prediction_model, memory_limit_mb=8000):
        self.prediction_model = prediction_model
        self.memory_limit_mb = memory_limit_mb
        
    def estimate_memory_usage(self, model_features, batch_size):
        """Estimate memory usage for a given batch size"""
        # Base memory is the model size
        base_memory = model_features['model_size_mb']
        
        # Additional memory for activations (simplified model)
        # This is a heuristic based on observed data patterns
        if model_features['total_parameters'] > 100000000:  # Large models like VGG16
            # Large models have higher activation memory requirements
            activation_memory = base_memory * 0.5 * batch_size
        else:  # Smaller models
            activation_memory = base_memory * 0.3 * batch_size
            
        # Memory for optimizer states (typically 2x model size for Adam)
        optimizer_memory = base_memory * 2
        
        # Total estimated memory
        total_memory = base_memory + activation_memory + optimizer_memory
        
        return total_memory
    
    def find_optimal_batch_size(self, model_features, min_batch=1, max_batch=32):
        """Find optimal batch size that maximizes throughput within memory constraints"""
        valid_batch_sizes = []
        throughputs = []
        
        for batch_size in range(min_batch, max_batch + 1):
            # Check memory constraint
            estimated_memory = self.estimate_memory_usage(model_features, batch_size)
            if estimated_memory > self.memory_limit_mb:
                continue
                
            # Create features for this batch size
            features = model_features.copy()
            features['batch_size'] = batch_size
            
            # Predict execution time
            exec_time = self.prediction_model.predict(features)
            
            # Calculate throughput (samples per second)
            throughput = (batch_size * 1000) / exec_time
            
            valid_batch_sizes.append(batch_size)
            throughputs.append(throughput)
            
        if not valid_batch_sizes:
            return min_batch  # Default to minimum if no valid batch size
            
        # Find batch size with maximum throughput
        optimal_idx = throughputs.index(max(throughputs))
        optimal_batch_size = valid_batch_sizes[optimal_idx]
        
        return optimal_batch_size
