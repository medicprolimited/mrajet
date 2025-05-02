# save metrics to training_run

import sys
print("Script starting", file=sys.stderr)
import os
import torch
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend first
import matplotlib.pyplot as plt  # Then import pyplot
import json
from collections import Counter

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'fine_tuning.log'), mode='a')
    ]
)

logger = logging.getLogger('fine_tune')

class LearningCurveCallback:
    """Enhanced callback to track and visualize metrics during training"""
    def __init__(self):
        self.loss_values = []
        self.steps = []
        self.epochs = []
        self.current_step = 0
        self.current_epoch = 0
        
        # Storage for evaluation metrics per epoch
        self.eval_metrics = {
            'cosine_pearson': [],
            'cosine_spearman': [],
            'euclidean_pearson': [],
            'manhattan_pearson': [],
            'dot_pearson': [],
        }
        
        # Create visualizations directory if it doesn't exist
        self.vis_dir = 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('fine_tune')
        self.logger.info("Learning Curve Callback initialized")
        
        # Generate timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def on_epoch_end(self, epoch_num, steps, loss_value, validation_results=None):
        """Called after each training epoch with evaluation results"""
        self.current_epoch = epoch_num
        self.epochs.append(epoch_num)
        
        # Track the average loss for this epoch
        self.loss_values.append(loss_value)
        
        # Add validation metrics if available
        if validation_results:
            for metric_name, metric_value in validation_results.items():
                if metric_name in self.eval_metrics:
                    self.eval_metrics[metric_name].append(metric_value)
        
        # Log progress
        self.logger.info(f"Epoch {epoch_num}: Loss = {loss_value:.6f}")
        if validation_results:
            self.logger.info(f"Validation metrics: {validation_results}")
        
        # Generate visualizations after each epoch
        if epoch_num > 0:  # Wait until we have at least 2 data points
            self._generate_visualizations()
    
    def on_batch_end(self, progress_info):
        """Called after each training batch - for step-level tracking"""
        if hasattr(progress_info, 'loss_value'):
            self.current_step += 1
            if self.current_step % 10 == 0:  # Log every 10 steps
                self.steps.append(self.current_step)
                self.logger.debug(f"Step {self.current_step}: Loss = {progress_info.loss_value:.6f}")
    
    def _generate_visualizations(self):
        """Generate visualizations for the current training state"""
        self.logger.info("Generating learning curve visualizations...")
        
        # Create figure directory with timestamp if this is the first visualization
        vis_path = os.path.join(self.vis_dir, f'training_run_{self.timestamp}')
        os.makedirs(vis_path, exist_ok=True)
        
        # 1. Training Loss Curve
        self._plot_training_loss(vis_path)
        
        # 2. Validation Metrics Curves
        self._plot_validation_metrics(vis_path)
        
        # 3. Combined Training and Validation Plot
        self._plot_combined_metrics(vis_path)
        
        self.logger.info(f"Visualizations saved to {vis_path}")
        
    def _plot_training_loss(self, vis_path):
        """Plot training loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.loss_values, 'b-', linewidth=2, marker='o', label='Training Loss')
        plt.title('Training Loss Over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add text annotation for final loss value
        if len(self.loss_values) > 0:
            plt.annotate(f'Final Loss: {self.loss_values[-1]:.4f}', 
                         xy=(self.epochs[-1], self.loss_values[-1]),
                         xytext=(self.epochs[-1] - 0.5, self.loss_values[-1] + 0.1),
                         fontsize=10, arrowprops=dict(arrowstyle='->'))
        
        # Save as PNG and PDF
        loss_plot_path = os.path.join(vis_path, f'training_loss_ep{self.current_epoch}')
        plt.savefig(f'{loss_plot_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{loss_plot_path}.pdf', bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training loss plot saved to {loss_plot_path}.png/pdf")
    
    def _plot_validation_metrics(self, vis_path):
        """Plot validation metrics over epochs"""
        if not any(len(values) > 0 for values in self.eval_metrics.values()):
            self.logger.warning("No validation metrics available for plotting")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot each validation metric with a different color
        colors = ['b', 'g', 'r', 'c', 'm']
        for (metric_name, values), color in zip(self.eval_metrics.items(), colors):
            if len(values) > 0:
                plt.plot(self.epochs[:len(values)], values, f'{color}-', 
                         linewidth=2, marker='o', label=f'{metric_name.replace("_", " ").title()}')
        
        plt.title('Validation Metrics Over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best')
        
        # Set y-axis limits for better visualization of correlation scores
        plt.ylim(-0.1, 1.1)
        
        # Save as PNG and PDF
        val_plot_path = os.path.join(vis_path, f'validation_metrics_ep{self.current_epoch}')
        plt.savefig(f'{val_plot_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{val_plot_path}.pdf', bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Validation metrics plot saved to {val_plot_path}.png/pdf")
    
    def _plot_combined_metrics(self, vis_path):
        """Create a combined plot showing both training and validation metrics"""
        if not any(len(values) > 0 for values in self.eval_metrics.values()):
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot training loss on primary y-axis
        ax1 = plt.gca()
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Training Loss', fontsize=14, color='b')
        ax1.plot(self.epochs, self.loss_values, 'b-', linewidth=2, marker='o', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot validation metrics on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Score', fontsize=14, color='r')
        
        # Choose the primary validation metric (cosine_pearson)
        if 'cosine_pearson' in self.eval_metrics and len(self.eval_metrics['cosine_pearson']) > 0:
            values = self.eval_metrics['cosine_pearson']
            ax2.plot(self.epochs[:len(values)], values, 'r-', 
                     linewidth=2, marker='s', label='Cosine Pearson (Validation)')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Set y-axis limits for better visualization of correlation scores
            ax2.set_ylim(-0.1, 1.1)
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=12)
        
        plt.title('Training Loss and Validation Performance', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save as PNG and PDF
        combined_plot_path = os.path.join(vis_path, f'combined_metrics_ep{self.current_epoch}')
        plt.savefig(f'{combined_plot_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{combined_plot_path}.pdf', bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Combined metrics plot saved to {combined_plot_path}.png/pdf")
    
    def save_metrics_to_json(self, final_test_metrics=None):
        """Save all tracked metrics to a JSON file for later analysis"""
        metrics_data = {
            'training': {
                'epochs': self.epochs,
                'loss_values': self.loss_values
            },
            'validation': {metric: values for metric, values in self.eval_metrics.items() if values}
        }
        
        if final_test_metrics:
            metrics_data['test'] = final_test_metrics
        
        # Save to JSON file
        vis_path = os.path.join(self.vis_dir, f'training_run_{self.timestamp}')
        os.makedirs(vis_path, exist_ok=True)
        metrics_file = os.path.join(vis_path, 'training_metrics.json')
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Training metrics saved to {metrics_file}")
        
        return metrics_file

# Original LossCallback, keeping for backward compatibility
class LossCallback:
    """Simple callback to track loss during training"""
    def __init__(self):
        self.loss_values = []
        self.steps = []
        self.current_step = 0
        
    def on_batch_end(self, progress_info):
        """Called after each training batch"""
        if hasattr(progress_info, 'loss_value'):
            self.current_step += 1
            if self.current_step % 10 == 0:  # Log every 10 steps
                self.loss_values.append(progress_info.loss_value)
                self.steps.append(self.current_step)
                logger.debug(f"Step {self.current_step}: Loss = {progress_info.loss_value:.6f}")

def extract_validation_results(evaluator_path, epoch):
    """
    Extract validation results from the EmbeddingSimilarityEvaluator output files
    
    Parameters:
    evaluator_path (str): Path to the directory containing evaluation results
    epoch (int): Current epoch number
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    # Path to the evaluation results file
    results_file = os.path.join(evaluator_path, f'vaping-validation_results.csv')
    
    # Default empty results
    results = {}
    
    # Check if the file exists
    if not os.path.exists(results_file):
        logging.getLogger('fine_tune').warning(f"Evaluation results file not found: {results_file}")
        return results
    
    try:
        # Read the CSV file
        df = pd.read_csv(results_file)
        
        # Get the row for the current epoch (if available)
        if len(df) >= epoch:
            row = df.iloc[epoch-1]  # Adjust for zero-indexed DataFrame
            
            # Extract metrics
            results = {
                'cosine_pearson': row.get('cosine_pearson', 0.0),
                'cosine_spearman': row.get('cosine_spearman', 0.0),
                'euclidean_pearson': row.get('euclidean_pearson', 0.0),
                'manhattan_pearson': row.get('manhattan_pearson', 0.0),
                'dot_pearson': row.get('dot_pearson', 0.0),
            }
    except Exception as e:
        logging.getLogger('fine_tune').error(f"Error extracting validation results: {str(e)}")
    
    return results

def create_visualization_summary(lcv_callback, final_test_metrics, timestamp):
    """
    Create a final visualization summary report
    
    Parameters:
    lcv_callback (LearningCurveCallback): The callback with all tracked metrics
    final_test_metrics (dict): Dictionary of final test metrics
    timestamp (str): Timestamp for the training run
    
    Returns:
    str: Path to the saved summary report
    """
    logger = logging.getLogger('fine_tune')
    logger.info("Creating visualization summary report...")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Model Training and Evaluation Summary', fontsize=16)
    
    # 1. Training Loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(lcv_callback.epochs, lcv_callback.loss_values, 'b-', linewidth=2, marker='o')
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Validation Metrics
    ax2 = plt.subplot(2, 2, 2)
    
    # Plot each validation metric with a different color
    colors = ['b', 'g', 'r', 'c', 'm']
    for (metric_name, values), color in zip(lcv_callback.eval_metrics.items(), colors):
        if len(values) > 0:
            ax2.plot(lcv_callback.epochs[:len(values)], values, f'{color}-', 
                     linewidth=2, marker='o', label=f'{metric_name.split("_")[0]}')
    
    ax2.set_title('Validation Metrics', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='best')
    ax2.set_ylim(-0.1, 1.1)
    
    # 3. Test Metrics Bar Chart
    ax3 = plt.subplot(2, 2, 3)
    
    # Prepare metrics for bar chart
    test_metrics = ['accuracy', 'precision', 'recall', 'f1']
    test_values = [final_test_metrics.get(metric, 0) for metric in test_metrics]
    
    bars = ax3.bar(test_metrics, test_values, color=['blue', 'green', 'red', 'purple'])
    ax3.set_title('Final Test Metrics', fontsize=14)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Training Time and Info
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')  # Turn off axis
    
    # Prepare information text
    info_text = [
        f"Training Summary",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Epochs: {lcv_callback.current_epoch}",
        f"Final Training Loss: {lcv_callback.loss_values[-1]:.4f}",
        f"Evaluation Time: {final_test_metrics.get('evaluation_time', 0):.2f} seconds",
        f"\nModel Performance:",
        f"Accuracy: {final_test_metrics.get('accuracy', 0):.4f}",
        f"Precision: {final_test_metrics.get('precision', 0):.4f}",
        f"Recall: {final_test_metrics.get('recall', 0):.4f}",
        f"F1 Score: {final_test_metrics.get('f1', 0):.4f}"
    ]
    
    ax4.text(0.1, 0.9, '\n'.join(info_text), fontsize=12, va='top')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Save the summary report
    vis_path = os.path.join('visualizations', f'training_run_{timestamp}')
    os.makedirs(vis_path, exist_ok=True)
    summary_path = os.path.join(vis_path, 'training_summary')
    
    # Save as PNG and PDF
    plt.savefig(f'{summary_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{summary_path}.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization summary report saved to {summary_path}.png/pdf")
    
    return summary_path

# Modified EmbeddingSimilarityEvaluator class to track training progress
class EmbeddingSimilarityEvaluatorWithCallback(EmbeddingSimilarityEvaluator):
    """Extension of EmbeddingSimilarityEvaluator that works with our callback"""
    
    def __init__(self, lcv_callback, samples, name=''):
        # Extract sentences and scores from the input examples
        sentences1 = []
        sentences2 = []
        scores = []
        
        for example in samples:
            if len(example.texts) >= 2:
                sentences1.append(example.texts[0])
                sentences2.append(example.texts[1])
                scores.append(example.label)
        
        # Call the parent class constructor with the correct parameters
        super().__init__(sentences1, sentences2, scores, name=name)
        
        self.lcv_callback = lcv_callback
        self.current_epoch = 0
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs):
        """
        Override the __call__ method to also update our callback
        """
        # Increment epoch counter
        self.current_epoch += 1
        
        # Call the parent class evaluation
        results = super().__call__(model, output_path, epoch, steps, *args, **kwargs)
        
        # Extract current loss from model if available
        current_loss = 0.0
        if hasattr(model, 'best_loss') and model.best_loss is not None:
            current_loss = model.best_loss
        
        # The evaluator returns a single float, not a dictionary
        # Create a dictionary with the result value
        validation_results = {
            'cosine_pearson': float(results)  # Convert numpy.float64 to Python float
        }
        
        # Update our LCV callback with the results
        self.lcv_callback.on_epoch_end(
            epoch_num=self.current_epoch,
            steps=steps,
            loss_value=current_loss,
            validation_results=validation_results
        )
        
        return results

def log_system_info():
    """Log system information for debugging"""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # CPU info
    import multiprocessing
    logger.info(f"CPU count: {multiprocessing.cpu_count()} cores")
    
    # PyTorch device info
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}, Device count: {device_count}")
    if torch.cuda.is_available():
        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("Running on CPU only")

def evaluate_model(model, test_statements, test_labels):
    """
    Evaluate model performance on test set
    
    Parameters:
    model (SentenceTransformer): Fine-tuned model
    test_statements (list): List of test statements
    test_labels (list): List of test labels (0 or 1)
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set")
    start_time = time.time()
    
    # Generate embeddings for test statements
    embeddings = model.encode(test_statements, convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity
    from sklearn.preprocessing import normalize
    normalized_embeddings = normalize(embeddings)
    
    # Calculate similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    # Create prediction task
    # For each statement labeled as misinformation (1), 
    # it should be more similar to other misinformation than to factual statements
    y_true = []
    y_pred = []
    
    for i, label in enumerate(test_labels):
        if label == 1:  # This is misinformation
            # Get most similar statements to this one
            sims = similarity_matrix[i]
            # Remove self-similarity
            sims[i] = 0
            
            # Get indices of top 5 most similar statements
            top_indices = np.argsort(sims)[-5:]
            
            # For each top similar statement, check if it's correctly identified as misinformation
            for idx in top_indices:
                y_true.append(1 if test_labels[idx] == 1 else 0)
                # Predict based on similarity threshold instead of always 1
                y_pred.append(1 if sims[idx] > 0.5 else 0)  # Adjust threshold as needed
    
    # Calculate metrics
    if y_true:  # Make sure we have data to evaluate
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)
    else:
        accuracy = precision = recall = f1 = 0.0
    
    evaluation_time = time.time() - start_time
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Evaluation time: {evaluation_time:.2f} seconds")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'evaluation_time': evaluation_time
    }

print("About to define fine_tune_model function", file=sys.stderr)
def fine_tune_model():
    """
    Fine-tune a SentenceTransformer model using the vaping_data.csv dataset
    """
    logger.info("=" * 80)
    logger.info("Starting model fine-tuning process")
    logger.info("=" * 80)
    
    # Log system information
    log_system_info()
    
    # Timing the entire process
    total_start_time = time.time()
    
    # Set device to use CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available, using GPU")
    else:
        device = "cpu"
        logger.info("CUDA is not available, using CPU")
    
    # Check if output directory exists
    output_dir = os.path.join('models', 'fine_tuned', 'vaping_misinfo_model')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.info(f"Output directory already exists: {output_dir}")
    
    # Create visualizations directory
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f"Created/verified visualizations directory: {vis_dir}")
    
    # Check if data file exists
    data_file = os.path.join('data', 'vaping_data.csv')
    if not os.path.exists(data_file):
        logger.error(f"Error: Data file not found at {data_file}")
        return
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data_load_start = time.time()
    
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Data loaded successfully: {len(df)} rows, {df.columns.tolist()} columns")
        
        # Data statistics
        label_counts = df['label'].value_counts().to_dict()
        logger.info(f"Label distribution: {label_counts}")
        
        if len(df) == 0:
            logger.error("Error: Empty dataset")
            return
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Extract statements and labels
    statements = df['text'].tolist()
    labels = df['label'].tolist()
    
    # First split: 80% training, 20% temporary
    train_statements, temp_statements, train_labels, temp_labels = train_test_split(
        statements, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Second split: 50% of temporary for validation, 50% for testing (each 10% of original data)
    val_statements, test_statements, val_labels, test_labels = train_test_split(
        temp_statements, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Log the sizes of the sets
    logger.info(f"Data split into {len(train_statements)} training, {len(val_statements)} validation, and {len(test_statements)} testing samples")

    # Log label distributions for each set
    train_label_dist = Counter(train_labels)
    val_label_dist = Counter(val_labels)
    test_label_dist = Counter(test_labels)
    logger.info(f"Training label distribution: {train_label_dist}")
    logger.info(f"Validation label distribution: {val_label_dist}")
    logger.info(f"Testing label distribution: {test_label_dist}")

    logger.info(f"Data preparation completed in {time.time() - data_load_start:.2f} seconds")
    
    # Load base model
    logger.info("Loading base model...")
    model_load_start = time.time()
    
    model_name = 'all-mpnet-base-v2'
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Base model loaded successfully in {time.time() - model_load_start:.2f} seconds")
        logger.info(f"Model dimensions: {model.get_sentence_embedding_dimension()}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Prepare training examples
    logger.info("Preparing training examples for contrastive learning")
    prep_start = time.time()
    
    # In fine_tune_model(), under "Prepare training examples"
    train_examples = []
    for i, statement in enumerate(train_statements):
        label = train_labels[i]
        if label == 1:
            # Positive pair (misinformation with itself)
            train_examples.append(InputExample(texts=[statement, statement], label=1.0))
            # Negative pair (misinformation with one factual statement)
            factual_indices = [j for j, l in enumerate(train_labels) if l == 0]
            if factual_indices:  # Ensure there's at least one factual statement
                train_examples.append(InputExample(texts=[statement, train_statements[factual_indices[0]]], label=0.0))
    
    # Log training pair statistics
    positive_pairs = sum(1 for ex in train_examples if ex.label == 1.0)
    negative_pairs = sum(1 for ex in train_examples if ex.label == 0.0)
    logger.info(f"Created {len(train_examples)} training pairs: {positive_pairs} positive, {negative_pairs} negative")
    logger.info(f"Training example preparation completed in {time.time() - prep_start:.2f} seconds")
    
    # Create data loader
    batch_size = 16
    logger.info(f"Creating DataLoader with batch size {batch_size}")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Prepare validation examples similar to how you prepared training examples
    val_examples = []
    for i, statement in enumerate(val_statements):
        label = val_labels[i]
        if label == 1:
            # Create positive and negative pairs as you did for training
            val_examples.append(InputExample(texts=[statement, statement], label=1.0))
            # Negative pair (misinformation with one factual statement)
            factual_indices = [j for j, l in enumerate(val_labels) if l == 0]
            if factual_indices:
                val_examples.append(InputExample(texts=[statement, val_statements[factual_indices[0]]], label=0.0))

    # Initialize the Learning Curve Callback
    lcv_callback = LearningCurveCallback()
    timestamp = lcv_callback.timestamp  # Get the timestamp for consistent naming

    # Create the evaluator with our LCV callback
    validation_evaluator = EmbeddingSimilarityEvaluatorWithCallback(
        lcv_callback=lcv_callback,
        samples=val_examples, 
        name='vaping-validation'
    )
    
    # Modify your model.fit() call to include the evaluator
    evaluator_path = os.path.join('logs', 'validation_results')
    os.makedirs(evaluator_path, exist_ok=True)

    # Use the multiple negatives ranking loss
    logger.info("Setting up MultipleNegativesRankingLoss")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Configure training
    num_epochs = 7
    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of train data for warmup
    
    # Add progress tracking
    class TrainingProgressCallback:
        def __init__(self, lcv_callback):
            self.lcv_callback = lcv_callback
            self.loss_history = []
            self.current_epoch = 0
            self.global_step = 0
            
        def __call__(self, score, epoch, steps):
            """Called by the SentenceTransformer fit() method"""
            # Increment step counter
            self.global_step += 1
            
            # Track loss
            if hasattr(score, 'loss_value'):
                self.loss_history.append(score.loss_value)
                
                # Log every 10 steps
                if self.global_step % 10 == 0:
                    # Call LCV callback's on_batch_end
                    self.lcv_callback.on_batch_end(score)
            
            # Track epoch boundaries
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                # Compute average loss for the epoch
                if self.loss_history:
                    avg_loss = sum(self.loss_history) / len(self.loss_history)
                    # Reset loss history for next epoch
                    self.loss_history = []
                    
                    # Extract validation results if available
                    validation_results = extract_validation_results(evaluator_path, epoch)
                    
                    # Update LCV callback with epoch data
                    self.lcv_callback.on_epoch_end(
                        epoch_num=epoch,
                        steps=steps,
                        loss_value=avg_loss,
                        validation_results=validation_results
                    )
    
    # Initialize progress callback
    progress_callback = TrainingProgressCallback(lcv_callback)
    
    # Train the model
    logger.info(f"Starting fine-tuning for {num_epochs} epochs ({len(train_dataloader)} steps per epoch)")
    logger.info(f"Warmup steps: {warmup_steps}")
    train_start = time.time()
    
    try:
        # Train the model with the extended callback
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': 2e-5},
            evaluator=validation_evaluator,
            evaluation_steps=len(train_dataloader),  # Evaluate after each epoch
            output_path=evaluator_path,  # Save checkpoints based on evaluation
            show_progress_bar=True,
            callback=progress_callback  # Use our progress callback
        )
        
        train_duration = time.time() - train_start
        logger.info(f"Training completed in {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)")
            
        # Evaluate model on test set
        eval_metrics = evaluate_model(model, test_statements, test_labels)
        
        # Generate final visualization summary
        summary_path = create_visualization_summary(lcv_callback, eval_metrics, timestamp)
        
        # Save all metrics to JSON for potential later analysis
        metrics_file = lcv_callback.save_metrics_to_json(eval_metrics)
        
        # Save evaluation metrics
        timestamp_for_metrics = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join('logs', f'eval_metrics_{timestamp_for_metrics}.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_name} fine-tuned\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training data: {len(train_statements)} samples\n")
            f.write(f"Validation data: {len(val_statements)} samples\n")
            f.write(f"Testing data: {len(test_statements)} samples\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Training duration: {train_duration:.2f} seconds\n")
            f.write("\nMetrics:\n")
            for metric, value in eval_metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"\nVisualization Summary: {summary_path}\n")
            f.write(f"All Metrics JSON: {metrics_file}\n")
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        logger.info(f"Learning curve visualizations saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    save_start = time.time()
    
    try:
        model.save(output_dir)
        logger.info(f"Model saved successfully in {time.time() - save_start:.2f} seconds")
        
        # Log saved model files
        if os.path.exists(output_dir):
            logger.info("Saved model files:")
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path) / (1024 * 1024)  # size in MB
                    logger.info(f"  - {os.path.relpath(full_path, output_dir)}: {file_size:.2f} MB")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    total_duration = time.time() - total_start_time
    logger.info(f"Fine-tuning process completed in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info("=" * 80)

if __name__ == "__main__":
    fine_tune_model()